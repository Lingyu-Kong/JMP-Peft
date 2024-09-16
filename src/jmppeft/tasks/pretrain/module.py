import copy
import math
from collections.abc import Callable, Iterable
from functools import cache, partial
from logging import getLogger
from typing import Annotated, Any, Literal, TypeAlias, cast

import nshconfig as C
import nshtrainer as nt
import nshtrainer.ll as ll
import nshutils.typecheck as tc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from einops import pack, rearrange, reduce
from jmppeft.modules.torch_scatter_polyfill import scatter
from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from nshtrainer.data.balanced_batch_sampler import BalancedBatchSampler
from nshtrainer.model.config import LightningTrainerKwargs
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.utils import dropout_edge
from torchmetrics import SumMetric
from typing_extensions import TypeVar, assert_never, override

from ...datasets.pretrain_lmdb import PretrainDatasetConfig as PretrainDatasetConfigBase
from ...datasets.pretrain_lmdb import PretrainLmdbDataset
from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import BackboneConfig as GOCBackboneConfig
from ...models.m3gnet.config import BackboneConfig as M3GNetBackboneConfig
from ...models.m3gnet.backbone import M3GNet, M3GNetBackboneOutput
from ...models.m3gnet.modules.message_passing import MainBlock
from ...models.gemnet.layers.base_layers import ScaledSiLU
from ...models.graphormer.config import Graphormer3DConfig
from ...models.torchmdnet.config import TorchMDNetBackboneConfig
from ...modules import transforms as T
from ...modules.dataset import dataset_transform as DT
from ...modules.dataset.common import (
    CommonDatasetConfig,
    DatasetSampleRatioConfig,
    wrap_common_dataset,
)
from ...modules.dataset.concat_dataset import MTDatasetConfig, MTSampledDataset
from ...modules.ema import EMAConfig
from ...modules.metrics import FMMetrics
from ...modules.scheduler.linear_warmup_cosine_annealing import (
    LinearWarmupCosineAnnealingLR,
)
from ...modules.transforms.normalize import NormalizationConfig
from ...utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from ..config import (
    EmbeddingConfig,
    OptimizerConfig,
    OutputConfig,
    optimizer_from_config,
)

log = getLogger(__name__)


class LinearWarmupCosineAnnealingSchedulerConfig(C.Config):
    name: Literal["linear_warmup_cosine_annealing"] = "linear_warmup_cosine_annealing"

    warmup_steps: int = 0
    max_steps: int | None = None
    max_epochs: int | None = None
    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1


LRSchedulerConfig: TypeAlias = Annotated[
    LinearWarmupCosineAnnealingSchedulerConfig, C.Field(discriminator="name")
]


class PretrainDatasetConfig(PretrainDatasetConfigBase, CommonDatasetConfig):
    pass


class TaskConfig(C.Config):
    name: str
    """Name of the task."""

    train_dataset: PretrainDatasetConfig
    """Train dataset configuration."""

    val_dataset: PretrainDatasetConfig
    """Validation dataset configuration."""

    node_energy_reduction: Literal["sum", "mean"] = "sum"
    """How to reduce the node energy scalar contributions (to get the total energy)."""

    additional_units: list[str] = []
    """Additional units to log for this task."""

    energy_loss_scale: float = 1.0
    """Scale factor for the energy loss."""
    force_loss_scale: float = 1.0
    """Scale factor for the force loss."""

    normalization: dict[str, NormalizationConfig] | None = None
    """
    Normalization to apply to the target values.
    Each key is the name of the target value
    and the value is a dict with the mean and std.
    """


BackboneConfig: TypeAlias = Annotated[
    GOCBackboneConfig | Graphormer3DConfig | TorchMDNetBackboneConfig | M3GNetBackboneConfig,
    C.Field(discriminator="name"),
]


class FSDPConfig(C.Config):
    gradient_checkpointing: bool
    """Whether to use gradient checkpointing."""

    cpu_offload: bool = False
    """Whether to offload the optimizer state to the CPU."""

    sharding_strategy: Literal[
        "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"
    ] = "HYBRID_SHARD"
    """Sharding strategy to use."""


class PretrainConfig(nt.BaseConfig):
    optimizer: OptimizerConfig
    """Optimizer to use."""
    lr_scheduler: LRSchedulerConfig | None = None
    """Learning rate scheduler configuration. If None, no learning rate scheduler is used."""

    dropout: float | None = None
    """The dropout rate to use in GemNet."""
    edge_dropout: float | None = None
    """The percentage of edges to drop. If None, no edges are dropped."""

    embedding: C.AllowMissing[EmbeddingConfig] = C.Config.MISSING
    """Configuration for the embedding layer."""
    backbone: BackboneConfig
    """Configuration for the backbone."""
    output: OutputConfig = OutputConfig(num_mlps=5)
    """Configuration for the output head."""

    batch_size: int
    """Batch size to use."""
    eval_batch_size: int | None = None
    """Batch size to use for evaluation. If None, use the same as batch_size."""
    num_workers: int
    """Number of workers to use for data loading."""
    pin_memory: bool = True
    """Whether to use pin memory for data loading."""

    shuffle_train: bool = True
    """Should we shuffle the training dataset?"""

    shuffle_val: bool = False
    """Should we shuffle the validation dataset?"""

    @property
    def activation_cls(self):
        match self.backbone.activation:
            case "scaled_silu" | "scaled_swish":
                return ScaledSiLU
            case "silu" | "swish":
                return nn.SiLU
            case "gelu":
                return nn.GELU
            case None:
                return nn.Identity
            case _:
                raise NotImplementedError(
                    f"Activation {self.backbone.activation=} is not implemented"
                )

    log_task_losses: bool = True
    """Log the loss for each task."""
    log_task_steps_and_epochs: bool = True
    """Log the number of steps and epochs for each task."""

    tasks: list[TaskConfig]
    """List of datasets/tasks to train on."""
    mt_dataset: MTDatasetConfig = MTDatasetConfig(
        balanced=True,
        strict=True,
    )
    """Configuration for the multi-task dataset."""

    exclude_keys: list[str] = [
        "id",  # only oc20,oc22 have this
        "fid",  # only oc20,oc22 have this
        "cell_offsets",  # only oc20 has this
        "edge_index",  # only oc20 has this
        "absolute_idx",  # only ani has this
        "target_pos",  # only ani has this
        "ref_energy",  # only ani/geom have this
        "pbc",  # only ani/transition1x have this
        "oc22",  # only oc22 has this
        "name",
    ]
    """Keys to exclude when creating a batch from a data list."""

    train_on_free_atoms_only: bool = False
    """Train only on free atoms."""

    eval_on_free_atoms_only: bool = True
    """Evaluate only on free atoms."""

    energy_loss_reduction: Literal["sum", "mean"] = "mean"
    """How to reduce the energy loss. "sum" or "mean"."""
    force_loss_reduction: Literal["sum", "mean"] = "mean"
    """How to reduce the force loss. "sum" or "mean"."""

    structurewise_loss_reduction: bool = True
    """Use the proposed structurewise loss (from the paper) reduction for the force loss."""

    ema: EMAConfig | None = None
    """Configuration for the exponential moving average."""

    fsdp: FSDPConfig | None = None
    """Configuration for the Fully Sharded Data Parallel (FSDP) strategy."""

    perf_metrics: bool = False

    disable_metrics: bool = False

    gradient_checkpointing: bool = False

    multi_head_loss_trick: bool = False

    global_train_sample_ratio: DatasetSampleRatioConfig | None = None
    global_val_sample_ratio: DatasetSampleRatioConfig | None = None

    generate_graphs_on_gpu: bool = False
    """Generate graphs on the GPU."""

    @override
    def __post_init__(self):
        super().__post_init__()

        assert (
            not self.trainer.use_distributed_sampler
        ), "config.trainer.use_distributed_sampler must be False"

        match self.backbone:
            case Graphormer3DConfig() as config:
                config.activation_dropout = config.dropout = self.dropout or 0.0
                config.attention_dropout = self.edge_dropout or 0.0
            case GOCBackboneConfig() as config:
                config.dropout = self.dropout
                config.edge_dropout = self.edge_dropout
            case M3GNetBackboneConfig() as config:
                pass
            case TorchMDNetBackboneConfig() as config:
                if self.dropout or self.edge_dropout:
                    raise NotImplementedError(
                        "Dropout/edge_dropout not implemented for TorchMDNetBackboneConfig"
                    )

                assert (
                    "edge_index" not in self.exclude_keys
                ), "edge_index must be not in exclude_keys for TorchMDNetBackboneConfig"
            case _:
                assert_never(self.backbone)

        if self.embedding is self.MISSING:
            info = self.backbone.atom_embedding_table_info()
            self.embedding = EmbeddingConfig(
                num_elements=info["num_embeddings"],
                embedding_size=info["embedding_dim"],
            )

        # If a size ratio is given, apply it here.
        if (ratio := self.global_train_sample_ratio) is not None:
            for task_config in self.tasks:
                dataset_config = task_config.train_dataset

                sample_ratio = ratio.sample_ratio
                seed = ratio.seed

                # If the task itself also has a ratio, multiply them
                if (task_sample_ratio := dataset_config.sample_ratio) is not None:
                    sample_ratio = sample_ratio * task_sample_ratio.sample_ratio

                    # Somehow combine the two seed values if they're not the same
                    if seed != task_sample_ratio.seed:
                        seed = hash((seed, task_sample_ratio.seed))

                    log.critical(
                        f"Both global (={ratio.sample_ratio}) and task (={task_sample_ratio.sample_ratio}) "
                        f"sample ratios are set for {task_config.name}_train. "
                        f"Multiplying the two together for a final sample ratio of {sample_ratio}. "
                        f"Seeds ({ratio.seed}, {task_sample_ratio.seed}) are combined to {seed}."
                    )

                dataset_config.sample_ratio = DatasetSampleRatioConfig(
                    sample_ratio=sample_ratio, seed=seed
                )
            self.global_train_sample_ratio = None

        if (ratio := self.global_val_sample_ratio) is not None:
            for task_config in self.tasks:
                dataset_config = task_config.val_dataset

                sample_ratio = ratio.sample_ratio
                seed = ratio.seed

                # If the task itself also has a ratio, multiply them
                if (task_sample_ratio := dataset_config.sample_ratio) is not None:
                    sample_ratio = sample_ratio * task_sample_ratio.sample_ratio

                    # Somehow combine the two seed values if they're not the same
                    if seed != task_sample_ratio.seed:
                        seed = hash((seed, task_sample_ratio.seed))

                    log.critical(
                        f"Both global (={ratio.sample_ratio}) and task (={task_sample_ratio.sample_ratio}) "
                        f"sample ratios are set for {task_config.name}_val. "
                        f"Multiplying the two together for a final sample ratio of {sample_ratio}. "
                        f"Seeds ({ratio.seed}, {task_sample_ratio.seed}) are combined to {seed}."
                    )

                dataset_config.sample_ratio = DatasetSampleRatioConfig(
                    sample_ratio=sample_ratio, seed=seed
                )
            self.global_val_sample_ratio = None

        if self.gradient_checkpointing:
            assert not self.fsdp, (
                "Gradient checkpointing and FSDP are not compatible.\n"
                "If you want to use gradient checkpointing with FSDP, set `fsdp.gradient_checkpointing=True`."
            )


Data: TypeAlias = Any


class Embedding(nt.Base[PretrainConfig], nn.Module):
    @override
    def __init__(self, hparams: PretrainConfig):
        super().__init__(hparams)

        self.atom_embedding = nn.Embedding(
            num_embeddings=self.config.embedding.num_elements,
            embedding_dim=self.config.embedding.embedding_size,
        )

    @override
    def forward(self, data: Data):
        atomic_numbers = data.atomic_numbers - 1
        x = self.atom_embedding(atomic_numbers)
        return x


class Output(nt.Base[PretrainConfig], nn.Module):
    @override
    def __init__(self, hparams: PretrainConfig):
        super().__init__(hparams)

        def dims(
            emb_size: int,
            *,
            num_targets: int = self.config.backbone.num_targets,
            num_mlps: int = self.config.output.num_mlps,
        ):
            return ([emb_size] * num_mlps) + [num_targets]

        self.out_energy = nt.nn.TypedModuleList(
            [
                nt.nn.MLP(
                    dims(self.config.backbone.emb_size_atom),
                    activation=self.config.activation_cls,
                )
                for _ in self.config.tasks
            ]
        )
        self.out_forces = nt.nn.TypedModuleList(
            [
                nt.nn.MLP(
                    dims(self.config.backbone.emb_size_edge),
                    activation=self.config.activation_cls,
                )
                for _ in self.config.tasks
            ]
        )

    @override
    def forward(self, data: Data, backbone_out: GOCBackboneOutput | M3GNetBackboneOutput):
        energy = backbone_out["energy"]
        forces = backbone_out["forces"]
        V_st = backbone_out["V_st"]
        idx_t = backbone_out["idx_t"]

        batch: torch.Tensor = data.batch
        n_molecules = int(torch.max(batch).item() + 1)
        n_atoms = data.atomic_numbers.shape[0]

        energy_list: list[torch.Tensor] = []
        forces_list: list[torch.Tensor] = []

        for energy_mlp, forces_mlp, task in zip(
            self.out_energy, self.out_forces, self.config.tasks
        ):
            E_t = energy_mlp(energy)  # (n_atoms, 1)
            E_t = scatter(
                E_t,
                batch,
                dim=0,
                dim_size=n_molecules,
                reduce=task.node_energy_reduction,
            )
            energy_list.append(E_t)  # (bsz, 1)

            F_st = forces_mlp(forces)  # (n_edges, 1)
            F_st = F_st * V_st  # (n_edges, 3)
            F_t = scatter(F_st, idx_t, dim=0, dim_size=n_atoms, reduce="sum")
            forces_list.append(F_t)  # (n_atoms, 3)

        E, _ = pack(energy_list, "bsz *")
        F, _ = pack(forces_list, "n_atoms p *")

        return E, F


class PerfMetrics(nn.Module):
    def __init__(self):
        super().__init__()

        self.total_num_systems = torchmetrics.SumMetric()
        self.total_num_atoms = torchmetrics.SumMetric()

    @override
    def forward(self, batch: Batch):
        self.total_num_systems(batch.y.shape[0])
        self.total_num_atoms(batch.atomic_numbers.shape[0])

        return {
            "total_num_systems": self.total_num_systems,
            "total_num_atoms": self.total_num_atoms,
        }


TItem = TypeVar("TItem", infer_variance=True)


def aslist(x: TItem | Iterable[TItem] | None) -> list[TItem]:
    if x is None:
        return []

    if not isinstance(x, Iterable):
        return [x]
    return list(x)


TReturn = TypeVar("TReturn", infer_variance=True)


def foreach(
    f: Callable[[TItem], TReturn], xs: TItem | Iterable[TItem] | None
) -> list[TReturn]:
    xs_list = aslist(xs)
    return [f(x) for x in xs_list]


GOCOutput = Output


class PretrainModel(nt.LightningModuleBase[PretrainConfig]):
    def _ckpt_layers(self) -> set[type[nn.Module]]:
        layers = set[type[nn.Module]]()
        match self.config.backbone:
            case Graphormer3DConfig():
                from ...models.graphormer.model import Graphormer3DEncoderLayer

                layers.update(
                    {
                        # nn.Embedding,
                        Graphormer3DEncoderLayer,
                        # Graphormer3D,
                        # GraphormerOutput,
                    }
                )
            case GOCBackboneConfig():
                from ...models.gemnet.backbone import InteractionBlock, OutputBlock

                layers.update(
                    {
                        # nn.Embedding,
                        # Bases,
                        InteractionBlock,
                        # PairInteraction,
                        # QuadrupletInteraction,
                        # TripletInteraction,
                        OutputBlock,
                        # GemNetOCBackbone,
                        # GOCOutput,
                    }
                )
            case M3GNetBackboneConfig():
                layers.update(
                    {
                        MainBlock,
                    }
                )
            case TorchMDNetBackboneConfig():
                from ...models.torchmdnet.backbone import EquivariantMultiHeadAttention

                # NeighborEmbedding,
                # TorchMD_ET,

                layers.update(
                    {
                        # nn.Embedding,
                        # NeighborEmbedding,
                        EquivariantMultiHeadAttention,
                        # TorchMD_ET,
                        # TorchMDOutput,
                    }
                )
            case _:
                assert_never(self.config.backbone)

        return layers

    @override
    def setup(self, stage: str):
        super().setup(stage)

        if self.config.gradient_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )

            layer_cls_tuple = tuple(self._ckpt_layers())
            apply_activation_checkpointing(
                self,
                checkpoint_wrapper_fn=partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                check_fn=lambda layer: isinstance(layer, layer_cls_tuple),
            )
            log.critical("Applied gradient checkpointing to the model.")

    def fsdp_trainer_kwargs(self) -> LightningTrainerKwargs:
        if not (config := self.config.fsdp):
            return {}

        from lightning.pytorch.strategies.fsdp import FSDPStrategy
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

        layers = self._ckpt_layers()
        strategy = FSDPStrategy(
            cpu_offload=CPUOffload(offload_params=True) if config.cpu_offload else None,
            state_dict_type="sharded",
            auto_wrap_policy=layers,
            activation_checkpointing_policy=layers
            if config.gradient_checkpointing
            else None,
            sharding_strategy=config.sharding_strategy,
        )

        return {"strategy": strategy}

    @classmethod
    @override
    def config_cls(cls):
        return PretrainConfig

    def _construct_backbone(self):
        match self.config.backbone:
            case Graphormer3DConfig():
                from ...models.graphormer import Graphormer3D

                return Graphormer3D(self.config.backbone)
            case GOCBackboneConfig():
                return GemNetOCBackbone(
                    self.config.backbone, **dict(self.config.backbone)
                )
            case M3GNetBackboneConfig():
                return M3GNet(**dict(self.config.backbone))
            case TorchMDNetBackboneConfig():
                return self.config.backbone.create_backbone()
            case _:
                assert_never(self.config.backbone)

    def _construct_output(self):
        match self.config.backbone:
            case Graphormer3DConfig():
                from ...models.graphormer.model import Output as GraphormerOutput

                return GraphormerOutput(
                    self.config.output,
                    self.config.tasks,
                    self.config.backbone,
                )
            case TorchMDNetBackboneConfig() as config:
                return config.create_output_head(self.config.tasks)
            case _:
                return Output(self.config)

    @override
    def __init__(self, hparams: PretrainConfig):
        super().__init__(hparams)

        # Set up callbacks
        if (ema := self.config.ema) is not None:
            self.register_callback(lambda: ema.construct_callback())

        # Set up the model
        if not self.config.backbone.handles_atom_embedding():
            self.embedding = Embedding(self.config)
        self.backbone = self._construct_backbone()
        self.output = self._construct_output()

        # Set up the metrics
        self.train_metrics = FMMetrics(
            {
                task.name: {"idx": idx, "additional_units": task.additional_units}
                for idx, task in enumerate(self.config.tasks)
            },
            denormalize=any(task.normalization for task in self.config.tasks),
            free_atoms_only=self.config.eval_on_free_atoms_only,
        )
        self.val_metrics = FMMetrics(
            {
                task.name: {"idx": idx, "additional_units": task.additional_units}
                for idx, task in enumerate(self.config.tasks)
            },
            denormalize=any(task.normalization for task in self.config.tasks),
            free_atoms_only=self.config.eval_on_free_atoms_only,
        )
        if self.config.perf_metrics:
            self.train_perf_metrics = PerfMetrics()

        # GemNet-OC re-uses some parameters at every layer.
        # We need to make sure that these parameters' gradients are
        # downscaled by the number of layers so that the gradients
        # are not too large.
        if shared_parameters := getattr(self.backbone, "shared_parameters", None):
            self.register_shared_parameters(shared_parameters)

        self._train_dataset_sizes: list[int] | None = None
        if self.config.log_task_steps_and_epochs:
            task_steps: dict[str, SumMetric] = {}
            for task in self.config.tasks:
                metric = SumMetric()
                metric.persistent(True)
                task_steps[task.name] = metric
            self.task_steps = nt.nn.TypedModuleDict(task_steps)

        if self.config.multi_head_loss_trick:
            self.automatic_optimization = False

    def backbone_state_dict(self):
        state_dict = {
            "backbone": self.backbone.state_dict(),
        }
        if not self.config.backbone.handles_atom_embedding():
            state_dict["embedding"] = self.embedding.atom_embedding.state_dict()

        return state_dict

    @override
    def on_train_batch_start(self, batch: Data, batch_idx: int):
        if not self.config.log_task_steps_and_epochs:
            return

        assert self._train_dataset_sizes

        task_mask = batch.task_mask  # (b, t)
        task_idx = reduce(task_mask, "b t -> t", "sum")  # (t,)
        for idx, task in enumerate(self.config.tasks):
            metric = self.task_steps[task.name]
            metric(task_idx[idx])

            step = metric.compute()
            self.log(f"train/{task.name}/step", step)

            epoch = step / self._train_dataset_sizes[idx]
            self.log(f"train/{task.name}/epoch", epoch)

    @override
    def forward(self, batch: Data):
        if self.config.generate_graphs_on_gpu:
            batch = self._generate_graphs_goc(
                batch,
                cutoffs=Cutoffs.from_constant(12.0),
                max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
                pbc=True,
                training=self.training,
            )

        if not self.config.backbone.handles_atom_embedding():
            h = self.embedding(batch)
            out = self.backbone(batch, h=h)
        else:
            out = self.backbone(batch)

        return self.output(batch, out)  # (n h), (n p h)
    
    def get_node_features(self, batch: Data):
        if self.config.generate_graphs_on_gpu:
            batch = self._generate_graphs_goc(
                batch,
                cutoffs=Cutoffs.from_constant(12.0),
                max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
                pbc=True,
                training=self.training,
            )

        if not self.config.backbone.handles_atom_embedding():
            h = self.embedding(batch)
            out = self.backbone(batch, h=h)
        else:
            out = self.backbone(batch)

        return out["h"]

    def _task_idx_onehot(self, task_idx: int):
        return F.one_hot(
            torch.tensor([task_idx], device=self.device, dtype=torch.long),
            num_classes=len(self.config.tasks),
        ).bool()

    def _force_loss(
        self, batch: Data, forces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.debug:
            assert forces.shape == batch.force.shape

        pred: torch.Tensor = rearrange(forces, "n p t -> n t p")
        target: torch.Tensor = rearrange(batch.force, "n p t -> n t p")

        mask = batch.task_mask  # b t
        mask = mask[batch.batch]  # n t
        if self.config.train_on_free_atoms_only:
            mask = mask & rearrange(~batch.fixed, "n -> n 1")

        force_loss = F.pairwise_distance(pred, target, p=2.0)  # (n, t)

        if (scale := getattr(batch, "force_scale", None)) is not None:
            # force_loss_scale: (b,)
            scale = scale[batch.batch]  # (n, t)
            if self.config.train_on_free_atoms_only:
                scale = scale[~batch.fixed]
            force_loss = force_loss * scale

        if (scale := getattr(batch, "force_scale_node", None)) is not None:
            # force_scale_node: (n, t)
            if self.config.train_on_free_atoms_only:
                scale = scale[~batch.fixed]
            force_loss = force_loss * scale

        force_loss = force_loss.masked_fill(~mask, 0.0)

        if self.config.log_task_losses:
            with torch.no_grad():
                for task_idx, task in enumerate(self.config.tasks):
                    task_mask = mask & self._task_idx_onehot(task_idx)
                    task_force_loss = force_loss.masked_fill(~task_mask, 0.0)
                    self.log(
                        f"{task.name}/force_loss",
                        self._reduce_loss(
                            task_force_loss,
                            task_mask,
                            reduction=self.config.force_loss_reduction,
                        ),
                    )

        # force_loss = self._reduce_force_loss(force_loss, mask)
        return force_loss, mask

    def _energy_loss(
        self, batch: Data, energy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch.task_mask  # (b, h)

        energy_loss = F.l1_loss(energy, batch.y, reduction="none")  # (b, num_tasks)

        if (scale := getattr(batch, "y_scale", None)) is not None:
            energy_loss = energy_loss * scale  # (b, t)

        energy_loss = energy_loss.masked_fill(~mask, 0.0)

        if self.config.log_task_losses:
            with torch.no_grad():
                for task_idx, task in enumerate(self.config.tasks):
                    task_mask = mask & self._task_idx_onehot(task_idx)
                    task_energy_loss = energy_loss.masked_fill(~task_mask, 0.0)
                    self.log(
                        f"{task.name}/energy_loss",
                        self._reduce_loss(
                            task_energy_loss,
                            task_mask,
                            reduction=self.config.energy_loss_reduction,
                        ),
                    )

        return energy_loss, mask

    @staticmethod
    def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b = b.masked_fill(b == 0.0, 1.0)
        return a / b

    def _reduce_loss(
        self,
        loss: torch.Tensor,
        mask: torch.Tensor,
        reduction: Literal["sum", "mean"],
    ):
        match reduction:
            case "sum":
                loss = reduce(loss, "... -> ", "sum")
            case "mean":
                # loss = reduce(loss, "... -> ", "sum") / reduce(mask, "... -> ", "sum")
                loss = self._safe_divide(
                    reduce(loss, "... -> ", "sum"),
                    reduce(mask, "... -> ", "sum"),
                )
            case _:
                raise ValueError(f"Unknown redution: {reduction}")

        return loss

    def compute_losses(self, batch: Data, energy: torch.Tensor, forces: torch.Tensor):
        # Compute the energy loss
        energy_loss, energy_loss_mask = self._energy_loss(
            batch, energy
        )  # (b, t), (b, t)
        energy_loss = self._reduce_loss(
            energy_loss, energy_loss_mask, reduction=self.config.energy_loss_reduction
        )
        self.log("energy_loss", energy_loss)

        # Compute the force loss
        force_loss, force_loss_mask = self._force_loss(batch, forces)
        if self.config.structurewise_loss_reduction:
            # Compute the per-structure force loss
            force_loss = scatter(force_loss, batch.batch, dim=0, reduce="sum")  # (b, t)
            force_loss_mask_natoms = scatter(
                force_loss_mask.float(), batch.batch, dim=0, reduce="sum"
            )  # (b, t)
            force_loss = self._safe_divide(force_loss, force_loss_mask_natoms)  # (b, t)
            force_loss_mask = force_loss_mask_natoms > 0.0  # (b, t)
        force_loss = self._reduce_loss(
            force_loss, force_loss_mask, reduction=self.config.force_loss_reduction
        )
        self.log("force_loss", force_loss)

        # Combine the losses
        loss = energy_loss + force_loss
        self.log("loss", loss)

        return loss

    def _training_step_multi_head_trick(self, data: Data, batch_idx: int):
        # Ref: https://arxiv.org/pdf/2404.19737
        assert self.config.multi_head_loss_trick
        assert self.config.structurewise_loss_reduction
        assert not self.automatic_optimization
        assert not self.config.train_on_free_atoms_only

        foreach(lambda opt: opt.zero_grad(), self.optimizers())

        match self.config.backbone:
            case GOCBackboneConfig():
                raise NotImplementedError
            case M3GNetBackboneConfig():
                raise NotImplementedError
            case Graphormer3DConfig():
                from ...models.graphormer.model import (
                    DenseData,
                    GraphormerBackboneOutput,
                )
                from ...models.graphormer.model import (
                    Output as GraphormerOutput,
                )

                assert isinstance(output := self.output, GraphormerOutput)

                backbone_out = cast(GraphormerBackboneOutput, self.backbone(data))
                z = backbone_out["output"]
                d = z.detach()
                d.requires_grad = True
                backbone_out["output"] = d

                dense_data: DenseData = data.jmp_dense_data

                dense_padding_mask = dense_data["atoms"].eq(0)
                dense_output_mask = dense_data["real_mask"] & ~dense_padding_mask

                energy_list: list[tc.Float[torch.Tensor, "b"]] = []
                for task_idx, (out_energy, task) in enumerate(
                    zip(output.out_energy, output.task_configs)
                ):
                    energy_loss = self._multi_head_trick_energy_loss(
                        data,
                        output,
                        backbone_out,
                        dense_output_mask,
                        energy_list,
                        task_idx,
                        out_energy,
                        task,
                    )

                    self.manual_backward(energy_loss)

                energies, _ = pack(energy_list, "bsz *")
                del energy_list

                forces_list: list[tc.Float[torch.Tensor, "n 3"]] = []
                for task_idx, (out_forces, task) in enumerate(
                    zip(output.out_forces, output.task_configs)
                ):
                    force_loss = self._multi_head_trick_force_loss(
                        data,
                        output,
                        backbone_out,
                        dense_output_mask,
                        task_idx,
                        task,
                        forces_list,
                        out_forces,
                    )

                    self.manual_backward(force_loss)

                forces, _ = pack(forces_list, "n_atoms p *")
                del forces_list

                self.manual_backward(z, gradient=d.grad)

            case TorchMDNetBackboneConfig():
                raise NotImplementedError
            case _:
                assert_never(self.config.backbone)

        foreach(lambda opt: opt.step(), self.optimizers())
        foreach(lambda lr_scheduler: lr_scheduler.step(), self.lr_schedulers())

        with torch.no_grad():
            self.log_dict(self.train_metrics(data, energy=energies, forces=forces))
            if self.config.perf_metrics:
                # We only log the metrics if we're updating the logs because we need to
                # synchronize the logs across all processes.
                self.log_dict(self.train_perf_metrics(data), sync_dist=True)

    def _multi_head_trick_force_loss(
        self,
        data,
        output,
        backbone_out,
        dense_output_mask,
        task_idx,
        task,
        forces_list,
        out_forces,
    ):
        forces = output._compute_forces(
            out_forces,
            backbone_out,
            dense_output_mask,
        )
        tc.tassert(tc.Float[torch.Tensor, "n 3"], forces)
        forces_list.append(forces.detach())  # keep for metrics

        # Compute the force loss for this task
        task_mask = data.task_mask[:, task_idx]
        tc.tassert(tc.Bool[torch.Tensor, "b"], task_mask)
        task_mask = task_mask[data.batch]
        tc.tassert(tc.Bool[torch.Tensor, "n"], task_mask)

        force_target = data.force[:, :, task_idx]
        tc.tassert(tc.Float[torch.Tensor, "n 3"], force_target)

        force_loss = F.pairwise_distance(forces, force_target, p=2.0)
        force_loss = force_loss.masked_fill(~task_mask, 0.0)
        tc.tassert(tc.Float[torch.Tensor, "n"], force_loss)

        if self.config.structurewise_loss_reduction:
            # Compute the per-structure force loss
            force_loss = scatter(force_loss, data.batch, dim=0, reduce="sum")
            tc.tassert(tc.Float[torch.Tensor, "b"], force_loss)
            task_mask_natoms = scatter(
                task_mask.float(), data.batch, dim=0, reduce="sum"
            )
            tc.tassert(tc.Float[torch.Tensor, "b"], task_mask_natoms)
            force_loss = self._safe_divide(force_loss, task_mask_natoms)
            tc.tassert(tc.Float[torch.Tensor, "b"], force_loss)

            task_mask = task_mask_natoms > 0.0  # (b, t)
            tc.tassert(tc.Bool[torch.Tensor, "b"], task_mask)

        force_loss = self._reduce_loss(
            force_loss,
            task_mask,
            reduction=self.config.force_loss_reduction,
        )
        force_loss = force_loss * task.force_loss_scale
        return force_loss

    def _multi_head_trick_energy_loss(
        self,
        data,
        output,
        backbone_out,
        dense_output_mask,
        energy_list,
        task_idx,
        out_energy,
        task,
    ):
        energy_pred = output._compute_energy(
            out_energy,
            backbone_out,
            dense_output_mask,
        )
        tc.tassert(tc.Float[torch.Tensor, "b"], energy_pred)
        energy_list.append(energy_pred.detach())  # keep for metrics

        # Compute the energy loss for this task
        task_mask = data.task_mask[:, task_idx]
        tc.tassert(tc.Bool[torch.Tensor, "b"], task_mask)

        energy_target = data.y[:, task_idx]
        tc.tassert(tc.Float[torch.Tensor, "b"], energy_target)

        energy_loss = F.l1_loss(energy_pred, energy_target, reduction="none")
        energy_loss = energy_loss.masked_fill(~task_mask, 0.0)
        energy_loss = self._reduce_loss(
            energy_loss,
            task_mask,
            reduction=self.config.energy_loss_reduction,
        )
        energy_loss = energy_loss * task.energy_loss_scale
        return energy_loss

    def _training_step_regular(self, batch: Data, batch_idx: int):
        energy, forces = self(batch)

        loss = self.compute_losses(batch, energy=energy, forces=forces)
        if not self.config.disable_metrics:
            with torch.no_grad():
                self.log_dict(self.train_metrics(batch, energy=energy, forces=forces))

        return loss

    @override
    def training_step(self, batch: Data, batch_idx: int):
        with self.log_context(prefix="train/"):
            if not self.config.multi_head_loss_trick:
                return self._training_step_regular(batch, batch_idx)
            else:
                return self._training_step_multi_head_trick(batch, batch_idx)

    @override
    def validation_step(self, batch: Data, batch_idx: int):
        with self.log_context(prefix="val/"):
            energy, forces = self(batch)

            metrics = self.val_metrics(batch, energy=energy, forces=forces)
            self.log_dict(metrics)

    def configure_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> LRSchedulerConfigType | None:
        match self.config.lr_scheduler:
            case None:
                return None
            case LinearWarmupCosineAnnealingSchedulerConfig() as config:
                if not (max_steps := config.max_steps):
                    if max_epochs := config.max_epochs:
                        _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                        num_steps_per_epoch = math.ceil(
                            self.trainer.num_training_batches
                            / self.trainer.accumulate_grad_batches
                        )
                        max_steps = max_epochs * num_steps_per_epoch
                    else:
                        max_steps = self.trainer.estimated_stepping_batches
                        assert math.isfinite(max_steps), f"{max_steps=} is not finite"
                        max_steps = int(max_steps)

                    log.critical(f"Setting {max_steps=} by default.")

                optim_lr = float(optimizer.param_groups[0]["lr"])
                min_lr = optim_lr * config.min_lr_factor
                warmup_start_lr = optim_lr * config.warmup_start_lr_factor
                lr_scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=config.warmup_steps,
                    max_epochs=max_steps,
                    warmup_start_lr=warmup_start_lr,
                    eta_min=min_lr,
                    last_epoch=config.last_step,
                )
                return {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "strict": True,  # type: ignore
                }

            case _:
                assert_never(self.config.lr_scheduler)

    @override
    def configure_optimizers(self):
        optimizer = optimizer_from_config([(self.config.optimizer, self.parameters())])
        out: OptimizerLRSchedulerConfig = {"optimizer": optimizer}
        if (lr_scheduler := self.configure_lr_scheduler(optimizer)) is not None:
            out["lr_scheduler"] = lr_scheduler

        return out

    def _task_dataset(self, task: TaskConfig, training: bool):
        config = task.val_dataset if not training else task.train_dataset
        dataset = PretrainLmdbDataset(config)
        dataset = wrap_common_dataset(dataset, config)

        # Apply data transform to the dataset
        if (transform := getattr(self, f"{task.name}_transform")) is None:
            raise ValueError(f"Transform not defined for {task.name}")
        transform = cast(Callable[[Data], Data], partial(transform, training=training))

        # Apply normalization to the dataset
        if task.normalization:
            log.info(f"Normalizing {task.name} with {task.normalization}")
            transform = T.compose([transform, T.normalize(task.normalization)])

        dataset = DT.transform(dataset, transform)

        return dataset

    def _construct_fm_datasets(self, training: bool):
        datasets = []
        for task in self.config.tasks:
            datasets.append(self._task_dataset(task, training=training))
        return datasets

    @cache
    def train_dataset(self):
        datasets = self._construct_fm_datasets(training=True)
        self._train_dataset_sizes = [len(d) for d in datasets]
        # if self.config.log_task_steps_and_epochs:
        dataset = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=False,
            num_tasks=len(self.config.tasks),
        )
        dataset = DT.transform(dataset, self.train_data_transform)
        return dataset

    def representative_batch_for_testing(self, *, n: int, start_index: int = 0):
        dataset = self.train_dataset()
        data_list = dataset.representative_batch_for_testing(
            n=n, start_index=start_index
        )
        data_list = [self.train_data_transform(data) for data in data_list]
        return data_list

    @cache
    def val_dataset(self):
        datasets = self._construct_fm_datasets(training=False)
        dataset = MTSampledDataset(
            datasets,
            self.config.mt_dataset,
            ignore_balancing=True,
            num_tasks=len(self.config.tasks),
        )
        dataset = DT.transform(dataset, self.val_data_transform)
        return dataset

    def collate_fn_gnn(self, data_list: list[Data]):
        return Batch.from_data_list(data_list, exclude_keys=self.config.exclude_keys)

    def collate_fn(self, data_list: list[Data]):
        match self.config.backbone:
            case Graphormer3DConfig():
                from ...models.graphormer.types import collate_fn

                return collate_fn(data_list, torch_geo_collate_fn=self.collate_fn_gnn)
            case _:
                return self.collate_fn_gnn(data_list)

    def distributed_sampler(self, dataset: Dataset, shuffle: bool):
        return DistributedSampler(
            dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=shuffle,
        )

    @override
    def train_dataloader(self):
        dataset = self.train_dataset()
        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_train)
        batch_sampler = BalancedBatchSampler(
            sampler,
            batch_size=self.config.batch_size,
            device=self.device,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return data_loader

    def _val_dataloader(self, with_sampler: bool):
        dataset = self.val_dataset()
        if with_sampler:
            sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_val)
            batch_sampler = BalancedBatchSampler(
                sampler,
                batch_size=self.config.batch_size,
                device=self.device,
            )
        else:
            batch_sampler = None
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return data_loader

    @override
    def val_dataloader(self):
        return self._val_dataloader(with_sampler=True)

    def _task_config(self, name: str):
        return next((task for task in self.config.tasks if task.name == name), None)

    @staticmethod
    def _to_int(value):
        return int(value.item() if torch.is_tensor(value) else value)

    def train_data_transform(self, data: Data):
        data = self.data_transform(data)
        return data

    def val_data_transform(self, data: Data):
        data = self.data_transform(data)
        return data

    def data_transform(self, data: BaseData):
        data.y = (
            data.y.float()
            if torch.is_tensor(data.y)
            else torch.tensor(data.y, dtype=torch.float)
        )

        data.fixed = data.fixed.bool()
        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = self._to_int(data.natoms)
        data.sid = self._to_int(data.sid)

        if (
            isinstance(self.config.backbone, GOCBackboneConfig)
            and not self.config.generate_graphs_on_gpu
        ):
            for graph_type in ["main", "a2a", "a2ee2a", "qint"]:
                key = f"{graph_type}_num_neighbors"
                setattr(data, key, self._to_int(data[key]))

        for attr in ("y", "force"):
            key = f"{attr}_scale"
            if not hasattr(data, key):
                raise ValueError(f"{key=} not found in data")

        # make all tensors contiguous
        for key in data.keys():
            if not torch.is_tensor(data[key]):
                continue

            data[key] = data[key].contiguous()

        return data

    def _process_aint_graph(self, graph: Graph, *, training: bool):
        if self.config.edge_dropout:
            graph["edge_index"], mask = dropout_edge(
                graph["edge_index"],
                p=self.config.edge_dropout,
                training=training,
            )
            graph["distance"] = graph["distance"][mask]
            graph["vector"] = graph["vector"][mask]
            graph["cell_offset"] = graph["cell_offset"][mask]

            if "id_swap_edge_index" in graph:
                graph["id_swap_edge_index"] = graph["id_swap_edge_index"][mask]

        return graph

    def _generate_graphs_goc(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
        *,
        training: bool,
    ):
        aint_graph = generate_graph(
            data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, pbc=pbc
        )
        aint_graph = self._process_aint_graph(aint_graph, training=training)
        subselect = partial(
            subselect_graph,
            data,
            aint_graph,
            cutoff_orig=cutoffs.aint,
            max_neighbors_orig=max_neighbors.aint,
        )
        main_graph = subselect(cutoffs.main, max_neighbors.main)
        aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)
        qint_graph = subselect(cutoffs.qint, max_neighbors.qint)

        # We can't do this at the data level: This is because the batch collate_fn doesn't know
        # that it needs to increment the "id_swap" indices as it collates the data.
        # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
        # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
        qint_graph = tag_mask(data, qint_graph, tags=self.config.backbone.qint_tags)

        graphs = {
            "main": main_graph,
            "a2a": aint_graph,
            "a2ee2a": aeaint_graph,
            "qint": qint_graph,
        }

        for graph_type, graph in graphs.items():
            for key, value in graph.items():
                setattr(data, f"{graph_type}_{key}", value)

        if not self.config.generate_graphs_on_gpu:
            for graph_type, graph in graphs.items():
                graph["num_neighbors"] = graph["edge_index"].shape[1]

        return data
    
    def _generate_graphs_m3gnet(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
        *,
        training: bool,
    ):
        aint_graph = generate_graph(
            data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, pbc=pbc
        )
        aint_graph = self._process_aint_graph(aint_graph, training=training)
        subselect = partial(
            subselect_graph,
            data,
            aint_graph,
            cutoff_orig=cutoffs.aint,
            max_neighbors_orig=max_neighbors.aint,
        )
        main_graph = subselect(cutoffs.main, max_neighbors.main)
        aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)

        # We can't do this at the data level: This is because the batch collate_fn doesn't know
        # that it needs to increment the "id_swap" indices as it collates the data.
        # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
        # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])

        graphs = {
            "main": main_graph,
            "a2a": aint_graph,
            "a2ee2a": aeaint_graph,
        }

        for graph_type, graph in graphs.items():
            for key, value in graph.items():
                setattr(data, f"{graph_type}_{key}", value)

        if not self.config.generate_graphs_on_gpu:
            for graph_type, graph in graphs.items():
                graph["num_neighbors"] = graph["edge_index"].shape[1]

        return data

    def _generate_graphs_graphormer(
        self,
        data: BaseData,
        cutoff: float,
        filter_src_pos_by_tag: int | None,  # should be 2 for oc20
        no_copy_tag: int | None,  # should be 2 for oc20 (no copy ads)
    ):
        from ...models.graphormer.types import data_transform

        return data_transform(
            data,
            cutoff=cutoff,
            filter_src_pos_by_tag=filter_src_pos_by_tag,
            no_copy_tag=no_copy_tag,
        )

    def _generate_graphs_torchmd(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
        *,
        training: bool,
    ):
        graph = generate_graph(
            data, cutoff=cutoffs.main, max_neighbors=max_neighbors.main, pbc=pbc
        )

        data.edge_index = graph["edge_index"]
        data.edge_distances = graph["distance"]
        data.edge_displacement_vectors = graph["vector"]

        return data

    def _generate_graphs(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
        filter_src_pos_by_tag: int | None,  # should be 2 for oc20
        no_copy_tag: int | None,  # should be 2 for oc20 (no copy ads)
        *,
        training: bool,
    ):
        match self.config.backbone:
            case GOCBackboneConfig():
                if not self.config.generate_graphs_on_gpu:
                    data = self._generate_graphs_goc(
                        data,
                        cutoffs=cutoffs,
                        max_neighbors=max_neighbors,
                        pbc=pbc,
                        training=training,
                    )
                return data
            case M3GNetBackboneConfig():
                if not self.config.generate_graphs_on_gpu:
                    data = self._generate_graphs_m3gnet(
                        data,
                        cutoffs=cutoffs,
                        max_neighbors=max_neighbors,
                        pbc=pbc,
                        training=training,
                    )
                return data
            case Graphormer3DConfig():
                assert (
                    not self.config.generate_graphs_on_gpu
                ), "generate_graphs_on_gpu not supported for this model"
                return self._generate_graphs_graphormer(
                    data,
                    cutoff=cutoffs.main,
                    filter_src_pos_by_tag=filter_src_pos_by_tag,
                    no_copy_tag=no_copy_tag,
                )
            case TorchMDNetBackboneConfig():
                assert (
                    not self.config.generate_graphs_on_gpu
                ), "generate_graphs_on_gpu not supported for this model"
                return self._generate_graphs_torchmd(
                    data,
                    cutoffs=cutoffs,
                    max_neighbors=max_neighbors,
                    pbc=pbc,
                    training=training,
                )
            case _:
                assert_never(self.config.backbone)

    def _initial_data_transform(self, data: BaseData):
        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y)
        data.y = data.y.view(-1)

        return data

    def oc20_transform(self, data: BaseData, *, training: bool):
        if hasattr(data, "energy"):  # upgrade old dataset
            data.y = data.pop("energy")
        if hasattr(data, "forces"):
            data.force = data.pop("forces")

        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("oc20")
        ) is not None, "OC20 task is not configured"

        # convert back these keys into required format for collation
        data.natoms = int(data.natoms.item() if torch.is_tensor(data) else data.natoms)

        data.atomic_numbers = data.atomic_numbers.long()
        data.tags = data.tags.long()

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale

        return self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=True,
            training=training,
            filter_src_pos_by_tag=2,
            no_copy_tag=2,
        )

    def oc22_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("oc22")
        ) is not None, "OC22 task is not configured"

        # convert back these keys into required format for collation
        data.natoms = int(data.natoms.item() if torch.is_tensor(data) else data.natoms)

        data.atomic_numbers = data.atomic_numbers.long()
        data.tags = data.tags.long()
        try:
            data.y = torch.tensor(float(data.y)).view(-1)
        except:
            data.y = torch.tensor(float(data.y_relaxed)).view(-1)
        data.name = "oc22"

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale

        return self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=True,
            training=training,
            filter_src_pos_by_tag=2,
            no_copy_tag=2,
        )

    @staticmethod
    def _set_inf_cell(data: BaseData, max_length: float = 1000.0):
        data.cell = (torch.eye(3) * max_length).unsqueeze(dim=0)
        return data

    def ani1x_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("ani1x")
        ) is not None, "ANI1x task is not configured"

        data.y = data.y.view(-1).float()
        if not hasattr(data, "sid"):
            data.sid = data.absolute_idx
        if not hasattr(data, "natoms"):
            data.natoms = data.num_nodes

        # data.fixed = torch.ones(data.natoms)
        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()
        data.name = "ani1x"

        data = self._set_inf_cell(data)

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale

        if self.config.generate_graphs_on_gpu:
            # We only support PBC graph generation on GPU,
            # so we need to add a dummy cell to the data
            data.cell = torch.eye(3).unsqueeze(dim=0) * 1000.0

        return self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(8.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
            training=training,
            filter_src_pos_by_tag=None,
            no_copy_tag=None,
        )

    def transition1x_transform(self, data: BaseData, *, training: bool):
        data = self._initial_data_transform(data)

        assert (
            config := self._task_config("transition1x")
        ) is not None, "Transition1x task is not configured"

        data.y = data.y.view(-1).float()
        if not hasattr(data, "sid"):
            data.sid = data.absolute_idx
        if not hasattr(data, "natoms"):
            data.natoms = data.num_nodes

        # data.fixed = torch.ones(data.natoms)
        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()
        data.name = "transition1x"

        data = self._set_inf_cell(data)

        data.y_scale = config.energy_loss_scale
        data.force_scale = config.force_loss_scale

        if self.config.generate_graphs_on_gpu:
            # We only support PBC graph generation on GPU,
            # so we need to add a dummy cell to the data
            data.cell = torch.eye(3).unsqueeze(dim=0) * 1000.0

        return self._generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(8.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
            training=training,
            filter_src_pos_by_tag=None,
            no_copy_tag=None,
        )

    if False and ll._experimental.MEASURE_FLOPS_AVAILABLE:

        @property
        @cache
        def flops_per_batch(self):
            # Let's take a copy of the model so we don't modify the original
            module = copy.deepcopy(self)

            # Make sure the model is on the CPU
            module.cpu()

            # Things should still be on the CPU here
            batch = next(iter(module._val_dataloader(with_sampler=False)))
            batch = move_data_to_device(batch, torch.device("cpu"))

            def loss_fn(model_output: tuple[torch.Tensor, torch.Tensor]):
                energy, forces = model_output
                return module.compute_losses(batch, energy=energy, forces=forces)

            return ll._experimental.measure_flops(
                lambda: module(batch),
                loss_fn=loss_fn,
            )

    def throughput_monitor_batch_stats(self, batch: Data):
        return {
            "batch_size": batch.y.shape[0],
            "length": batch.atomic_numbers.shape[0],
        }
