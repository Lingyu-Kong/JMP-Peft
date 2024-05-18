import copy
import fnmatch
import math
import time
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, cast

import rich
import rich.console
import rich.markdown
import rich.table
import rich.tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import (
    OptimizerLRScheduler,
)
from ll import (
    ActSave,
    AllowMissing,
    BaseConfig,
    Field,
    LightningModuleBase,
    TypedConfig,
)
from ll.data.balanced_batch_sampler import BalancedBatchSampler, DatasetWithSizes
from ll.util.typed import TypedModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar, assert_never, override

from ...datasets.finetune_lmdb import (
    FinetuneDatasetConfig as FinetuneLmdbDatasetConfigBase,
)
from ...datasets.finetune_lmdb import FinetuneLmdbDataset
from ...datasets.finetune_pdbbind import PDBBindConfig, PDBBindDataset
from ...datasets.matbench_discovery_ase import MatbenchDiscoveryAseDataset
from ...datasets.matbench_discovery_megnet_npz import (
    MatbenchTrajectoryDataset as MatbenchDiscoveryMegNetNpzDataset,
)
from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import BackboneConfig
from ...models.gemnet.layers.base_layers import ScaledSiLU
from ...modules import transforms as T
from ...modules.dataset import dataset_transform as DT
from ...modules.dataset.common import CommonDatasetConfig, wrap_common_dataset
from ...modules.dist_lora import AdapterLayer, DLoraConfig
from ...modules.ema import EMAConfig
from ...modules.lora import Linear as LoraLinear
from ...modules.lora import LoraConfig, LoRALayer, LoraRootConfig
from ...modules.scheduler.linear_warmup_cos_rlp import (
    PerParamGroupLinearWarmupCosineAnnealingRLPLR,
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
from ...utils.gradient_checkpointing import GradientCheckpointingConfig
from ...utils.state_dict import load_state_dict
from ..config import (
    EmbeddingConfig,
    OptimizerConfig,
    OutputConfig,
    optimizer_from_config,
)
from .metrics import FinetuneMetrics, MetricPair, MetricsConfig
from .output_head import (
    GradientForcesTargetConfig,
    GraphBinaryClassificationTargetConfig,
    GraphMulticlassClassificationTargetConfig,
    GraphScalarTargetConfig,
    GraphTargetConfig,
    NodeTargetConfig,
    NodeVectorTargetConfig,
    OutputHeadInput,
)

log = getLogger(__name__)


DatasetType: TypeAlias = (
    FinetuneLmdbDataset
    | PDBBindDataset
    | MatbenchDiscoveryAseDataset
    | MatbenchDiscoveryMegNetNpzDataset
)


class RLPWarmupConfig(TypedConfig):
    step_type: Literal["step", "epoch"]
    """The type of step to use for the warmup"""

    steps: int
    """Number of steps for the warmup"""

    start_lr_factor: float
    """The factor to multiply the initial learning rate by at the start of the warmup"""


class RLPConfig(TypedConfig):
    name: Literal["rlp"] = "rlp"

    monitor: str | None = None
    mode: Literal["min", "max"] | None = None
    patience: int = 10
    factor: float = 0.1
    min_lr: float = 0.0
    eps: float = 1.0e-8
    cooldown: int = 0
    threshold: float = 1.0e-4
    threshold_mode: Literal["rel", "abs"] = "rel"
    interval: Literal["epoch", "step"] = "epoch"
    frequency: int = 1
    warmup: RLPWarmupConfig | None = None

    def _to_linear_warmup_cos_rlp_dict(self):
        """
        Params for PerParamGroupLinearWarmupCosineAnnealingRLPLR's RLP
            mode="min",
            factor=0.1,
            patience=10,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=False,
        """
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "min_lr": self.min_lr,
            "eps": self.eps,
            "verbose": True,
        }


class WarmupCosRLPConfig(TypedConfig):
    name: Literal["warmup_cos_rlp"] = "warmup_cos_rlp"

    warmup_steps: int | None = None
    warmup_epochs: int | float | None = None
    max_steps: int | None = None
    max_epochs: int | float | None = None
    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1
    should_restart: bool = False

    rlp: RLPConfig

    @override
    def __post_init__(self):
        super().__post_init__()

        assert self.rlp.warmup is None, "RLP warmup is not supported"


LRSchedulerConfig: TypeAlias = Annotated[
    RLPConfig | WarmupCosRLPConfig, Field(discriminator="name")
]


class FreezeConfig(TypedConfig):
    backbone: bool = False
    """Should the backbone be frozen?"""
    embedding: bool = False
    """Should the embedding layer be frozen?"""

    backbone_bases: bool = False
    """Should the basis functions in the backbone be frozen?"""
    backbone_interaction_layers: list[int] | None = None
    """Which interaction layers, if any, in the backbone should be frozen?"""
    backbone_output_layers: list[int] | None = None
    """Which output layers, if any, in the backbone should be frozen?"""

    parameter_patterns: list[str] = []
    """List of parameter patterns to freeze"""

    ensure_non_frozen_parameter_patterns: list[str] = []
    """List of parameter patterns to ensure are not frozen"""

    report_parameters: bool = False
    """
    If `True`, we will print a large table of all parameters and their requires_grad status.
    """


class ParamSpecificOptimizerConfig(TypedConfig):
    name: str | None = None
    """The name of the parameter group for this config"""

    paremeter_patterns: list[str] = []
    """List of parameter patterns to match for this config"""

    optimizer: OptimizerConfig | None = None
    """
    The optimizer config for this parameter group.
    If None, the default optimizer will be used.
    """

    lr_scheduler: LRSchedulerConfig | None = None
    """
    The learning rate scheduler config for this parameter group.
    If None, the default learning rate scheduler will be used.
    """


class CheckpointLoadConfig(TypedConfig):
    ignored_key_patterns: list[str] = []
    """Patterns to ignore when loading the checkpoint"""

    ignored_missing_keys: list[str] = []
    """Keys to ignore if they are missing in the checkpoint"""

    ignored_unexpected_keys: list[str] = []
    """Keys to ignore if they are unexpected in the checkpoint"""

    reset_embeddings: bool = False
    """
    If true, it will reset the embeddings to the initial state
    after loading the checkpoint
    """


class TestConfig(TypedConfig):
    save_checkpoint_base_dir: Path | None = None
    """Where to save the checkpoint information for this run (or None to disable)"""

    save_results_base_dir: Path | None = None
    """Where to save the results for this run (or None to disable)"""


class FinetuneLmdbDatasetConfig(FinetuneLmdbDatasetConfigBase, CommonDatasetConfig):
    name: Literal["lmdb"] = "lmdb"

    def create_dataset(self):
        return FinetuneLmdbDataset(self)


class FinetunePDBBindDatasetConfig(PDBBindConfig, CommonDatasetConfig):
    name: Literal["pdbbind"] = "pdbbind"

    def create_dataset(self):
        return PDBBindDataset(task=self.task, split=self.split)


class FinetuneMatbenchDiscoveryDatasetConfig(CommonDatasetConfig):
    name: Literal["matbench_discovery"] = "matbench_discovery"

    split_csv_path: Path
    base_path: Path
    atoms_metadata: Path | None = None
    energy_linref_path: Path | None = None
    fractional_coordinates: bool = False

    def create_dataset(self):
        return MatbenchDiscoveryAseDataset(
            split_csv_path=self.split_csv_path,
            base_path=self.base_path,
            atoms_metadata=self.atoms_metadata,
            energy_linref_path=self.energy_linref_path,
            fractional_coordinates=self.fractional_coordinates,
        )


class FinetuneMatbenchDiscoveryMegNet133kDatasetConfig(CommonDatasetConfig):
    name: Literal["matbench_discovery_megnet_133k"] = "matbench_discovery_megnet_133k"

    base_path: Path
    energy_linref_path: Path | None = None

    def create_dataset(self):
        return MatbenchDiscoveryMegNetNpzDataset(
            base_path=self.base_path, energy_linref_path=self.energy_linref_path
        )


FinetuneDatasetConfig: TypeAlias = Annotated[
    FinetuneLmdbDatasetConfig
    | FinetunePDBBindDatasetConfig
    | FinetuneMatbenchDiscoveryDatasetConfig
    | FinetuneMatbenchDiscoveryMegNet133kDatasetConfig,
    Field(discriminator="name"),
]


class BatchDumpConfig(TypedConfig):
    dump_if_loss_gt: float | None = None
    """Dump the batch if the loss is greater than this value"""


class FinetuneConfigBase(BaseConfig):
    gradient_checkpointing: GradientCheckpointingConfig | None = None
    """Gradient checkpointing configuration"""

    train_dataset: FinetuneDatasetConfig | None = None
    """Configuration for the train dataset"""
    val_dataset: FinetuneDatasetConfig | None = None
    """Configuration for the val dataset"""
    test_dataset: FinetuneDatasetConfig | None = None
    """Configuration for the test dataset"""

    optimizer: OptimizerConfig
    """Optimizer to use."""
    lr_scheduler: LRSchedulerConfig | None = None
    """Learning rate scheduler configuration. If None, no learning rate scheduler is used."""

    embedding: AllowMissing[EmbeddingConfig] = TypedConfig.MISSING
    """Configuration for the embedding layer."""
    backbone: BackboneConfig
    """Configuration for the backbone."""
    output: OutputConfig = OutputConfig(num_mlps=5)
    """Configuration for the output head."""
    lora: LoraRootConfig | None = None
    """Low-rank Adaptation (LoRA) configuration"""
    dlora: DLoraConfig | None = None
    """Distributation-Learning of Rank-Adaptation (DLora) configuration"""

    batch_size: int
    """Batch size to use."""
    eval_batch_size: int | None = None
    """Batch size to use for evaluation. If None, use the same as batch_size."""
    num_workers: int = 8
    """Number of workers to use for data loading."""
    pin_memory: bool = True
    """Whether to use pin memory for data loading."""

    @property
    def activation_cls(self):
        match self.backbone.activation:
            case "scaled_silu" | "scaled_swish":
                return ScaledSiLU
            case "silu" | "swish":
                return nn.SiLU
            case None:
                return nn.Identity
            case _:
                raise NotImplementedError(
                    f"Activation {self.backbone.activation=} is not implemented"
                )

    test: TestConfig | None = None
    """Configuration for test stage"""

    graph_targets: list[GraphTargetConfig] = []
    """List of graph targets (e.g., energy, is_metal)"""

    node_targets: list[NodeTargetConfig] = []
    """List of node targets (e.g., force)"""

    @property
    def targets(self):
        """List of all targets, i.e., graph and node targets"""
        return self.graph_targets + self.node_targets

    normalization: dict[str, NormalizationConfig] = {}
    """Normalization parameters for each target"""

    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] | None = None
    """Configuration for parameter-specific optimizers"""

    use_balanced_batch_sampler: bool = False
    """
    Whether to use balanced batch sampler.

    This balances the batches across all distributed nodes (i.e., GPUs, TPUs, nodes, etc.)
    to ensure that each batch has an equal number of **atoms** across all nodes.
    """

    freeze: FreezeConfig = FreezeConfig()
    """Configuration for freezing parameters"""

    ckpt_load: CheckpointLoadConfig = CheckpointLoadConfig()
    """Configuration for behavior when loading checkpoints"""

    shuffle_val: bool = False
    """Whether to shuffle the validation set"""
    shuffle_test: bool = False
    """Whether to shuffle the test set"""

    metrics: MetricsConfig = MetricsConfig()
    """Configuration for metrics"""

    ema: EMAConfig | None = None
    """Configuration for exponential moving average"""

    debug_print_every: int | None = None
    """Print debug information every `debug_print_every` iterations. `None` to disable."""

    batch_dump: BatchDumpConfig | None = None
    """Configuration for dumping batches"""

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.embedding is TypedConfig.MISSING:
            self.embedding = EmbeddingConfig(
                num_elements=self.backbone.num_elements,
                embedding_size=self.backbone.emb_size_atom,
            )

        if self.use_balanced_batch_sampler:
            assert not self.trainer.use_distributed_sampler, "config.trainer.use_distributed_sampler must be False when using balanced batch sampler"

        assert self.targets, (
            "At least one target must be specified, "
            f"but none are specified: {self.targets=}"
        )

        if self.batch_dump is not None:
            assert self.trainer.actsave, "Batch dump requires actsave to be enabled"


TConfig = TypeVar("TConfig", bound=FinetuneConfigBase)


class FinetuneModelBase(LightningModuleBase[TConfig], Generic[TConfig]):
    @abstractmethod
    def metric_prefix(self) -> str: ...

    @override
    def on_test_end(self):
        super().on_test_end()

        match self.config.test:
            case TestConfig(save_checkpoint_base_dir=Path() as base):
                # The save dir for this run should be base/{metric_prefix()}/{config.name}-{config.id}
                base = base / self.metric_prefix()
                base.mkdir(parents=True, exist_ok=True)
                save_dir = base / f"{self.config.name}-{self.config.id}"
                if save_dir.exists():
                    i = 0
                    while (
                        save_dir := base / f"{self.config.name}-{self.config.id}-{i}"
                    ).exists():
                        i += 1
                save_dir.mkdir(parents=True, exist_ok=True)

                # Get ckpt path from config
                ckpt_path = self.config.meta.get("ckpt_path")
                if ckpt_path is None:
                    raise ValueError(
                        f"Checkpoint path not found in meta: {self.config.meta=}"
                    )
                ckpt_path = Path(ckpt_path)
                if not ckpt_path.exists():
                    raise ValueError(f"Checkpoint path does not exist: {ckpt_path=}")

                # Create a symlink to the checkpoint
                symlink_path = base / f"pretrained-{ckpt_path.name}"
                if symlink_path.exists():
                    raise ValueError(f"Symlink path already exists: {symlink_path=}")
                symlink_path.symlink_to(ckpt_path)

                # Also create an ckptpath.txt file that contains the original ckpt path
                _ = (base / "ckptpath.txt").write_text(
                    str(ckpt_path.resolve().absolute())
                )

                log.critical(f"Saving checkpoint information to {save_dir}")
            case _:
                pass

    def primary_metric(self, split: Literal["train", "val", "test"] | None = "val"):
        if (config := self.config.primary_metric) is None:
            raise ValueError("Primary metric not set in config")
        metric = config.name
        if split is not None:
            metric = f"{split}/{metric}"
        return metric, config.mode

    def _set_rlp_config_monitors(self):
        match self.config.lr_scheduler:
            case RLPConfig(monitor=None) as rlp_config:
                rlp_config.monitor, rlp_config.mode = self.primary_metric()
            case WarmupCosRLPConfig(rlp=RLPConfig(monitor=None) as rlp_config):
                rlp_config.monitor, rlp_config.mode = self.primary_metric()
            case _:
                pass

    def _construct_backbone(self):
        log.critical("Using regular backbone")

        backbone = GemNetOCBackbone(
            self.config.backbone,
            **dict(self.config.backbone),
            lora=self.config.lora.create_lora_config()
            if self.config.lora
            else LoraConfig.disabled(),
            gradient_checkpointing=self.config.gradient_checkpointing,
            dlora=self.config.dlora,
        )
        log.critical(f"Constructed backbone with dlora={self.config.dlora}")

        return backbone

    def metrics_provider(
        self,
        prop: str,
        batch: BaseData,
        preds: dict[str, torch.Tensor],
    ) -> MetricPair | None:
        if (pred := preds.get(prop)) is None or (
            target := getattr(batch, prop, None)
        ) is None:
            return None

        if (
            self.config.normalization
            and (norm := self.config.normalization.get(prop)) is not None
        ):
            # Denormalize the predictions and targets
            pred = pred * norm.std + norm.mean
            target = target * norm.std + norm.mean

        return MetricPair(predicted=pred, ground_truth=target)

    @override
    def __init__(self, hparams: TConfig):
        super().__init__(hparams)

        # Set up callbacks
        if (ema := self.config.ema) is not None:
            self.register_callback(lambda: ema.construct_callback())

        self._set_rlp_config_monitors()

        self.embedding = nn.Embedding(
            num_embeddings=self.config.embedding.num_elements,
            embedding_dim=self.config.embedding.embedding_size,
        )

        self.backbone = self._construct_backbone()
        self.register_shared_parameters(
            [(p, c) for p, c in self.backbone.shared_parameters if p.requires_grad]
        )

        self.construct_output_heads()

        self.train_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_targets,
            self.config.node_targets,
        )
        self.val_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_targets,
            self.config.node_targets,
        )
        self.test_metrics = FinetuneMetrics(
            self.config.metrics,
            self.metrics_provider,
            self.config.graph_targets,
            self.config.node_targets,
        )

        # Sanity check: ensure all named_parameters have requires_grad=True,
        #   otherwise add them to ignored_parameters.
        self.ignored_parameters = set[nn.Parameter]()
        ignored_parameters_list: list[str] = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                continue
            self.ignored_parameters.add(param)
            ignored_parameters_list.append(name)
        log.info(f"List of ignored parameters: {ignored_parameters_list}")

        self.process_freezing()

        for target in self.config.graph_targets:
            match target:
                case GraphMulticlassClassificationTargetConfig(
                    class_weights=class_weights
                ) if class_weights:
                    self.register_buffer(
                        f"{target.name}_class_weights",
                        torch.tensor(class_weights, dtype=torch.float),
                        persistent=False,
                    )
                case _:
                    pass

        if self.config.dlora is not None:
            # Report the number of new adapter parameters added.
            num_new_params = sum(
                p.numel() for _, p in self.dlora_added_parameters().items()
            )
            log.critical(f"[DLora]: Added {num_new_params} new parameters.")

    def freeze_parameters(self, parameters: Iterable[nn.Parameter], *, name: str):
        n_params = 0
        for param in parameters:
            if param in self.ignored_parameters:
                continue

            param.requires_grad = False
            n_params += param.numel()
        log.critical(f"Freezing {n_params} parameters in {name}")

    def named_parameters_matching_patterns(
        self,
        patterns: list[str],
        ignored_parameters: set[nn.Parameter] | None = None,
        requires_grad_only: bool = False,
    ):
        ignored_parameters_set = self.ignored_parameters | (ignored_parameters or set())

        for name, param in self.named_parameters():
            if param in ignored_parameters_set:
                continue
            if requires_grad_only and not param.requires_grad:
                continue
            if (
                matching_pattern := next(
                    (pattern for pattern in patterns if fnmatch.fnmatch(name, pattern)),
                    None,
                )
            ) is None:
                continue

            yield name, param, matching_pattern

    def process_freezing(self):
        config = self.config.freeze

        if config.backbone:
            self.freeze_parameters(self.backbone.parameters(), name="backbone")

        if config.embedding:
            self.freeze_parameters(self.embedding.parameters(), name="embedding")

        if config.backbone_interaction_layers:
            for layer_idx in config.backbone_interaction_layers:
                self.freeze_parameters(
                    self.backbone.int_blocks[layer_idx].parameters(),
                    name=f"backbone.int_blocks[{layer_idx}]",
                )

        if config.backbone_output_layers:
            for layer_idx in config.backbone_output_layers:
                self.freeze_parameters(
                    self.backbone.out_blocks[layer_idx].parameters(),
                    name=f"backbone.out_blocks[{layer_idx}]",
                )

        if config.backbone_bases:
            self.freeze_parameters(
                self.backbone.bases.parameters(), name="backbone.bases"
            )

        if config.parameter_patterns:
            for (
                name,
                param,
                matching_pattern,
            ) in self.named_parameters_matching_patterns(config.parameter_patterns):
                param.requires_grad = False
                log.info(f"Freezing {name} (pattern: {matching_pattern})")

        if (
            (lc := self.config.lora) is not None
            and lc.enabled
            and lc.freeze_non_lora_backbone
        ):
            # See https://github.com/microsoft/LoRA/blob/main/loralib/utils.py#L13
            # Get all non-LoRA parameters. We make this a set
            #   because it'll be more efficient to pop from a set than a list.
            parameters = {
                param
                for name, param in self.backbone.named_parameters()
                if not name.endswith(".lora_A") and not name.endswith(".lora_B")
            }
            # Handle the bias config
            match lc.bias:
                case "none":
                    # Do nothing
                    pass
                case "all":
                    # Unfreeze all bias parameters.
                    for name, param in self.backbone.named_parameters():
                        if "bias" not in name:
                            continue
                        if param not in parameters:
                            continue
                        parameters.remove(param)
                case "lora_only":
                    # Unfreeze only the bias parameters of nn.Linear layers
                    #   that are LoRA layers.
                    for m in self.backbone.modules():
                        if (
                            not isinstance(m, LoRALayer)
                            or (bias := getattr(m, "bias", None)) is None
                        ):
                            continue

                        if bias not in parameters:
                            log.warning(
                                f"LoRA layer bias parameter {m} not found in list of parameters to freeze. "
                                "This should not happen."
                            )
                            continue

                        parameters.remove(bias)

                case _:
                    assert_never(lc.bias)

            self.freeze_parameters(parameters, name="backbone (non-LoRA)")

        # After we do all the freezing, we want to unfreeze any parameters that
        #   match the ensure_non_frozen_parameter_patterns.
        if nonfrozen := config.ensure_non_frozen_parameter_patterns:
            for (
                name,
                param,
                matching_pattern,
            ) in self.named_parameters_matching_patterns(nonfrozen):
                if param.requires_grad:
                    continue

                param.requires_grad = True
                log.info(f"Unfreezing {name} (pattern: {matching_pattern})")

        if config.report_parameters:
            tree = rich.tree.Tree("Parameters")

            def _add_module(
                root_tree: rich.tree.Tree,
                module: nn.Module,
                name: str,
            ):
                # Compute the total number of parameters for title
                num_params = sum(p.numel() for p in module.parameters())
                num_trainable = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                percent_trainble = int(math.ceil(num_trainable / num_params * 100))
                title = (
                    f"{name} --- {num_trainable:,}/{num_params:,} ({percent_trainble}%)"
                )

                # Parameter table
                table = rich.table.Table()
                table.add_column("Name", justify="left")
                table.add_column("Trainable?", justify="center")
                table.add_column("# (Trainable/Total)", justify="right")

                # # Sort parameters by first requires_grad (True first) and then name (A-Z)
                # parameters = sorted(
                #     submodule.named_parameters(),
                #     key=lambda x: (not x[1].requires_grad, x[0]),
                # )
                for module_name, submodule in module.named_modules():
                    immediate_parameters = list(
                        submodule.named_parameters(recurse=False)
                    )
                    # If no parameters, skip
                    if not immediate_parameters or not sum(
                        param.numel() for _, param in immediate_parameters
                    ):
                        continue

                    # Add parameters to the table
                    for i, (name, param) in enumerate(immediate_parameters):
                        num_params = param.numel()
                        trainable = num_params if param.requires_grad else 0
                        prefix = f"**{module_name}**" if i % 2 == 0 else module_name
                        table.add_row(
                            rich.markdown.Markdown(f"\t{prefix}.{name}"),
                            "✅" if param.requires_grad else "❌",
                            f"{trainable:,}/{num_params:,}",
                        )

                group = rich.console.Group(title, table)
                root_tree.add(group)

            _add_module(tree, self.embedding, "Embedding")
            _add_module(tree, self.backbone, "Backbone")
            _add_module(tree, self.outputs, "Output")

            rich.print(tree)

        all_parameters = [
            param for param in self.parameters() if param not in self.ignored_parameters
        ]
        num_frozen = sum(
            param.numel() for param in all_parameters if not param.requires_grad
        )
        num_train = sum(
            param.numel() for param in all_parameters if param.requires_grad
        )
        num_total = sum(param.numel() for param in all_parameters)
        percent_frozen = num_frozen / num_total * 100
        log.critical(
            f"Freezing {num_frozen:,} parameters ({percent_frozen:.2f}%) out of "
            f"{num_total:,} total parameters ({num_train:,} trainable)"
        )

    def construct_output_heads(self):
        self.outputs = TypedModuleDict(
            {
                target.name: target.construct_output_head(
                    self.config.output,
                    self.config.backbone.emb_size_atom,
                    self.config.backbone.emb_size_edge,
                    self.config.activation_cls,
                )
                for target in self.config.targets
            },
            key_prefix="ft_mlp_",
        )

    def lora_added_bias_parameters(self):
        if (
            (lc := self.config.lora) is None
            or not lc.enabled
            or not lc.add_bias_to_lora_linear
        ):
            return {}

        # Get all LoRA A/B parameters
        lora_added_bias_parameters: dict[str, torch.Tensor] = {}
        for name, module in self.backbone.named_modules(
            remove_duplicate=False,
            # ^ `remove_duplicate=False` is necessary because we have multiple
            #   modules with the same name (e.g., see `seq_energy_pre` in GemNet's output blocks).
        ):
            if (
                not isinstance(module, LoraLinear)
                or not module.has_new_lora_linear_bias
            ):
                continue

            lora_added_bias_parameters[f"{name}.bias"] = module.bias

        return lora_added_bias_parameters

    def dlora_added_parameters(self):
        if self.config.dlora is None:
            return {}

        dlora_added_parameters: dict[str, torch.Tensor] = {}
        for name, module in self.backbone.named_modules(
            remove_duplicate=False,
            # ^ `remove_duplicate=False` is necessary because we have multiple
            #   modules with the same name (e.g., see `seq_energy_pre` in GemNet's output blocks).
        ):
            if not isinstance(module, AdapterLayer):
                continue

            dlora_added_parameters.update(
                {f"{name}.{k}": v for k, v in module.state_dict(keep_vars=True).items()}
            )

        return dlora_added_parameters

    def load_backbone_state_dict(
        self,
        *,
        backbone: Mapping[str, Any],
        embedding: Mapping[str, Any],
        strict: bool = True,
    ):
        ignored_key_patterns = copy.deepcopy(self.config.ckpt_load.ignored_key_patterns)
        # If we're dumping the backbone's force out heads, then we need to ignore
        #   the unexpected keys for the force out MLPs and force out heads.
        if (
            not self.config.backbone.regress_forces
            or not self.config.backbone.direct_forces
        ):
            ignored_key_patterns.append("out_mlp_F.*")
            for block_idx in range(self.config.backbone.num_blocks + 1):
                ignored_key_patterns.append(f"out_blocks.{block_idx}.scale_rbf_F.*")
                ignored_key_patterns.append(f"out_blocks.{block_idx}.dense_rbf_F.*")
                ignored_key_patterns.append(f"out_blocks.{block_idx}.seq_forces.*")

        # Ignore non-existant LoRA parameters
        lora_added_bias_parameters = set(self.lora_added_bias_parameters().keys())
        log.debug(f"LoRA added bias parameters: {lora_added_bias_parameters}")

        dlora_added_parameters = set(self.dlora_added_parameters().keys())

        def should_ignore_missing_key_fn(k: str):
            # LoRA added params
            if (lc := self.config.lora) is not None and lc.enabled:
                # Ignore non-existant LoRA A/B keys
                if k.endswith(".lora_A") or k.endswith(".lora_B"):
                    return True

                # Ignore new bias keys added by us
                if k in lora_added_bias_parameters:
                    return True

                # Ignore adapter layer keys for DLora
                if k in dlora_added_parameters:
                    return True

            # New LN layers
            if self.config.backbone.ln_per_layer:
                if "h_lns" in k or "m_lns" in k:
                    return True

            if self.config.backbone.scale_factor_to_ln:
                if "scale" in k and "ln" in k:
                    return True

            return False

        load_state_dict(
            self.backbone,
            backbone,
            strict=strict,
            ignored_key_patterns=ignored_key_patterns,
            ignored_missing_keys=self.config.ckpt_load.ignored_missing_keys,
            ignored_unexpected_keys=self.config.ckpt_load.ignored_unexpected_keys,
            should_ignore_missing_key_fn=should_ignore_missing_key_fn,
        )
        if not self.config.ckpt_load.reset_embeddings:
            load_state_dict(self.embedding, embedding, strict=strict)
        log.critical("Loaded backbone state dict (backbone and embedding).")

    _lora_debug_start_time: float

    def _lora_debug_print(self):
        # Check print_every parameter.
        if (
            self.config.debug_print_every is None
            or (self.global_step % self.config.debug_print_every) != 0
        ):
            return

        # Create a rich table to print the following:
        # - Optimizer state memory usage
        # - Total memory usage

        table = rich.table.Table(
            title="Memory Usage",
        )

        # Add columns
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Memory Usage", justify="right", style="magenta")

        # Add rows
        # - Optimizer state memory usage
        optimizer_state_memory_usage = 0
        optimizers = self.optimizers()
        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    optimizer_state_memory_usage += param.numel() * param.element_size()

        optimizer_state_memory_usage = optimizer_state_memory_usage / (1024**3)
        table.add_row("Optimizer State", f"{optimizer_state_memory_usage:,} GB")

        # - Total memory usage
        total_memory_usage = torch.cuda.memory_allocated()
        total_memory_usage = total_memory_usage / (1024**3)
        table.add_row("Total", f"{total_memory_usage:,} GB")

        # Print the table
        current_time = time.time()
        time_taken = self._lora_debug_start_time - current_time
        title = f"Time Taken: {time_taken:.2f} seconds"
        rich.print(rich.console.Group(title, table))

    @override
    def forward(self, data: BaseData):
        atomic_numbers = data.atomic_numbers - 1
        h = self.embedding(atomic_numbers)  # (N, d_model)
        out = cast(GOCBackboneOutput, self.backbone(data, h=h))

        output_head_input: OutputHeadInput = {
            "backbone_output": out,
            "data": data,
        }

        preds = {
            target: module(output_head_input) for target, module in self.outputs.items()
        }
        return preds

    def compute_losses(self, batch: BaseData, preds: dict[str, torch.Tensor]):
        losses: list[torch.Tensor] = []

        # Compute losses for graph targets
        for target in self.config.graph_targets:
            match target:
                case GraphScalarTargetConfig():
                    loss = F.l1_loss(preds[target.name], batch[target.name])
                case GraphBinaryClassificationTargetConfig():
                    y_input = preds[target.name]
                    y_target = batch[target.name].float()
                    pos_weight = None
                    if target.pos_weight is not None:
                        pos_weight = y_input.new_tensor(target.pos_weight)
                    loss = F.binary_cross_entropy_with_logits(
                        y_input, y_target, reduction="sum", pos_weight=pos_weight
                    )
                case GraphMulticlassClassificationTargetConfig():
                    weight = None
                    if target.class_weights:
                        weight = self.get_buffer(f"{target.name}_class_weights")

                    loss = F.cross_entropy(
                        preds[target.name],
                        batch[target.name].long(),
                        weight=weight,
                        reduction="sum",
                    )
                case _:
                    assert_never(target)

            # Log the loss
            self.log(f"{target.name}_loss", loss)

            # Multiply by the loss coefficient and log the scaled loss
            loss = target.loss_coefficient * loss
            self.log(f"{target.name}_loss_scaled", loss)

            losses.append(loss)

        for target in self.config.node_targets:
            match target:
                case NodeVectorTargetConfig() | GradientForcesTargetConfig():
                    assert preds[target.name].shape[-1] == 3
                    match target.loss:
                        case "l2mae":
                            loss = F.pairwise_distance(
                                preds[target.name], batch[target.name], p=2.0
                            ).mean()
                        case "mae":
                            loss = F.l1_loss(preds[target.name], batch[target.name])
                        case "mse":
                            loss = F.mse_loss(preds[target.name], batch[target.name])
                        case _:
                            assert_never(target.loss)
                case _:
                    assert_never(target)

            # Log the loss
            self.log(f"{target.name}_loss", loss)

            # Multiply by the loss coefficient and log the scaled loss
            loss = target.loss_coefficient * loss
            self.log(f"{target.name}_loss_scaled", loss)

            losses.append(loss)

        loss = sum(losses)
        self.log("loss", loss)

        return loss

    def _rlp_metric(self, config: RLPConfig):
        monitor = config.monitor
        assert monitor is not None, "RLP monitor must be specified."

        metric_prefix = f"val/{self.metric_prefix()}/"
        assert monitor.startswith(
            metric_prefix
        ), f"RLP {monitor=} must start with {metric_prefix}"
        monitor = monitor[len(metric_prefix) :]

        if (
            monitor.endswith("_mae")
            and (mae_metric := self.val_metrics.maes.get(monitor[: -len("_mae")]))
            is not None
        ):
            return mae_metric

        if (
            monitor.endswith("_balanced_accuracy")
            and (
                cls_metric := self.val_metrics.cls_metrics.get(
                    monitor[: -len("_balanced_accuracy")]
                )
            )
            is not None
        ):
            return cls_metric

        avail_mae_metrics = list(self.val_metrics.maes.keys())
        avail_cls_metrics = list(self.val_metrics.cls_metrics.keys())
        raise ValueError(
            f"RLP monitor {monitor} not found in metrics. "
            f"Available MAE metrics: {avail_mae_metrics}. "
            f"Available classification metrics: {avail_cls_metrics}"
        )

    def _cos_rlp_schedulers(self):
        if (lr_schedulers := self.lr_schedulers()) is None:
            log.warning("No LR scheduler found.")
            return

        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        for scheduler in lr_schedulers:
            if isinstance(scheduler, PerParamGroupLinearWarmupCosineAnnealingRLPLR):
                yield scheduler

    def _on_validation_epoch_end_cos_rlp(self, config: WarmupCosRLPConfig):
        rlp_monitor = self._rlp_metric(config.rlp)
        log.info(f"LR scheduler metrics: {rlp_monitor}")

        metric_value: torch.Tensor | None = None
        for scheduler in self._cos_rlp_schedulers():
            if scheduler.is_in_rlp_stage(self.global_step):
                if metric_value is None:
                    metric_value = rlp_monitor.compute()

                log.info(f"LR scheduler is in RLP mode. RLP metric: {metric_value}")
                scheduler.rlp_step(metric_value)

    def _on_train_batch_start_cos_rlp(self):
        for scheduler in self._cos_rlp_schedulers():
            scheduler.on_new_step(self.global_step)

    @override
    def on_train_batch_start(self, batch: BaseData, batch_idx: int):
        # Check print_every parameter.
        if self.config.debug_print_every is not None:
            if (
                not hasattr(self, "_lora_debug_start_time")
                or (self.global_step % self.config.debug_print_every) == 1
            ):
                self._lora_debug_start_time = time.time()

        match self.config.lr_scheduler:
            case WarmupCosRLPConfig():
                self._on_train_batch_start_cos_rlp()
            case _:
                pass

    @override
    def on_validation_epoch_end(self):
        match self.config.lr_scheduler:
            case WarmupCosRLPConfig() as config:
                self._on_validation_epoch_end_cos_rlp(config)
            case _:
                pass

    @override
    def training_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix=f"train/{self.metric_prefix()}/"):
            preds = self(batch)

            loss = self.compute_losses(batch, preds)
            self._process_batch_dump(batch, loss)

            self.log_dict(self.train_metrics(batch, preds))

            self._lora_debug_print()

            return loss

    @torch.no_grad()
    def _process_batch_dump(self, batch: BaseData, loss: torch.Tensor):
        if (batch_dump := self.config.batch_dump) is None:
            return
        if batch_dump.dump_if_loss_gt is None:
            return

        loss_float = loss.item()
        if loss_float > batch_dump.dump_if_loss_gt:
            log.critical(
                f"Loss {loss_float} is greater than {batch_dump.dump_if_loss_gt}. Dumping batch."
            )

            ActSave(
                {
                    f"batch_dump_if_loss_gt::{k}": v
                    for k, v in {
                        **batch.to_dict(),
                        "loss": loss_float,
                    }.items()
                }
            )

    @override
    def validation_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix=f"val/{self.metric_prefix()}/"):
            preds = self(batch)

            self.log_dict(self.val_metrics(batch, preds))

    @override
    def test_step(self, batch: BaseData, batch_idx: int):
        with self.log_context(prefix=f"test/{self.metric_prefix()}/"):
            preds = self(batch)

            self.log_dict(self.test_metrics(batch, preds))

    def outhead_parameters(self):
        head_params = (
            list(self.graph_outputs.parameters())
            + list(self.node_outputs.parameters())
            + list(self.graph_classification_outputs.parameters())
        )
        return head_params

    def backbone_outhead_parameters(
        self,
    ):
        main_params = list(self.parameters())
        head_params = self.outhead_parameters()
        head_params_set = set(head_params)
        main_params = [p for p in main_params if p not in head_params_set]
        return main_params, head_params

    def _warmup_step(
        self,
        config: RLPWarmupConfig,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
    ):
        # Compute the current step
        match config.step_type:
            case "step":
                current_step = self.global_step
            case "epoch":
                current_step = self.current_epoch
            case _:
                assert_never(config.step_type)

        if current_step > config.steps:
            return

        initial_lr = self.config.optimizer.lr
        lr_scale = min(1.0, float(current_step + 1) / config.steps)
        for pg in optimizer.param_groups:
            pg["lr"] = initial_lr * lr_scale

    @override
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        match self.config.lr_scheduler:
            case RLPConfig(warmup=RLPWarmupConfig() as warmup):
                self._warmup_step(warmup, optimizer)
            case _:
                pass

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def split_parameters(
        self,
        pattern_lists: list[list[str]],
        no_double_counting: bool = True,
        requires_grad_only: bool = True,
    ):
        """
        Splits the parameters of the model into multiple groups based on the provided pattern lists.

        Args:
            pattern_lists (list[list[str]]): A list of pattern lists. Each pattern list contains a set of patterns
                used to match parameter names.
            no_double_counting (bool): If True, parameters that match multiple patterns will only be counted once.
            requires_grad_only (bool): If True, only parameters with requires_grad=True will be considered.

        Returns:
            parameters (list[list[nn.Parameter]]): A list of parameter groups. Each group contains the parameters
                that match the patterns in the corresponding pattern list.
            all_parameters (list[nn.Parameter]): The remaining parameters that do not match any of the patterns.
        """

        matched_parameters = set[nn.Parameter]()
        all_parameters = [
            p for p in self.parameters() if not requires_grad_only or p.requires_grad
        ]

        parameters: list[list[nn.Parameter]] = []
        for patterns in pattern_lists:
            matching = [
                p
                for _, p, _ in self.named_parameters_matching_patterns(
                    patterns,
                    ignored_parameters=matched_parameters,
                    requires_grad_only=requires_grad_only,
                )
            ]

            parameters.append(matching)

            # Remove matching parameters from all_parameters.
            all_parameters = [
                p for p in all_parameters if all(p is not m for m in matching)
            ]

            # If no_double_counting is True, add the matching parameters to the set of matched parameters.
            if no_double_counting:
                matched_parameters.update(matching)

        return parameters, all_parameters

    def _cos_annealing_hparams(
        self, lr_config: WarmupCosRLPConfig, *, lr_initial: float
    ):
        if (warmup_steps := lr_config.warmup_steps) is None:
            if warmup_epochs := lr_config.warmup_epochs:
                assert warmup_epochs >= 0, f"Invalid warmup_epochs: {warmup_epochs}"
                _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                warmup_steps = int(warmup_epochs * num_steps_per_epoch)
            else:
                warmup_steps = 0
        log.critical(f"Computed warmup_steps: {warmup_steps}")

        if not (max_steps := lr_config.max_steps):
            if max_epochs := lr_config.max_epochs:
                _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                max_steps = int(max_epochs * num_steps_per_epoch)
            else:
                max_steps = self.trainer.estimated_stepping_batches
                assert math.isfinite(max_steps), f"{max_steps=} is not finite"
                max_steps = int(max_steps)

        log.critical(f"Computed max_steps: {max_steps}")

        assert (
            lr_config.min_lr_factor > 0 and lr_config.min_lr_factor <= 1
        ), f"Invalid {lr_config.min_lr_factor=}"
        min_lr = lr_initial * lr_config.min_lr_factor

        assert (
            lr_config.warmup_start_lr_factor > 0
            and lr_config.warmup_start_lr_factor <= 1
        ), f"Invalid {lr_config.warmup_start_lr_factor=}"
        warmup_start_lr = lr_initial * lr_config.warmup_start_lr_factor

        lr_scheduler_hparams = dict(
            warmup_epochs=warmup_steps,
            max_epochs=max_steps,
            warmup_start_lr=warmup_start_lr,
            eta_min=min_lr,
            should_restart=lr_config.should_restart,
        )

        return lr_scheduler_hparams

    def _construct_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, config: RLPConfig
    ):
        assert config.monitor is not None, f"{config=}"
        assert config.mode is not None, f"{config=}"

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            patience=config.patience,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps,
            verbose=True,
        )

        return {
            "scheduler": lr_scheduler,
            "monitor": config.monitor,
            "interval": config.interval,
            "frequency": config.frequency,
            "strict": True,
        }

    def configure_optimizers_param_specific_optimizers(
        self, configs: list[ParamSpecificOptimizerConfig]
    ):
        params_list, rest_params = self.split_parameters(
            [c.paremeter_patterns for c in configs]
        )
        optimizer = optimizer_from_config(
            [
                *(
                    (
                        self.config.optimizer if c.optimizer is None else c.optimizer,
                        params,
                        c.name or ",".join(c.paremeter_patterns),
                    )
                    for c, params in zip(configs, params_list)
                    # Ignore empty parameter groups
                    if params
                ),
                (self.config.optimizer, rest_params, "rest"),
            ],
            base=self.config.optimizer,
        )

        out: OptimizerLRScheduler = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        match lr_config:
            case RLPConfig():
                assert all(
                    c.lr_scheduler is None for c in configs
                ), f"lr_scheduler is not None for some configs: {configs=}"

                if (
                    lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
                ) is not None:
                    out["lr_scheduler"] = lr_scheduler
            case WarmupCosRLPConfig():
                param_group_lr_scheduler_settings = [
                    *(
                        self._cos_annealing_hparams(
                            (
                                lr_config
                                if c.lr_scheduler is None
                                or not isinstance(c.lr_scheduler, WarmupCosRLPConfig)
                                else c.lr_scheduler
                            ),
                            lr_initial=param_group["lr"],
                        )
                        for c, param_group in zip(configs, optimizer.param_groups[:-1])
                    ),
                    self._cos_annealing_hparams(
                        lr_config, lr_initial=optimizer.param_groups[-1]["lr"]
                    ),
                ]

                log.critical(f"{param_group_lr_scheduler_settings=}")
                lr_scheduler = PerParamGroupLinearWarmupCosineAnnealingRLPLR(
                    optimizer,
                    param_group_lr_scheduler_settings,
                    lr_config.rlp._to_linear_warmup_cos_rlp_dict(),
                    max_epochs=next(
                        (s["max_epochs"] for s in param_group_lr_scheduler_settings)
                    ),
                )
                out["lr_scheduler"] = {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            case _:
                assert_never(lr_config)

        return out

    def _report_parameters(self):
        trainable_parameters: list[str] = []
        non_trainable_parameters: list[str] = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_parameters.append(name)
            else:
                non_trainable_parameters.append(name)

        trainable_parameters_str = "\n".join(f"\t-{p}" for p in trainable_parameters)
        log.debug(f"Trainable parameters {trainable_parameters_str}")

        non_trainable_parameters_str = "\n".join(
            f"\t-{p}" for p in non_trainable_parameters
        )
        log.debug(f"Non-trainable parameters {non_trainable_parameters_str}")

    @override
    def configure_optimizers(self):
        if self.config.parameter_specific_optimizers is not None:
            out = self.configure_optimizers_param_specific_optimizers(
                self.config.parameter_specific_optimizers
            )
            self._report_parameters()
            return out

        optimizer = optimizer_from_config(
            [(self.config.optimizer, self.parameters())],
        )

        out: OptimizerLRScheduler = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        assert isinstance(
            lr_config, RLPConfig
        ), "Only RLPConfig is supported if `parameter_specific_optimizers` is None"
        if (
            lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
        ) is not None:
            out["lr_scheduler"] = lr_scheduler

        return out

    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    def generate_graphs(
        self,
        data: BaseData,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        pbc: bool,
    ):
        aint_graph = generate_graph(
            data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, pbc=pbc
        )
        aint_graph = self.process_aint_graph(aint_graph)
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

        return data

    def create_dataset(
        self, split: Literal["train", "val", "test"]
    ) -> DatasetType | None:
        match split:
            case "train":
                if (config := self.config.train_dataset) is None:
                    return None
            case "val":
                if (config := self.config.val_dataset) is None:
                    return None
            case "test":
                if (config := self.config.test_dataset) is None:
                    return None
            case _:
                assert_never(split)

        dataset = config.create_dataset()
        dataset = wrap_common_dataset(dataset, config)
        return dataset

    def validate_dataset(self, dataset: DatasetType):
        if self.config.use_balanced_batch_sampler:
            assert isinstance(
                dataset, DatasetWithSizes
            ), f"BalancedBatchSampler requires a DatasetWithSizes, but got {type(dataset)}"

    def _transform_cls_data(self, data: BaseData):
        """
        Transforms the classification targets in the given data object based on the configuration.

        For binary classification targets, the target is converted to a float tensor (i.e., 0.0 or 1.0).
        For multiclass classification targets, the target is converted to a long tensor (which is used as
            the class index by `F.cross_entropy`).

        Args:
            data (BaseData): The data object containing the classification targets.

        Returns:
            BaseData: The transformed data object.
        """
        for target_config in self.config.graph_targets:
            match target_config:
                case GraphBinaryClassificationTargetConfig():
                    if (value := getattr(data, target_config.name, None)) is None:
                        log.warning(f"target {target_config.name} not found in data")
                        continue

                    setattr(data, target_config.name, value.float())
                case GraphMulticlassClassificationTargetConfig():
                    if (value := getattr(data, target_config.name, None)) is None:
                        log.warning(f"target {target_config.name} not found in data")
                        continue

                    setattr(data, target_config.name, value.long())
                case _:
                    pass

        return data

    def _apply_dataset_transforms(self, dataset: DatasetType):
        dataset = DT.transform(dataset, self.data_transform)
        if self.config.normalization:
            dataset = DT.transform(dataset, T.normalize(self.config.normalization))
        dataset = DT.transform(dataset, self._transform_cls_data)
        return dataset

    def train_dataset(self):
        if (dataset := self.create_dataset("train")) is None:
            return None
        self.validate_dataset(dataset)
        dataset = self._apply_dataset_transforms(dataset)
        return dataset

    def val_dataset(self):
        if (dataset := self.create_dataset("val")) is None:
            return None
        self.validate_dataset(dataset)
        dataset = self._apply_dataset_transforms(dataset)
        return dataset

    def test_dataset(self):
        if (dataset := self.create_dataset("test")) is None:
            return None
        self.validate_dataset(dataset)
        dataset = self._apply_dataset_transforms(dataset)
        return dataset

    def distributed_sampler(
        self,
        dataset: Dataset,
        shuffle: bool,
        world_size: int | None = None,
        global_rank: int | None = None,
    ):
        if world_size is None:
            world_size = self.trainer.world_size
        if global_rank is None:
            global_rank = self.trainer.global_rank
        return DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=world_size,
            rank=global_rank,
        )

    @override
    def train_dataloader(self):
        if (dataset := self.train_dataset()) is None:
            raise ValueError("No train dataset")

        sampler = self.distributed_sampler(dataset, shuffle=True)
        if not self.config.use_balanced_batch_sampler:
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.config.batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        else:
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
            )

        return data_loader

    @override
    def val_dataloader(self):
        if (dataset := self.val_dataset()) is None:
            raise ValueError("No val dataset")

        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_val)
        batch_size = self.config.eval_batch_size or self.config.batch_size
        if not self.config.use_balanced_batch_sampler:
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        else:
            batch_sampler = BalancedBatchSampler(
                sampler,
                batch_size=batch_size,
                device=self.device,
            )
            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        return data_loader

    @override
    def test_dataloader(self):
        if (dataset := self.test_dataset()) is None:
            raise ValueError("No test  dataset")

        sampler = self.distributed_sampler(dataset, shuffle=self.config.shuffle_test)
        batch_size = self.config.eval_batch_size or self.config.batch_size
        if not self.config.use_balanced_batch_sampler:
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        else:
            batch_sampler = BalancedBatchSampler(
                sampler,
                batch_size=batch_size,
                device=self.device,
            )
            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
            )
        return data_loader

    def data_transform(self, data: BaseData):
        return data

    def collate_fn(self, data_list: list[BaseData]):
        return Batch.from_data_list(data_list)

    def debug_get_distributed_batch(
        self,
        dataset_fn: Callable[[], DatasetType],
        step_idx: int,
        world_size: int,
    ):
        dataloaders = [
            DataLoader(
                dataset_fn(),
                batch_size=self.config.batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.config.num_workers,
                sampler=self.distributed_sampler(
                    dataset_fn(), shuffle=True, world_size=world_size, global_rank=i
                ),
            )
            for i in range(world_size)
        ]
        dataloader_iters = [iter(dl) for dl in dataloaders]

        for _ in range(step_idx):
            for dl_iter in dataloader_iters:
                next(dl_iter)

        return [cast(BaseData, next(dl_iter)) for dl_iter in dataloader_iters]
