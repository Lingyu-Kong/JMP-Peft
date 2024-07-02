from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import ExitStack
from functools import cache, partial
from logging import getLogger
from pathlib import Path
from typing import Any, Generic, Literal, cast

import ll
import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.prediction_writer import BasePredictionWriter
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData, Data
from typing_extensions import TypeVar, override

from ...models.gemnet.backbone import GOCBackboneOutput
from ...modules.loss import L2MAELossConfig, LossConfig, MAELossConfig
from ...modules.relaxer import RelaxationOutput, Relaxer, RelaxerConfig
from ...modules.transforms.normalize import denormalize_batch
from .base import FinetuneConfigBase, FinetuneModelBase
from .output_head import (
    GradientForcesTargetConfig,
    GradientOutputHeadInput,
    GradientStressTargetConfig,
    GraphScalarTargetConfig,
    GraphTargetConfig,
    NodeTargetConfig,
    NodeVectorTargetConfig,
    OutputHeadInput,
)

log = getLogger(__name__)


class SkipBatch(Exception):
    pass


class RelaxationConfig(ll.TypedConfig):
    validation: RelaxerConfig | None = None
    """Relaxer configuration for validation. If None, relaxer is disabled for validation."""

    test: RelaxerConfig | None = None
    """Relaxer configuration for testing. If None, relaxer is disabled for testing."""

    predict: RelaxerConfig | None = None
    """Relaxer configuration for predicting. If None, relaxer is disabled for predicting."""

    energy_key: str = "y"
    """Key for the energy in the model's output."""

    force_key: str = "force"
    """Key for the forces in the model's output."""

    stress_key: str | None = None
    """Key for the stress in the model's output. If None, stress is not computed."""

    relaxed_energy_key: str = "y_above_hull"
    """Key for the relaxed energy in the PyG `Batch` object."""

    relaxed_energy_per_atom: bool = True
    """If true, relaxed energy metrics are reported in `eV/atom` units. Otherwise, they are reported in `eV` units."""

    add_dummy_y_and_force: bool = True
    """Whether to add dummy `y` and `force` keys to the batch if they are not present.
    This is to prevent errors when the model does not output `y` and `force` during validation and testing.
    """

    relaxed_energy_linref_path: Path | None = None
    """Path to the linear reference energies for the relaxed energies.
    If set, we assume that the dataset's `y_relaxed` is the linear reference energy
    and will undo the linear reference transformation to get the total energy (for metrics).
    """

    use_chgnet_for_relaxed_energy: bool = False
    """Whether to use the `chgnet` model to compute the relaxed energy."""

    output_dir: Path | None = None
    """Directory to save the predictions of the relaxation model during validation and testing."""

    def __bool__(self):
        return (
            self.validation is not None
            or self.test is not None
            or self.predict is not None
        )


class EnergyForcesConfigBase(FinetuneConfigBase):
    graph_targets: list[GraphTargetConfig] = [
        GraphScalarTargetConfig(name="y", loss_coefficient=1.0),
    ]
    node_targets: list[NodeTargetConfig] = [
        GradientForcesTargetConfig(
            name="force",
            energy_name="y",
            loss_coefficient=100.0,
        ),
    ]

    compute_graphs_on_cpu: bool = False

    relaxation: RelaxationConfig = RelaxationConfig()
    """Relaxation configuration for validation and testing."""

    sanity_check_mace: bool = False

    def energy_config_(self):
        self.graph_targets = [
            GraphScalarTargetConfig(name="y", loss_coefficient=1.0),
        ]
        self.node_targets = []

    def forces_config_(self, *, gradient: bool, coefficient: float = 100.0):
        self.graph_targets = (
            [
                GraphScalarTargetConfig(name="y", loss_coefficient=0.0),
            ]
            if gradient
            else []
        )
        self.node_targets = [
            GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=coefficient,
            )
            if gradient
            else NodeVectorTargetConfig(name="force", loss_coefficient=coefficient),
        ]

        if gradient:
            self.trainer.inference_mode = False

    def energy_forces_config_(
        self,
        *,
        gradient: bool,
        energy_coefficient: float = 1.0,
        energy_loss: LossConfig = MAELossConfig(),
        energy_pooling: Literal["mean", "sum"] = "mean",
        force_coefficient: float = 100.0,
        force_loss: LossConfig = L2MAELossConfig(),
    ):
        self.graph_targets = [
            GraphScalarTargetConfig(
                name="y",
                loss_coefficient=energy_coefficient,
                loss=energy_loss,
                reduction=energy_pooling,
            ),
        ]
        self.node_targets = [
            GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=force_coefficient,
                loss=force_loss,
            )
            if gradient
            else NodeVectorTargetConfig(
                name="force", loss_coefficient=force_coefficient, loss=force_loss
            ),
        ]

        if gradient:
            self.trainer.inference_mode = False

    def energy_forces_stress_config_(
        self,
        *,
        gradient: bool,
        energy_coefficient: float = 1.0,
        energy_loss: LossConfig = MAELossConfig(),
        energy_pooling: Literal["mean", "sum"] = "mean",
        force_coefficient: float = 100.0,
        force_loss: LossConfig = L2MAELossConfig(),
        stress_coefficient: float = 1.0,
        stress_loss: LossConfig = MAELossConfig(),
    ):
        if not gradient:
            raise ValueError("Stress target requires gradient forces.")

        self.graph_targets = [
            GraphScalarTargetConfig(
                name="y",
                loss_coefficient=energy_coefficient,
                loss=energy_loss,
                reduction=energy_pooling,
            ),
            GradientStressTargetConfig(
                name="stress",
                energy_name="y",
                loss_coefficient=stress_coefficient,
                loss=stress_loss,
                forces=True,
            ),
        ]
        self.node_targets = [
            GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=force_coefficient,
                loss=force_loss,
                use_stress_forces=True,
            )
        ]

        if gradient:
            self.trainer.inference_mode = False

    def supports_inference_mode(self):
        return all(t.supports_inference_mode() for t in self.node_targets)

    @property
    def gradient_force_target(self):
        return next(
            (
                target
                for target in self.node_targets
                if isinstance(target, GradientForcesTargetConfig)
            ),
            None,
        )

    @override
    def __post_init__(self):
        super().__post_init__()

        if not self.supports_inference_mode():
            assert not self.trainer.inference_mode, "`config.trainer.inference_mode` is True, but the model does not support inference mode."

        if any(
            isinstance(target, NodeVectorTargetConfig) for target in self.node_targets
        ):
            # assert (
            #     self.trainer.inference_mode
            # ), "NodeVectorTargetConfig requires inference mode to be enabled."
            assert (
                self.backbone.regress_forces
            ), "NodeVectorTargetConfig requires `backbone.regress_forces` to be True."
            assert (
                self.backbone.direct_forces
            ), "NodeVectorTargetConfig requires `backbone.direct_forces` to be True."

        if self.sanity_check_mace:
            assert (
                self.batch_size == 1
            ), "Sanity check MACE only works with batch size 1."
            if self.eval_batch_size:
                assert (
                    self.eval_batch_size == 1
                ), "Sanity check MACE only works with batch size 1."


TConfig = TypeVar("TConfig", bound=EnergyForcesConfigBase, infer_variance=True)


class _Writer(BasePredictionWriter):
    def __init__(self):
        super().__init__(write_interval="batch")

    @override
    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        base_path = cast(
            ll.LightningModuleBase[ll.BaseConfig], pl_module
        ).config.subdirectory("predictions")
        base_path.mkdir(exist_ok=True, parents=True)
        out_pred = (
            base_path / f"predif=ction_rank{trainer.global_rank}_batch{batch_idx}.pt"
        )
        torch.save(prediction, out_pred)


class EnergyForcesModelBase(
    FinetuneModelBase[TConfig],
    nn.Module,
    ABC,
    Generic[TConfig],
):
    @override
    def __init__(self, hparams: TConfig):
        super().__init__(hparams)

        if self.config.relaxation:
            self.register_callback(lambda: _Writer())

    @override
    def _construct_model(self):
        if self.config.sanity_check_mace:
            from mace.calculators import mace_mp

            self._mace_calc = mace_mp(
                "https://tinyurl.com/5yyxdm76", device="cuda", default_dtype="float64"
            )
        else:
            super()._construct_model()

    @override
    def construct_output_heads(self):
        self.graph_outputs = ll.nn.TypedModuleDict(
            {
                target.name: target.construct_output_head(
                    self.config.output,
                    self.config.backbone.emb_size_atom,
                    self.config.backbone.emb_size_edge,
                    self.config.activation_cls,
                )
                for target in self.config.graph_targets
            },
            key_prefix="ft_mlp_",
        )
        self.node_outputs = ll.nn.TypedModuleDict(
            {
                target.name: target.construct_output_head(
                    self.config.output,
                    self.config.backbone.emb_size_atom,
                    self.config.backbone.emb_size_edge,
                    self.config.activation_cls,
                )
                for target in self.config.node_targets
            },
            key_prefix="ft_mlp_",
        )

    def _forward(self, data: BaseData):
        preds: dict[str, torch.Tensor] = {}

        with ExitStack() as stack:
            # Enter all the necessary contexts for output heads.
            # Right now, this is only for gradient forces, which
            #   requires torch.inference_mode(False), torch.enable_grad,
            #   and data.pos.requires_grad_(True).
            for target in self.config.targets:
                stack.enter_context(target.model_forward_context(data))

            # Generate graphs on the GPU
            try:
                if not self.config.compute_graphs_on_cpu:
                    data = self.generate_graphs_transform(data)
                else:
                    data = self.postprocess_graphs_gpu(data)
            except Exception as e:
                # If this is a CUDA error, rethrow it
                if "CUDA" in str(data):
                    raise

                # Otherwise, log the error and skip the batch
                log.error(f"Error generating graphs: {e}")
                raise SkipBatch()

            # Run the backbone
            atomic_numbers = data.atomic_numbers - 1
            h = self.embedding(atomic_numbers)  # (N, d_model)
            out = cast(GOCBackboneOutput, self.backbone(data, h=h))

            graph_preds: dict[str, torch.Tensor] = {}
            node_preds: dict[str, torch.Tensor] = {}
            output_head_input = {
                "backbone_output": out,
                "data": data,
                "graph_preds": graph_preds,
                "node_preds": node_preds,
            }
            for target, module in self.graph_outputs.items():
                graph_preds[target] = module(output_head_input)
            preds.update(graph_preds)

            for target, module in self.node_outputs.items():
                node_preds[target] = module(output_head_input)
            preds.update(node_preds)

        return preds

    def _mace_atoms_to_graph(self, atoms: Atoms) -> Batch:
        """
        Convert the given ASE `Atoms` object to a PyG `Batch` object.

        Args:
            atoms (Atoms): The input `Atoms` object.

        Returns:
            Batch: The converted `Batch` object.
        """
        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float).view(1, 3, 3)
        tags = (2 * torch.ones_like(atomic_numbers)).long()
        fixed = torch.zeros_like(tags, dtype=torch.bool)
        natoms = len(atoms)

        data = Data.from_dict(
            {
                "atomic_numbers": atomic_numbers,
                "pos": pos,
                "cell": cell,
                "tags": tags,
                "fixed": fixed,
                "natoms": natoms,
            }
        )
        batch = self.collate_fn([data])

        return batch

    def _mace_graph_to_atoms(self, data_or_batch: Data | Batch) -> Atoms:
        """
        Convert a graph representation to an `Atoms` object.

        Args:
            data_or_batch (Data | Batch): The input graph or batch of graphs.

        Returns:
            Atoms: The converted `Atoms` object.

        Raises:
            AssertionError: If the batch size is not 1 for relaxation.
            AssertionError: If the fixed flag is not a boolean tensor.
        """
        graph = data_or_batch
        if isinstance(graph, Batch):
            data_list = graph.to_data_list()
            assert len(data_list) == 1, "Batch size must be 1 for relaxation."

            graph = data_list[0]

        atomic_numbers = graph.atomic_numbers.detach().cpu().numpy()
        pos = graph.pos.detach().cpu().numpy()
        cell = graph.cell.detach().cpu().view(3, 3).numpy()

        atoms = Atoms(numbers=atomic_numbers, positions=pos, cell=cell, pbc=True)

        # Apply constraints based on the fixed flag
        if (fixed := getattr(graph, "fixed", None)) is not None:
            assert fixed.dtype == torch.bool, "Fixed flag must be boolean."
            fixed = fixed.detach().cpu().numpy()

            from ase.constraints import FixAtoms

            atoms.set_constraint(FixAtoms(mask=fixed))

        return atoms

    def _mace_forward(self, data: BaseData):
        # Convert the data to an ASE Atoms object
        atoms = self._mace_graph_to_atoms(data)

        # Run the MACE calculator
        atoms.calc = self._mace_calc

        # Get the energy and forces
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # Convert the results to PyTorch tensors
        energy = torch.tensor([energy], dtype=torch.float, device=self.device)
        forces = torch.from_numpy(forces).to(self.device)

        return {"y": energy, "force": forces}

    @override
    def forward(self, data: BaseData):
        if self.config.sanity_check_mace:
            with torch.enable_grad():
                return self._mace_forward(data)

        return self._forward(data)

    @abstractmethod
    def generate_graphs_transform(self, data: BaseData) -> BaseData: ...

    def postprocess_graphs_gpu(self, data: BaseData) -> BaseData:
        raise NotImplementedError("`postprocess_graphs_gpu` must be implemented.")

    def _relaxer_forward(
        self,
        batch: Batch,
        initial_graph: Batch,
        *,
        config: RelaxerConfig,
    ):
        out = self(batch)
        _, out = denormalize_batch(initial_graph.clone(), out)

        energy = out[self.config.relaxation.energy_key]
        force = out[self.config.relaxation.force_key]
        stress = None
        if config.compute_stress and self.config.relaxation.stress_key is not None:
            stress = out[self.config.relaxation.stress_key]

        # If the relaxed energy is per atom, we need to multiply it by the number of atoms
        energy = self._process_energy(energy, batch)

        return energy, force, stress

    @cache
    def _relaxer_energy_linref(self):
        if self.config.relaxation.relaxed_energy_linref_path is None:
            return None

        return torch.tensor(
            np.load(self.config.relaxation.relaxed_energy_linref_path),
            dtype=torch.float,
            device=self.device,
        )

    def _process_energy(self, energy: torch.Tensor, batch: BaseData):
        # Undo the linear reference transformation
        if (linref := self._relaxer_energy_linref()) is not None:
            energy = energy + linref[batch.atomic_numbers].sum()

        # Per atom energies
        if self.config.relaxation.relaxed_energy_per_atom:
            energy = energy / int(batch.atomic_numbers.numel())
        return energy

    def _relaxer_y_relaxed(self, batch: BaseData):
        # First, denormalize the batch
        batch, _ = denormalize_batch(batch.clone())

        key = self.config.relaxation.relaxed_energy_key
        if (y_relaxed := getattr(batch, key, None)) is None:
            raise AttributeError(f"Batch does not have `{key}` attribute.")

        # If the relaxed energy is per atom, we need to multiply it by the number of atoms
        y_relaxed = self._process_energy(y_relaxed, batch)

        return y_relaxed

    @cache
    def _relaxer_chgnet(self):
        from chgnet.graph.converter import CrystalGraphConverter
        from chgnet.model import CHGNet

        model = CHGNet.load().to(self.device)
        model.eval()

        converter = CrystalGraphConverter(
            atom_graph_cutoff=12.0,
            bond_graph_cutoff=3.0,
        ).to(self.device)
        converter.eval()
        return model, converter

    def _relaxer_chgnet_relaxation_output_to_energy(self, output: RelaxationOutput):
        # Get the model
        model, converter = self._relaxer_chgnet()

        # Convert the final structure to a CrystalGraph
        from chgnet.graph import CrystalGraph
        from pymatgen.core import Structure

        final_structure = cast(Structure, output.final_structure)
        graph = cast(CrystalGraph, converter(final_structure))

        # Compute the energy
        with torch.no_grad(), torch.inference_mode():
            out = model([graph.to(self.device)], task="e")
            energy = out["e"].view(-1)

        # get_e_form_per_atom
        if True:
            from matbench_discovery.energy import get_e_form_per_atom

            energy = torch.tensor(
                get_e_form_per_atom(
                    {
                        "energy": energy.item(),
                        "composition": final_structure.composition,
                    }
                ),
                dtype=torch.float,
                device=self.device,
            )

        return energy

    @override
    def predict_step(self, batch, batch_idx):
        if self.config.relaxation.predict is None:
            return super().predict_step(batch, batch_idx)

        assert self.predict_relaxer is not None, "Relaxer must be initialized"
        relax_out = self.predict_relaxer._relax(batch)
        # relaxed_energy = self.predict_relaxer.relaxation_output_to_energy(relax_out)

        batch_dict: dict[str, Any] = {}
        for key, value in batch.to_dict().items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            batch_dict[key] = value

        return {
            "relax_out": relax_out.as_dict(),
            "batch": batch_dict,
            # "relaxed_energy": relaxed_energy.detach().cpu().numpy(),
        }

    @override
    def on_predict_epoch_start(self):
        super().on_predict_epoch_start()

        self.predict_relaxer = None
        if (config := self.config.relaxation.predict) is not None:
            self.predict_relaxer = Relaxer(
                config,
                partial(self._relaxer_forward, config=config),
                self.collate_fn,
                self.device,
                relaxation_output_to_energy=self._relaxer_relaxation_output_to_energy,
            )

    @property
    def _relaxer_relaxation_output_to_energy(self):
        if self.config.relaxation.use_chgnet_for_relaxed_energy:
            return self._relaxer_chgnet_relaxation_output_to_energy
        return None

    @override
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        self.val_relaxer = None
        if (config := self.config.relaxation.validation) is not None:
            self.val_relaxer = Relaxer(
                config,
                partial(self._relaxer_forward, config=config),
                self.collate_fn,
                self.device,
                relaxation_output_to_energy=self._relaxer_relaxation_output_to_energy,
            )

    @override
    def validation_step(self, batch: BaseData, batch_idx: int):
        if self.config.relaxation.validation is None:
            return super().validation_step(batch, batch_idx)

        assert self.val_relaxer is not None, "Relaxer must be initialized"
        with self.log_context(prefix=f"val/{self.metric_prefix()}/"):
            self.val_relaxer.step(
                batch,
                self._relaxer_y_relaxed(batch),
            )

    @override
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        if self.config.relaxation.validation is not None:
            assert self.val_relaxer is not None, "Relaxer must be initialized"

            # Compute the relaxation metrics and report them.
            with self.log_context(prefix=f"val/relax/{self.metric_prefix()}/"):
                self.log_dict(self.val_relaxer.compute_metrics(self), on_epoch=True)

    @override
    def on_test_epoch_start(self):
        super().on_test_epoch_start()

        self.test_relaxer = None
        if (config := self.config.relaxation.test) is not None:
            self.test_relaxer = Relaxer(
                config,
                partial(self._relaxer_forward, config=config),
                self.collate_fn,
                self.device,
                relaxation_output_to_energy=self._relaxer_relaxation_output_to_energy,
            )

    @override
    def test_step(self, batch: BaseData, batch_idx: int):
        if self.config.relaxation.test is None:
            return super().test_step(batch, batch_idx)

        assert self.test_relaxer is not None, "Relaxer must be initialized"
        with self.log_context(prefix=f"test/{self.metric_prefix()}/"):
            self.test_relaxer.step(
                batch,
                self._relaxer_y_relaxed(batch),
            )

    @override
    def on_test_epoch_end(self):
        super().on_test_epoch_end()

        if self.config.relaxation.test is not None:
            assert self.test_relaxer is not None, "Relaxer must be initialized"

            # Compute the relaxation metrics and report them.
            with self.log_context(prefix=f"test/relax/{self.metric_prefix()}/"):
                self.log_dict(self.test_relaxer.compute_metrics(self), on_epoch=True)

    @override
    def data_transform(self, data: BaseData):
        if self.config.relaxation and self.config.relaxation.add_dummy_y_and_force:
            data.y = 0.0
            data.force = torch.zeros_like(data.pos)

        data = super().data_transform(data)

        if self.config.compute_graphs_on_cpu:
            data = self.generate_graphs_transform(data)

        return data
