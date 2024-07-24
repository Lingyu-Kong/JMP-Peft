from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cache, partial
from logging import getLogger
from pathlib import Path
from typing import Any, Generic, cast

import nshtrainer.ll as ll
import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.prediction_writer import BasePredictionWriter
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar, override

from ....modules.relaxer import RelaxationOutput, Relaxer, RelaxerConfig
from ....modules.transforms.normalize import denormalize_batch

log = getLogger(__name__)


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
            base_path / f"prediction_rank{trainer.global_rank}_batch{batch_idx}.pt"
        )
        torch.save(prediction, out_pred)


TConfig = TypeVar("TConfig", bound=ll.BaseConfig, infer_variance=True)


class RelaxationModel(
    ll.LightningModuleBase[TConfig],
    ABC,
    Generic[TConfig],
):
    @abstractmethod
    def relax_config(self) -> RelaxationConfig | None: ...

    @property
    def _rc(self) -> RelaxationConfig:
        if (config := self.relax_config()) is None:
            config = RelaxationConfig(validation=None, test=None, predict=None)
        return config

    @override
    def __init__(self, hparams: TConfig):
        super().__init__(hparams)

        if self._rc:
            self.register_callback(lambda: _Writer())

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

        energy = out[self._rc.energy_key]
        force = out[self._rc.force_key]
        stress = None
        if config.compute_stress and self._rc.stress_key is not None:
            stress = out[self._rc.stress_key]

        # If the relaxed energy is per atom, we need to multiply it by the number of atoms
        energy = self._relaxer_process_energy(energy, batch)

        return energy, force, stress

    @cache
    def _relaxer_energy_linref(self):
        if self._rc.relaxed_energy_linref_path is None:
            return None

        return torch.tensor(
            np.load(self._rc.relaxed_energy_linref_path),
            dtype=torch.float,
            device=self.device,
        )

    def _relaxer_process_energy(self, energy: torch.Tensor, batch: BaseData):
        # Undo the linear reference transformation
        if (linref := self._relaxer_energy_linref()) is not None:
            energy = energy + linref[batch.atomic_numbers].sum()

        # Per atom energies
        if self._rc.relaxed_energy_per_atom:
            energy = energy / int(batch.atomic_numbers.numel())
        return energy

    def _relaxer_y_relaxed(self, batch: BaseData):
        # First, denormalize the batch
        batch, _ = denormalize_batch(batch.clone())

        key = self._rc.relaxed_energy_key
        if (y_relaxed := getattr(batch, key, None)) is None:
            raise AttributeError(f"Batch does not have `{key}` attribute.")

        # If the relaxed energy is per atom, we need to multiply it by the number of atoms
        y_relaxed = self._relaxer_process_energy(y_relaxed, batch)

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
        if self._rc.predict is None:
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
        if (config := self._rc.predict) is not None:
            self.predict_relaxer = Relaxer(
                config,
                partial(self._relaxer_forward, config=config),
                self.collate_fn,
                self.device,
                relaxation_output_to_energy=self._relaxer_relaxation_output_to_energy,
            )

    @property
    def _relaxer_relaxation_output_to_energy(self):
        if self._rc.use_chgnet_for_relaxed_energy:
            return self._relaxer_chgnet_relaxation_output_to_energy
        return None

    @override
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        self.val_relaxer = None
        if (config := self._rc.validation) is not None:
            self.val_relaxer = Relaxer(
                config,
                partial(self._relaxer_forward, config=config),
                self.collate_fn,
                self.device,
                relaxation_output_to_energy=self._relaxer_relaxation_output_to_energy,
            )

    @override
    def validation_step(self, batch: BaseData, batch_idx: int):
        if self._rc.validation is None:
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

        if self._rc.validation is not None:
            assert self.val_relaxer is not None, "Relaxer must be initialized"

            # Compute the relaxation metrics and report them.
            with self.log_context(prefix=f"val/relax/{self.metric_prefix()}/"):
                self.log_dict(self.val_relaxer.compute_metrics(self), on_epoch=True)

    @override
    def on_test_epoch_start(self):
        super().on_test_epoch_start()

        self.test_relaxer = None
        if (config := self._rc.test) is not None:
            self.test_relaxer = Relaxer(
                config,
                partial(self._relaxer_forward, config=config),
                self.collate_fn,
                self.device,
                relaxation_output_to_energy=self._relaxer_relaxation_output_to_energy,
            )

    @override
    def test_step(self, batch: BaseData, batch_idx: int):
        if self._rc.test is None:
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

        if self._rc.test is not None:
            assert self.test_relaxer is not None, "Relaxer must be initialized"

            # Compute the relaxation metrics and report them.
            with self.log_context(prefix=f"test/relax/{self.metric_prefix()}/"):
                self.log_dict(self.test_relaxer.compute_metrics(self), on_epoch=True)
