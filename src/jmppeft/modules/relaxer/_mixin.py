from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Literal

import ll
import torch
from ase import Atoms
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from typing_extensions import TypeVar, assert_never

from ._relaxer import Relaxer


class RelaxerConfig(ll.TypedConfig):
    enabled: bool = False

    energy_key: str = "y"
    """Key for the energy in the graph data."""

    force_key: str = "force"
    """Key for the forces in the node data."""

    stress_key: str | None = None
    """Key for the stress in the graph data (or `None` if stress is not computed)."""

    optimizer: Literal[
        "FIRE",
        "BFGS",
        "LBFGS",
        "LBFGSLineSearch",
        "MDMin",
        "SciPyFminCG",
        "SciPyFminBFGS",
        "BFGSLineSearch",
    ] = "FIRE"
    """Optimizer to use for relaxation."""

    optimizer_kwargs: Mapping[str, Any] = {}
    """Keyword arguments for the optimizer"""

    relax_cell: bool = False
    """Whether to relax the cell."""

    stress_weight: float = 0.01
    """Weight for the stress loss."""

    fmax: float = 0.1
    """
    Total force tolerance for relaxation convergence.
        Here fmax is a sum of force and stress forces
        (if stress is computed).
    """

    steps: int = 500
    """
    Maximum number of steps for relaxation.
    """

    traj_file: Path | None = None
    """
    The trajectory file for saving. (If None, no trajectory is saved.)
    """

    interval: int = 1
    """
    The step interval for saving the trajectories
    """

    verbose: bool = False
    """
    Whether to print the relaxation progress.
    """

    @property
    def optimizer_cls(self):
        match self.optimizer:
            case "FIRE":
                from ase.optimize.fire import FIRE

                return FIRE
            case "BFGS":
                from ase.optimize.bfgs import BFGS

                return BFGS
            case "LBFGS":
                from ase.optimize.lbfgs import LBFGS

                return LBFGS
            case "LBFGSLineSearch":
                from ase.optimize.lbfgs import LBFGSLineSearch

                return LBFGSLineSearch
            case "MDMin":
                from ase.optimize.mdmin import MDMin

                return MDMin
            case "SciPyFminCG":
                from ase.optimize.sciopt import SciPyFminCG

                return SciPyFminCG
            case "SciPyFminBFGS":
                from ase.optimize.sciopt import SciPyFminBFGS

                return SciPyFminBFGS
            case "BFGSLineSearch":
                from ase.optimize.bfgslinesearch import BFGSLineSearch

                return BFGSLineSearch
            case _:
                assert_never(self.optimizer)

    def __bool__(self):
        return self.enabled


TConfig = TypeVar("TConfig", bound=ll.BaseConfig)


class LightningModuleRelaxerMixin(ll.LightningModuleBase[TConfig], ABC):
    @abstractmethod
    def relaxer_config(self) -> RelaxerConfig | None: ...

    @abstractmethod
    def relaxer_collate_fn(self, data: Sequence[Data]) -> Batch: ...

    def _relaxer_atoms_to_graph(self, atoms: Atoms) -> Batch:
        assert self.relaxer_config(), "Relaxer is not enabled in the config."

        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float)
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
        batch = self.relaxer_collate_fn([data])

        return batch

    def _relaxer_graph_to_atoms(self, data_or_batch: Data | Batch) -> Atoms:
        assert self.relaxer_config(), "Relaxer is not enabled in the config."

        graph = data_or_batch
        if isinstance(graph, Batch):
            data_list = graph.to_data_list()
            assert len(data_list) == 1, "Batch size must be 1 for relaxation."

            graph = data_list[0]

        atomic_numbers = graph.atomic_numbers.detach().cpu().numpy()
        pos = graph.pos.detach().cpu().numpy()
        cell = graph.cell.detach().cpu().numpy()

        atoms = Atoms(numbers=atomic_numbers, positions=pos, cell=cell, pbc=True)

        # Apply constraints based on the fixed flag
        if (fixed := getattr(graph, "fixed", None)) is not None:
            assert fixed.dtype == torch.bool, "Fixed flag must be boolean."
            fixed = fixed.detach().cpu().numpy()

            from ase.constraints import FixAtoms

            atoms.set_constraint(FixAtoms(mask=fixed))

        return atoms

    def _relaxer_potential(self, graph: Batch, include_stresses: bool = True):
        assert (c := self.relaxer_config()), "Relaxer is not enabled in the config."

        # First, move the graph to the model's device
        graph = graph.to(self.device)

        # Compute the energy and forces
        out: dict[str, torch.Tensor] = self(graph)
        energy = out[c.energy_key].detach().float().cpu().numpy()
        forces = out[c.force_key].detach().float().cpu().numpy()

        if include_stresses:
            assert (
                c.stress_key is not None
            ), "Stress key is not set in the relaxer config."

            stress = out[c.stress_key].detach().float().cpu().numpy()
            return energy, forces, stress
        else:
            return energy, forces

    @property
    @cache
    def _relaxer(self):
        if (c := self.relaxer_config()) is None:
            raise ValueError("Relaxer is not enabled in the config.")

        return Relaxer(
            potential=self._relaxer_potential,
            graph_converter=self._relaxer_atoms_to_graph,
            optimizer_cls=c.optimizer_cls,
            relax_cell=c.relax_cell,
            stress_weight=c.stress_weight,
        )

    def relaxer_validate_dataloader(self, dl: DataLoader):
        # Make sure the batch size is 1, as ASE doesn't support batched relaxation
        assert dl.batch_size == 1, "Batch size must be 1 for relaxation."

    def relax(self, atoms: Atoms | Data | Batch):
        assert (c := self.relaxer_config()), "Relaxer is not enabled in the config."

        if not isinstance(atoms, Atoms):
            atoms = self._relaxer_graph_to_atoms(atoms)

        return self._relaxer.relax(
            atoms,
            fmax=c.fmax,
            steps=c.steps,
            traj_file=str(c.traj_file.absolute()) if c.traj_file else None,
            interval=c.interval,
            verbose=c.verbose,
            **c.optimizer_kwargs,
        )
