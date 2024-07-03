from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypedDict

import ll
import torch
from ase import Atoms
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from typing_extensions import NotRequired, assert_never

from ._relaxer import RelaxationOutput
from ._relaxer import Relaxer as _Relaxer


class RelaxerConfig(ll.TypedConfig):
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

    compute_stress: bool = False
    """Whether to compute the stress."""

    stress_weight: float = 0.01
    """Weight for the stress loss."""

    fmax: float = 0.05
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

    stability_threshold: float = 0.0
    """
    Threshold for stable metrics, used to compute the relaxation metrics.
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


def default_relaxation_output_to_energy(output: RelaxationOutput) -> torch.Tensor:
    return torch.tensor(output.trajectory.energies[-1], dtype=torch.float)


class ModelOutput(TypedDict):
    energy: torch.Tensor
    forces: torch.Tensor
    stress: NotRequired[torch.Tensor]


class Relaxer:
    def __init__(
        self,
        config: RelaxerConfig,
        model: Callable[[Batch], ModelOutput],
        collate_fn: Callable[[list[BaseData]], Batch],
        device: torch.device,
    ):
        super().__init__()

        self.config = config
        self.model = model
        self.collate_fn = collate_fn
        self.device = device

    def _atoms_to_graph(self, atoms: Atoms) -> Batch:
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

    def _graph_to_atoms(self, data_or_batch: Data | Batch) -> Atoms:
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

    def _potential(self, graph: Batch):
        """
        Compute the potential energy, forces, and stresses of the given graph.

        Args:
            initial_graph (Batch): The initial graph (from the original dataset).
            graph (Batch): The input graph.

        Returns:
            tuple: A tuple containing the potential energy and forces. If `config.compute_stress`
            is True, the tuple also includes the stresses.

        Raises:
            AssertionError: If the stress key is not set in the relaxer config.
        """
        # First, move the graph to the model's device
        graph = graph.to(self.device)

        # Compute the energy and forces
        model_out = self.model(graph)
        energy = model_out["energy"]
        forces = model_out["forces"]
        stress = model_out.get("stress")
        energy = energy.detach().float().cpu()
        forces = forces.detach().float().cpu()

        if self.config.compute_stress:
            assert stress is not None, "Stress key must be set in the relaxer config."
            stress = stress.detach().float().cpu()
            return energy, forces, stress
        else:
            return energy, forces

    def relax(self, graph: Batch):
        """
        Perform relaxation on the given atoms.

        Args:
            graph (Batch): The input graph.

        Returns:
            RelaxationOutput: The relaxation output.

        Raises:
            None

        """
        atoms = self._graph_to_atoms(graph)

        relaxer = _Relaxer(
            potential=self._potential,
            graph_converter=self._atoms_to_graph,
            optimizer_cls=self.config.optimizer_cls,
            relax_cell=self.config.relax_cell,
            compute_stress=self.config.compute_stress,
            stress_weight=self.config.stress_weight,
        )
        return relaxer.relax(
            atoms,
            fmax=self.config.fmax,
            steps=self.config.steps,
            traj_file=str(self.config.traj_file.absolute())
            if self.config.traj_file
            else None,
            interval=self.config.interval,
            verbose=self.config.verbose,
            **self.config.optimizer_kwargs,
        )
