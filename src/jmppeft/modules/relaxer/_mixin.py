from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import ll
import numpy as np
import torch
from ase import Atoms
from lightning.pytorch import LightningModule
from matbench_discovery.energy import get_e_form_per_atom
from matbench_discovery.metrics import STABILITY_THRESHOLD, stable_metrics
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from typing_extensions import assert_never

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

    stability_threshold: float = STABILITY_THRESHOLD
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


@dataclass
class RelaxationEpochState:
    y_pred: list[float] = field(default_factory=lambda: [])
    y_true: list[float] = field(default_factory=lambda: [])


@runtime_checkable
class RelaxationOutputToEnergy(Protocol):
    def __call__(self, output: RelaxationOutput) -> torch.Tensor: ...


def _default_relaxation_output_to_energy(output: RelaxationOutput) -> torch.Tensor:
    return torch.tensor(output.trajectory.energies[-1], dtype=torch.float)


class Relaxer:
    def initialize_state(self):
        """
        Initializes the state for relaxation epoch.

        Returns:
            RelaxationEpochState: The initialized relaxation epoch state.
        """
        return RelaxationEpochState()

    def __init__(
        self,
        config: RelaxerConfig,
        model: Callable[
            [Batch, Batch], tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ],
        collate_fn: Callable[[list[BaseData]], Batch],
        device: torch.device,
        relaxation_output_to_energy: RelaxationOutputToEnergy | None = None,
        state: RelaxationEpochState | None = None,
    ):
        super().__init__()

        self.config = config
        self.model = model
        self.collate_fn = collate_fn
        self.device = device
        self.relaxation_output_to_energy = (
            _default_relaxation_output_to_energy
            if relaxation_output_to_energy is None
            else relaxation_output_to_energy
        )
        self.state = self.initialize_state() if state is None else state

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

    def _potential(self, graph: Batch, initial_graph: Batch):
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
        energy, forces, stress = self.model(graph, initial_graph)
        energy = energy.detach().float().cpu()
        forces = forces.detach().float().cpu()

        if self.config.compute_stress:
            assert stress is not None, "Stress key must be set in the relaxer config."
            stress = stress.detach().float().cpu()
            return energy, forces, stress
        else:
            return energy, forces

    def _relax(self, graph: Batch):
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
            potential=partial(self._potential, initial_graph=graph),
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

    def step(self, input: Batch, y_true: torch.Tensor):
        """
        Perform a relaxation step on the input structure.

        Args:
            input (Batch): The input batch of structures.
            y_true (torch.Tensor): The true target values.
        """
        state = self.state

        # Relax the structure
        relax_out = self._relax(input)

        # Add to state
        state.y_pred.append(float(self.relaxation_output_to_energy(relax_out).item()))
        state.y_true.append(float(y_true.float().item()))

        self.state = state

    def compute_metrics(self, module: LightningModule):
        """
        Calculates the metrics at the end of a relaxation epoch.

        Returns:
            metrics (dict): A dictionary containing the calculated metrics.
        """

        y_pred = torch.tensor(
            self.state.y_pred, dtype=torch.float, device=self.device
        ).view(-1)
        y_true = torch.tensor(
            self.state.y_true, dtype=torch.float, device=self.device
        ).view(-1)
        assert (
            len(y_pred) == len(y_true)
        ), f"Shapes of y_pred and y_true must match, got {len(y_pred)=} and {len(y_true)=}."

        # Gather all the metrics from all nodes
        y_pred = cast(torch.Tensor | list[torch.Tensor], module.all_gather(y_pred))
        y_true = cast(torch.Tensor | list[torch.Tensor], module.all_gather(y_true))

        # Flatten the tensors
        if isinstance(y_pred, list):
            y_pred = torch.cat(y_pred, dim=0)
        if isinstance(y_true, list):
            y_true = torch.cat(y_true, dim=0)

        # Convert to lists of floats
        y_pred = cast(list[float], y_pred.tolist())
        y_true = cast(list[float], y_true.tolist())
        assert (
            len(y_pred) == len(y_true)
        ), f"Shapes of y_pred and y_true must match, got {len(y_pred)=} and {len(y_true)=}."

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute the metrics
        metrics = stable_metrics(
            y_true,
            y_pred,
            stability_threshold=self.config.stability_threshold,
        )
        return metrics
