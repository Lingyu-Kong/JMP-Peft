"""
Dynamics calculations -- heavily based on M3GNet's implementation from:
https://github.com/materialsvirtuallab/m3gnet/blob/main/m3gnet/models/_dynamics.py
"""

import contextlib
import io
import pickle
import sys
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
from ase.calculators.calculator import all_changes
from ase.constraints import ExpCellFilter
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Batch
from typing_extensions import override

Property: TypeAlias = str
Graph: TypeAlias = Batch


@runtime_checkable
class Potential(Protocol):
    def __call__(
        self, graph: Graph
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ): ...


@runtime_checkable
class GraphConverter(Protocol):
    def __call__(self, atoms: Atoms) -> Graph: ...


OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}


class Calculator(ASECalculator):
    implemented_properties: list[Property] = [
        "energy",
        "free_energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        potential: Potential,
        graph_converter: GraphConverter,
        compute_stress: bool = False,
        stress_weight: float = 1.0,
        **kwargs,
    ):
        """

        Args:
            potential (Potential): The potential for calculating the energy,
                force, stress
            graph_converter (GraphConverter): The graph converter for converting
                the atoms to a graph
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs:
        """
        super().__init__(**kwargs)

        self.potential = potential
        self.graph_converter = graph_converter
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[Property] | None = None,
        system_changes: list[Property] | None = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:

        """
        assert atoms is not None, "Atoms must be provided"

        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        graph = self.graph_converter(atoms)
        results = self.potential(graph)
        self.results.update(
            energy=results[0].numpy().ravel()[0],
            free_energy=results[0].numpy().ravel()[0],
            forces=results[1].numpy(),
        )
        if self.compute_stress:
            assert len(results) == 3, "Stress must be calculated"
            self.results.update(stress=results[2].numpy()[0] * self.stress_weight)


class TrajectoryObserver:
    """
    Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures
    """

    def __init__(self, atoms: Atoms, compute_stress: bool = False):
        """
        Args:
            atoms (Atoms): the structure to observe
        """
        self.atoms = atoms
        self.energies: list[float | np.float32 | np.float64] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] | None = [] if compute_stress else None
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """
        The logic for saving the properties of an Atoms during the relaxation
        Returns:
        """
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        if self.stresses is not None:
            self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def compute_energy(self) -> float | np.float32 | np.float64:
        """
        calculate the energy, here we just use the potential energy
        Returns:
        """
        energy = self.atoms.get_potential_energy()
        return energy

    def as_dict(self):
        return {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }

    def save(self, filename: str):
        """
        Save the trajectory to file
        Args:
            filename (str): filename to save the trajectory
        Returns:
        """
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "energy": self.energies,
                    "forces": self.forces,
                    "stresses": self.stresses,
                    "atom_positions": self.atom_positions,
                    "cell": self.cells,
                    "atomic_number": self.atoms.get_atomic_numbers(),
                },
                f,
            )

    @override
    def __repr__(self) -> str:
        return f"TrajectoryObserver(energy=[{len(self.energies)}], forces=[{len(self.forces)}], stresses=[{len(self.stresses)}], atom_positions=[{len(self.atom_positions)}], cell=[{len(self.cells)}])"


@dataclass
class RelaxationTrajectoryFrame:
    energy: torch.Tensor
    forces: torch.Tensor
    stresses: torch.Tensor | None
    pos: torch.Tensor
    cell: torch.Tensor


@dataclass
class RelaxationTrajectory:
    frames: list[RelaxationTrajectoryFrame]

    @classmethod
    def from_observer(cls, observer: TrajectoryObserver):
        frames = [
            RelaxationTrajectoryFrame(
                energy=torch.tensor(float(energy), dtype=torch.float),
                forces=torch.from_numpy(forces).float(),
                stresses=torch.from_numpy(stresses).float(),
                pos=torch.from_numpy(pos).float(),
                cell=torch.from_numpy(cell).float(),
            )
            for energy, forces, stresses, pos, cell in zip(
                observer.energies,
                observer.forces,
                observer.stresses
                if observer.stresses is not None
                else [None] * len(observer.energies),
                observer.atom_positions,
                observer.cells,
            )
        ]
        return cls(frames)


@dataclass
class RelaxationOutput:
    trajectory: RelaxationTrajectory
    atoms: Atoms

    @cached_property
    def structure(self):
        return AseAtomsAdaptor.get_structure(self.atoms)

    def as_dict(self):
        return {
            "structure": self.structure.as_dict(),
            "trajectory": asdict(self.trajectory),
        }


class Relaxer:
    """
    Relaxer is a class for structural relaxation
    """

    def __init__(
        self,
        potential: Potential,
        graph_converter: GraphConverter,
        optimizer_cls: type[Optimizer],
        relax_cell: bool = False,
        stress_weight: float = 0.01,
        compute_stress: bool = False,
    ):
        """

        Args:
            potential (Optional[Union[Potential, str]]): a potential,
                a str path to a saved model or a short name for saved model
                that comes with M3GNet distribution
            optimizer_cls (str or ase Optimizer): the optimization algorithm.
                Defaults to "FIRE"
            relax_cell (bool): whether to relax the lattice cell
            stress_weight (float): the stress weight for relaxation
            compute_stress (bool): whether to compute the stress
        """

        self.optimizer_cls = optimizer_cls
        self.calculator = Calculator(
            potential=potential,
            graph_converter=graph_converter,
            stress_weight=stress_weight,
            compute_stress=compute_stress,
        )
        self.compute_stress = compute_stress
        self.relax_cell = relax_cell
        self.potential = potential

    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str | None = None,
        interval=1,
        verbose=False,
        **kwargs,
    ):
        """

        Args:
            atoms (Atoms): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
                Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            **kwargs:
        Returns:
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms, self.compute_stress)
            if self.relax_cell:
                atoms = cast(Atoms, ExpCellFilter(atoms))
            optimizer = self.optimizer_cls(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, ExpCellFilter):
            atoms = cast(Atoms, atoms.atoms)

        # return {
        #     "final_structure": AseAtomsAdaptor.get_structure(atoms),
        #     "trajectory": obs,
        # }

        # atoms = AseAtomsAdaptor.get_structure(atoms)
        trajectory = RelaxationTrajectory.from_observer(obs)

        return RelaxationOutput(trajectory, atoms)
