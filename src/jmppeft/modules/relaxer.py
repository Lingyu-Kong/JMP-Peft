"""
Dynamics calculations -- heavily based on M3GNet's implementation from:
https://github.com/materialsvirtuallab/m3gnet/blob/main/m3gnet/models/_dynamics.py
"""

import contextlib
import io
import pickle
import sys
from typing import Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
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
    def get_efs_tensor(
        self,
        graph: Graph,
        include_stresses: bool = True,
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


class JMPCalculator(Calculator):
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
        compute_stress: bool = True,
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
        results = self.potential.get_efs_tensor(
            graph, include_stresses=self.compute_stress
        )
        self.results.update(
            energy=results[0].numpy().ravel()[0],
            free_energy=results[0].numpy().ravel()[0],
            forces=results[1].numpy(),
        )
        if self.compute_stress:
            assert len(results) == 3, "Stress must be calculated"
            self.results.update(stress=results[2].numpy()[0] * self.stress_weight)


class Relaxer:
    """
    Relaxer is a class for structural relaxation
    """

    def __init__(
        self,
        potential: Potential,
        graph_converter: GraphConverter,
        optimizer_cls: type[Optimizer],
        relax_cell: bool = True,
        stress_weight: float = 0.01,
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
        """

        self.optimizer_cls = optimizer_cls
        self.calculator = JMPCalculator(
            potential=potential,
            graph_converter=graph_converter,
            stress_weight=stress_weight,
        )
        self.relax_cell = relax_cell
        self.potential = potential
        self.ase_adaptor = AseAtomsAdaptor()

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
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
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

        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),
            "trajectory": obs,
        }


class TrajectoryObserver:
    """
    Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures
    """

    def __init__(self, atoms: Atoms):
        """
        Args:
            atoms (Atoms): the structure to observe
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """
        The logic for saving the properties of an Atoms during the relaxation
        Returns:
        """
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def compute_energy(self) -> float:
        """
        calculate the energy, here we just use the potential energy
        Returns:
        """
        energy = self.atoms.get_potential_energy()
        return energy

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
