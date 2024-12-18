from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols
from ase.io import write
from ase.stress import full_3x3_to_voigt_6_stress
from tqdm import tqdm

from jmppeft.tasks.finetune import base


class FakeCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, energy: float, forces: np.ndarray, stress: np.ndarray):
        super().__init__()
        self.energy = energy
        self.forces = forces
        self.stress = stress

    def calculate(self, atoms=None, properties=None, system_changes=None):
        all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]

        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )
        self.results.update(
            {
                "energy": self.energy,
                "free_energy": self.energy,
                "forces": self.forces,
                "stress": full_3x3_to_voigt_6_stress(self.stress),
            }
        )


split: Literal["train", "val", "test"] = "train"
dataset_config = base.FinetuneMPTrjHuggingfaceDatasetConfig(
    split=split,
    energy_column_mapping={
        "y": "corrected_total_energy",
        "y_relaxed": "corrected_total_energy_relaxed",
    },
)
dataset = dataset_config.create_dataset()

atoms_list = []
species = ["Zn", "Mn", "O"]
print("Extracting atoms with species:", species)
pbar = tqdm(total=len(dataset))
for i in range(len(dataset)):
    data = dataset[i]
    energy = data["y"].item()
    forces = data["force"].numpy()  ## [Nx3]
    stress = data["stress"][0].numpy()  ## [3x3]
    atomic_numbers = data["atomic_numbers"].numpy()
    chemical_symbols_ = [chemical_symbols[number] for number in atomic_numbers]
    if set(chemical_symbols_) <= set(species):
        atoms = Atoms(
            numbers=atomic_numbers,
            positions=data["pos"],
            cell=data["cell"][0],
            pbc=True,
        )
        calc = FakeCalculator(energy, forces, stress)
        atoms.set_calculator(calc)
        energy_ = atoms.get_potential_energy()
        forces_ = atoms.get_forces()
        stress_ = atoms.get_stress(voigt=False)
        # print(energy_, energy)
        # print(forces_, forces)
        # print(stress_, stress)
        # break
        atoms_list.append(atoms)
    pbar.update(1)
pbar.close()

print("Number of atoms with species:", species, len(atoms_list))

write("{}-mptrj.xyz".format("".join(species)), atoms_list)
