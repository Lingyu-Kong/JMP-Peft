import logging
from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeAlias

import datasets
import nshconfig as C
import nshutils.typecheck as tc
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import final, override
from ase import Atoms
from ..modules.dataset.common import CommonDatasetConfig

log = logging.getLogger(__name__)


class BaseReferenceConfig(C.Config, ABC):
    @abstractmethod
    def compute_references(
        self,
        compositions: tc.Int[np.ndarray, "dataset_size n_atomic_numbers"],
        energies: tc.Float[np.ndarray, "dataset_size"],
    ) -> tc.Float[np.ndarray, "n_atomic_numbers"]: ...


@final
class LinearReferenceConfig(BaseReferenceConfig):
    name: Literal["linear_reference"] = "linear_reference"

    @override
    def compute_references(self, compositions, energies):
        from sklearn.linear_model import LinearRegression

        c = compositions
        y = energies
        num_chem_species = c.shape[1]

        # tweak to fine tune training from many-element to small element
        zero_indices = np.all(c == 0, axis=0)
        c_reduced = c[:, ~zero_indices]
        full_coeff = np.zeros(num_chem_species)
        coef_reduced = LinearRegression(fit_intercept=False).fit(c_reduced, y).coef_
        full_coeff[~zero_indices] = coef_reduced

        return full_coeff


@final
class RidgeReferenceConfig(BaseReferenceConfig):
    name: Literal["ridge_reference"] = "ridge_reference"

    alpha: float

    @override
    def compute_references(self, compositions, energies):
        from sklearn.linear_model import Ridge

        c = compositions
        y = energies
        num_chem_species = c.shape[1]

        # tweak to fine tune training from many-element to small element
        zero_indices = np.all(c == 0, axis=0)
        c_reduced = c[:, ~zero_indices]
        full_coeff = np.zeros(num_chem_species)
        coef_reduced = (
            Ridge(alpha=self.alpha, fit_intercept=False).fit(c_reduced, y).coef_
        )
        full_coeff[~zero_indices] = coef_reduced

        return full_coeff


ReferenceConfig: TypeAlias = Annotated[
    LinearReferenceConfig | RidgeReferenceConfig,
    C.Field(discriminator="name"),
]


class FinetuneMPTrjHuggingfaceDatasetConfig(CommonDatasetConfig):
    name: Literal["mp_trj_huggingface"] = "mp_trj_huggingface"

    split: Literal["train", "val", "test"]
    references: dict[str, ReferenceConfig] = {}

    energy_column_mapping: dict[str, str]

    filter_small_systems: bool = True

    def create_dataset(self):
        return FinetuneMPTrjHuggingfaceDataset(self)


def _is_small(num_atoms: int, *, threshold: int):
    return num_atoms > threshold


class FinetuneMPTrjHuggingfaceDataset(Dataset[Data]):
    def __init__(self, config: FinetuneMPTrjHuggingfaceDatasetConfig):
        super().__init__()

        self.config = config
        del config

        dataset = datasets.load_dataset("nimashoghi/mptrj", split=self.config.split)
        assert isinstance(dataset, datasets.Dataset)

        if self.config.filter_small_systems:
            dataset = dataset.filter(
                _is_small,
                fn_kwargs={"threshold": 4},
                input_columns=["num_atoms"],
            )

        self.dataset = dataset
        self.dataset.set_format("torch")

        self.atoms_metadata: np.ndarray = self.dataset["num_atoms"].numpy()

        self.atom_references: dict[str, np.ndarray] = {}
        for key, reference_config in self.config.references.items():
            self.atom_references[key] = reference_config.compute_references(
                self.dataset["composition"].numpy(),
                self.dataset[self.config.energy_column_mapping.get(key, key)].numpy(),
            )
            print(dataset["composition"].numpy()[0])
            print(dataset[self.config.energy_column_mapping.get(key, key)].numpy().shape)
            print(key)
            exit()
            log.critical(f"Computed atom references for {key}.")
            log.debug(f"Atom references for {key}: {self.atom_references}")

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __len__(self):
        return len(self.dataset)

    @override
    def __getitem__(self, idx: int) -> Data:
        data_dict = self.dataset[idx]
        dict_ = {
            "idx": idx,
            "atomic_numbers": data_dict["numbers"],
            "pos": data_dict["positions"],
            "tags": torch.zeros_like(data_dict["numbers"]),
            "fixed": torch.zeros_like(data_dict["numbers"], dtype=torch.bool),
            "force": data_dict["forces"],
            "cell": data_dict["cell"].unsqueeze(dim=0),
            "stress": data_dict["stress"].unsqueeze(dim=0),
            # "y": data_dict[self.config.energy_column],
            "natoms": data_dict["num_atoms"],
        }
        # if self.config.relaxed_energy_column is not None:
        #     dict_["y_relaxed"] = data_dict[self.config.relaxed_energy_column]

        for key, mapped_key in self.config.energy_column_mapping.items():
            value = data_dict[mapped_key]
            dict_[key] = value

        # Add atom references
        for key, reference in self.atom_references.items():
            if (unreferenced_value := dict_.get(key)) is None:
                raise ValueError(f"Missing key {key} in data_dict")

            dict_[key] = (
                unreferenced_value - reference[dict_["atomic_numbers"]].sum().item()
            )

        data = Data.from_dict(dict_)

        return data
    
    
class MPTrjDatasetFromXYZConfig(CommonDatasetConfig):
    name: Literal["mp_trj_xyz"] = "mp_trj_xyz"
    file_path: str|list[str]
    split: Literal["train", "val", "test", "all"]
    split_ratio: list[float] = [0.8, 0.1, 0.1]
    references: dict[str, ReferenceConfig] = {}
    filter_small_systems: bool = True
    def create_dataset(self):
        return MPTrjDatasetFromXYZ(self)


class MPTrjDatasetFromXYZ(Dataset[Data]):
    """
    Construct a dataset from a one or list of .xyz files.
    """

    def __init__(self, config: MPTrjDatasetFromXYZConfig):
        super().__init__()
        self.xyz_files = config.file_path
        if isinstance(self.xyz_files, str):
            self.xyz_files = [self.xyz_files]
        self.atoms_list: list[Atoms] = []
        for xyz_file in self.xyz_files:
            from ase.io import read
            atoms = read(xyz_file, index=":")
            self.atoms_list.extend(atoms)
        
        ## Discard structures under 4 atoms
        if config.filter_small_systems:
            self.atoms_list = [atoms for atoms in self.atoms_list if len(atoms) >= 4]
        
        ## Split the dataset
        if config.split == "train":
            self.atoms_list = self.atoms_list[:int(config.split_ratio[0] * len(self.atoms_list))]
        elif config.split == "val":
            self.atoms_list = self.atoms_list[int(config.split_ratio[0] * len(self.atoms_list)):int((config.split_ratio[0]+config.split_ratio[1]) * len(self.atoms_list))]
        elif config.split == "test":
            self.atoms_list = self.atoms_list[int((config.split_ratio[0]+config.split_ratio[1]) * len(self.atoms_list)):]
        elif config.split == "all":
            pass
        else:
            raise ValueError(f"Invalid split: {config.split}")
        
        ## Meta data for num_atoms
        self.atoms_metadata = np.array([len(atoms) for atoms in self.atoms_list])
        
        ## Compute atom references
        def get_chemical_composition(atoms: Atoms) -> np.ndarray:
            chemical_numbers = np.array(atoms.get_atomic_numbers()) - 1
            return np.bincount(chemical_numbers, minlength=120)
        self.atom_references: dict[str, np.ndarray] = {}
        for key, reference_config in config.references.items():
            self.atom_references[key] = reference_config.compute_references(
                np.array([get_chemical_composition(atoms) for atoms in self.atoms_list]),
                np.array([atoms.get_total_energy() for atoms in self.atoms_list]),
            )
            log.critical(f"Computed atom references for {key}.")
            log.debug(f"Atom references for {key}: {self.atom_references}")
    
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __len__(self):
        return len(self.atoms_list)

    @override
    def __getitem__(self, idx: int) -> Data:
        from ase import Atoms
        atoms:Atoms = self.atoms_list[idx]
        dict_ = {
            "idx": idx,
            "atomic_numbers": torch.tensor(np.array(atoms.get_atomic_numbers()), dtype=torch.long),
            "pos": torch.tensor(np.array(atoms.get_positions()), dtype=torch.float),
            "tags": torch.zeros_like(torch.tensor(np.array(atoms.get_atomic_numbers()))),
            "fixed": torch.zeros_like(torch.tensor(np.array(atoms.get_atomic_numbers())), dtype=torch.bool),
            "force": torch.tensor(np.array(atoms.get_forces()), dtype=torch.float),
            "cell": torch.tensor(np.array(atoms.get_cell()), dtype=torch.float).unsqueeze(dim=0),
            "stress": torch.tensor(np.array(atoms.get_stress(voigt=False)), dtype=torch.float).unsqueeze(dim=0),
            "y": torch.tensor(atoms.get_total_energy(), dtype=torch.float),
            "natoms": torch.tensor(len(atoms), dtype=torch.long),
        }
        
        for key, reference in self.atom_references.items():
            if (unreferenced_value := dict_.get(key)) is None:
                raise ValueError(f"Missing key {key} in data_dict")

            dict_[key] = (
                unreferenced_value - reference[dict_["atomic_numbers"]].sum().item()
            )

        data = Data.from_dict(dict_)

        return data