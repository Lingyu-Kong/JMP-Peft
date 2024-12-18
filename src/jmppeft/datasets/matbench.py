import logging
from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeAlias

import datasets
import nshconfig as C
import nshutils.typecheck as tc
import numpy as np
import torch
from ase import Atoms
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import final, override
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ..modules.dataset.common import CommonDatasetConfig
from matbench.bench import MatbenchBenchmark


class MatBenchDatasetConfig(CommonDatasetConfig):
    name: Literal["matbench"] = "matbench"
    task: str | None = None
    fold_idx: Literal[0, 1, 2, 3, 4] = 0
    split: Literal["train", "val", "test"] = "train"
    split_ratio: float = 0.8
    
    def create_dataset(self) -> "MatBenchDataset":
        return MatBenchDataset(self)
    
    
class MatBenchDataset(Dataset[Data]):
    
    def __init__(
        self,
        config: MatBenchDatasetConfig,
    ):
        super().__init__()
        self.config = config
        self._initialize_benchmark()
        self._load_data()
        
        print(f"Loaded {len(self)} samples for task {self.config.task} with split {self.config.split} and fold {self.config.fold_idx}")
    
    def _initialize_benchmark(self) -> None:
        """Initialize the Matbench benchmark and task."""

        if self.config.task is None:
            mb = MatbenchBenchmark(autoload=False)
            all_tasks = list(mb.metadata.keys())
            raise ValueError(f"Please specify a task from {all_tasks}")
        else:
            mb = MatbenchBenchmark(autoload=False, subset=[self.config.task])
            self._task = list(mb.tasks)[0]
            self._task.load()
            
    def _load_data(self) -> None:
        """Load and process the dataset split."""
        if self.config.split == "test":
            fold = self._task.folds[self.config.fold_idx]
            inputs_data = self._task.get_test_data(fold)
        else:
            fold = self._task.folds[self.config.fold_idx]
            inputs_data, outputs_data = self._task.get_train_and_val_data(fold)
            if self.config.split == "train":
                inputs_data = inputs_data[: int(self.config.split_ratio * len(inputs_data))]
                outputs_data = outputs_data[: int(self.config.split_ratio * len(outputs_data))]
            elif self.config.split == "val":
                inputs_data = inputs_data[int(self.config.split_ratio * len(inputs_data)) :]
                outputs_data = outputs_data[int(self.config.split_ratio * len(outputs_data)) :]
        structures = [inputs_data[i] for i in range(len(inputs_data)) if type(inputs_data[i]) == Structure]
        labels = [outputs_data[i] for i in range(len(outputs_data))]
        self.atoms_list, self.atoms_metadata = self._convert_structures_to_atoms(structures, labels)
        self.atoms_metadata = np.array(self.atoms_metadata)
        
    def _convert_structures_to_atoms(
        self,
        structures: list[Structure],
        property_values: list[float] | None = None,
    ) -> tuple[list[Atoms], list[int]]:
        """Convert pymatgen structures to ASE atoms.

        Args:
            structures: List of pymatgen Structure objects.
            property_values: Optional list of property values to add to atoms.info.

        """

        adapter = AseAtomsAdaptor()
        atoms_list = []
        atoms_metadata = []
        for i, structure in enumerate(structures):
            atoms = adapter.get_atoms(structure)
            assert isinstance(atoms, Atoms), "Expected an Atoms object"
            if property_values is not None:
                atoms.info[self.config.task] = property_values[i]
            atoms_list.append(atoms)
            atoms_metadata.append(len(atoms))
        return atoms_list, np.array(atoms_metadata)
    
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __len__(self):
        return len(self.atoms_list)
    
    @override
    def __getitem__(self, idx: int) -> Data:
        from ase import Atoms

        atoms: Atoms = self.atoms_list[idx]
        dict_ = {
            "idx": idx,
            "atomic_numbers": torch.tensor(
                np.array(atoms.get_atomic_numbers()), dtype=torch.long
            ),
            "pos": torch.tensor(np.array(atoms.get_positions()), dtype=torch.float),
            "tags": torch.zeros_like(
                torch.tensor(np.array(atoms.get_atomic_numbers()))
            ),
            "fixed": torch.zeros_like(
                torch.tensor(np.array(atoms.get_atomic_numbers())), dtype=torch.bool
            ),
            "cell": torch.tensor(
                np.array(atoms.get_cell()), dtype=torch.float
            ).unsqueeze(dim=0),
            "natoms": torch.tensor(len(atoms), dtype=torch.long),
            self.config.task: torch.tensor(
                atoms.info[self.config.task], dtype=torch.float
            ),
        }

        data = Data.from_dict(dict_)

        return data
