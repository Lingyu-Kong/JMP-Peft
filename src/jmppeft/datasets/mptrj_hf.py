from typing import Literal

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override

from ..modules.dataset.common import CommonDatasetConfig


class FinetuneMPTrjHuggingfaceDatasetConfig(CommonDatasetConfig):
    name: Literal["mp_trj_huggingface"] = "mp_trj_huggingface"

    split: Literal["train", "val", "test"]
    debug_repeat_largest_systems_for_testing: bool = False
    energy_column: str = "energy_per_atom"
    # See: https://github.com/janosh/matbench-discovery/issues/103#issuecomment-2070941629

    filter_small_systems: bool = True

    def create_dataset(self):
        return FinetuneMPTrjHuggingfaceDataset(self)


class FinetuneMPTrjHuggingfaceDataset(Dataset[Data]):
    def __init__(self, config: FinetuneMPTrjHuggingfaceDatasetConfig):
        super().__init__()

        self.config = config
        del config

        dataset = datasets.load_dataset("nimashoghi/mptrj", split=self.config.split)
        assert isinstance(dataset, datasets.Dataset)

        if self.config.filter_small_systems:
            dataset = dataset.filter(lambda x: x["num_atoms"] > 5)

        self.dataset = dataset
        self.dataset.set_format("torch")

        self.atoms_metadata: np.ndarray = self.dataset["num_atoms"].numpy()

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __len__(self):
        return len(self.dataset)

    @override
    def __getitem__(self, idx: int) -> Data:
        if self.config.debug_repeat_largest_systems_for_testing:
            largest_idx = int(np.argmax(self.atoms_metadata))
            data_dict = self.dataset[largest_idx]
        else:
            data_dict = self.dataset[idx]

        data = Data.from_dict(
            {
                "idx": idx,
                "atomic_numbers": data_dict["numbers"],
                "pos": data_dict["positions"],
                "tags": torch.zeros_like(data_dict["numbers"]),
                "fixed": torch.zeros_like(data_dict["numbers"], dtype=torch.bool),
                "force": data_dict["forces"],
                "cell": data_dict["cell"].unsqueeze(dim=0),
                "stress": data_dict["stress"].unsqueeze(dim=0),
                "y": data_dict[self.config.energy_column],
                "natoms": data_dict["num_atoms"],
            }
        )

        return data
