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

        for key, value in self.config.energy_column_mapping.items():
            dict_[key] = data_dict[value]

        data = Data.from_dict(dict_)

        return data
