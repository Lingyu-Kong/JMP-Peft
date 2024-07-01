from collections.abc import Sequence
from typing import Any, Literal

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from typing_extensions import override

from ..modules.dataset.common import CommonDatasetConfig


def _mptrj_transform_single(data: dict[str, Any]):
    numbers = torch.tensor(data["numbers"], dtype=torch.long)
    return Data.from_dict(
        {
            "atomic_numbers": numbers,
            "pos": torch.tensor(data["positions"], dtype=torch.float),
            "tags": torch.zeros_like(numbers),
            "fixed": torch.zeros_like(numbers, dtype=torch.bool),
            "force": torch.tensor(data["forces"], dtype=torch.float),
            "cell": torch.tensor(data["cell"], dtype=torch.float).unsqueeze(dim=0),
            "stress": torch.tensor(data["stress"], dtype=torch.float).unsqueeze(dim=0),
            "y": data["ef_per_atom"],
            "natoms": data["num_atoms"],
        }
    )


def mptrj_transform_fn(data: dict[str, Any]):
    if not isinstance(data["num_atoms"], Sequence):
        return _mptrj_transform_single(data)

    batch = Batch.from_data_list(
        [
            _mptrj_transform_single({k: v[i] for k, v in data.items()})
            for i in range(len(data["num_atoms"]))
        ]
    )
    return batch


# def mptrj_transform_fn(data: dict[str, Any]):
#     return Data.from_dict(
#         {
#             # "idx": data["idx"],
#             "atomic_numbers": data["numbers"],
#             "pos": data["positions"],
#             "tags": torch.zeros_like(data["numbers"]),
#             "fixed": torch.zeros_like(data["numbers"], dtype=torch.bool),
#             "force": data["forces"],
#             "cell": data["cell"].unsqueeze(dim=0),
#             "stress": data["stress"].unsqueeze(dim=0),
#             "y": data["ef_per_atom"],
#             "natoms": data["num_atoms"],
#         }
#     )


class FinetuneMPTrjHuggingfaceDatasetConfig(CommonDatasetConfig):
    name: Literal["mp_trj_huggingface"] = "mp_trj_huggingface"

    split: Literal["train", "val", "test"]

    def create_dataset(self):
        # dataset = datasets.load_dataset("nimashoghi/mptrj", split=self.split)
        # assert isinstance(dataset, datasets.Dataset)

        # # dataset.set_transform(mptrj_transform_fn)
        # # Can't use format and transform together
        # # dataset.set_format("torch")

        # atoms_metadata = dataset.with_format("numpy")["num_atoms"]
        # dataset.atoms_metadata = atoms_metadata

        # def data_sizes(indices: list[int]) -> np.ndarray:
        #     return dataset["num_atoms"][indices].numpy()

        # dataset.data_sizes = data_sizes

        # dataset = DT.transform(dataset, mptrj_transform_fn, copy_data=False)
        # return dataset

        return FinetuneMPTrjHuggingfaceDataset(self)


class FinetuneMPTrjHuggingfaceDataset(Dataset[Data]):
    def __init__(self, config: FinetuneMPTrjHuggingfaceDatasetConfig):
        super().__init__()

        dataset = datasets.load_dataset("nimashoghi/mptrj", split=config.split)
        assert isinstance(dataset, datasets.Dataset)
        self.dataset = dataset
        self.dataset.set_format("torch")

        self.atoms_metadata: np.ndarray = self.dataset["num_atoms"].numpy()

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __len__(self):
        return len(self.dataset)

    @override
    def __getitem__(self, idx: int) -> Data:
        data = self.dataset[idx]
        return Data.from_dict(
            {
                # "idx": data["idx"],
                "atomic_numbers": data["numbers"],
                "pos": data["positions"],
                "tags": torch.zeros_like(data["numbers"]),
                "fixed": torch.zeros_like(data["numbers"], dtype=torch.bool),
                "force": data["forces"],
                "cell": data["cell"].unsqueeze(dim=0),
                "stress": data["stress"].unsqueeze(dim=0),
                "y": data["ef_per_atom"],
                "natoms": data["num_atoms"],
            }
        )
