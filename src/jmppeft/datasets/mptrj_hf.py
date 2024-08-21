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
