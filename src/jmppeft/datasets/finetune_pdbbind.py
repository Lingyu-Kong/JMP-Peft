import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cache, lru_cache
from logging import getLogger
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
import torch
from einops import pack
from nshtrainer.ll import TypedConfig
from torch.utils.data import Dataset
from torch_geometric.data.data import BaseData, Data
from typing_extensions import override

from ..modules import transforms as T

try:
    import deepchem.feat as feat
    import deepchem.molnet as molnet
    import deepchem.splits as splits
    from Bio import PDB
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    from deepchem.data import Dataset as DCDataset
    from deepchem.trans import NormalizationTransformer, Transformer
    from rdkit import Chem

    import_error: ImportError | None = None
except ImportError as err:
    if TYPE_CHECKING:
        raise
    import_error = err

if TYPE_CHECKING:
    from deepchem.data import Dataset as DCDataset
    from deepchem.trans import Transformer


log = getLogger(__name__)


class _MolNetDatasetBase(Dataset[BaseData], ABC):
    split_idx = {
        "train": 0,
        "val": 1,
        "test": 2,
    }

    @override
    def __init__(
        self,
        dataset_output: Callable[
            [], tuple[list[str], tuple["DCDataset", ...], list["Transformer"]]
        ],
        task: str,
        split: Literal["train", "val", "test"],
        linref_coeffs: torch.Tensor | None = None,
        transform: Callable[[BaseData], BaseData] | None = None,
    ):
        super().__init__()

        if import_error is not None:
            raise ImportError("Failed to import necessary modules") from import_error

        tasks, datasets, transformers = dataset_output()

        self.task_idx = tasks.index(task)
        split_idx = self.split_idx[split]

        dataset = datasets[split_idx]

        self.y_mean = torch.tensor(0.0)
        self.y_std = torch.tensor(1.0)
        if (
            transformer := next(
                (t for t in transformers if isinstance(t, NormalizationTransformer)),
                None,
            )
        ) is not None and transformer.transform_y:
            # If the transformer is a NormalizationTransformer, we keep
            # the coefficients for later use.
            y_means = transformer.y_means
            y_stds = transformer.y_stds

            # If y_means and y_stds are not arrays, then self.task_idx must be 0
            if isinstance(y_means, float) or (
                isinstance(y_means, np.ndarray) and y_means.size == 1
            ):
                assert self.task_idx == 0, "Only one task is supported"
                y_means = [y_means]

            if isinstance(y_stds, float) or (
                isinstance(y_stds, np.ndarray) and y_stds.size == 1
            ):
                assert self.task_idx == 0, "Only one task is supported"
                y_stds = [y_stds]

            self.y_mean = torch.tensor(y_means[self.task_idx], dtype=torch.float)
            self.y_std = torch.tensor(y_stds[self.task_idx], dtype=torch.float)

        y = dataset.y
        # If y is a 1D array, then self.task_idx must be 0
        if y.ndim == 1:
            assert self.task_idx == 0, "Only one task is supported"
            y = y[:, None]

        self.X = cast(list[Chem.rdchem.Mol], dataset.X)
        self.y = y[:, self.task_idx]
        self.linref_coeffs = linref_coeffs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    @abstractmethod
    def X_to_data(self, idx: int) -> BaseData: ...

    @override
    def __getitem__(self, idx: int) -> BaseData:
        data = self.X_to_data(idx)
        # If the number of atoms is less than 4, the molecule is not valid
        if data.atomic_numbers.shape[0] < 4:
            # HACK: Just return the next molecule
            return self.__getitem__((idx + 1) % len(self))

        if self.linref_coeffs is not None:
            data = T.atomref_transform(data, {"y": self.linref_coeffs})

        if (transform := self.transform) is not None:
            data = transform(data)

        return data


PDBBindTask: TypeAlias = Literal["-logKd/Ki"]


class PDBBindDataset(_MolNetDatasetBase):
    @override
    def __init__(
        self,
        task: PDBBindTask,
        split: Literal["train", "val", "test"],
        linref_coeffs: torch.Tensor | None = None,
        transform: Callable[[BaseData], BaseData] | None = None,
    ):
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)

        super().__init__(
            lambda: molnet.load_pdbbind(
                featurizer=feat.RawFeaturizer(),
                # Random splitting is recommended for this dataset.
                # See https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#pdbbind-datasets
                splitter=splits.RandomSplitter(),
            ),
            task,
            split,
            linref_coeffs,
            transform,
        )

    @override
    def X_to_data(self, idx: int):
        ligand_sdf_file, pocket_pdb_file = cast(tuple[str, str], self.X[idx])
        if (ligand_sdf_info := self._get_sdf_info(ligand_sdf_file)) is None:
            raise RuntimeError(
                f"Failed to extract info from {ligand_sdf_file=} for {idx=}"
            )
        if (pocket_pdb_info := self._get_pdb_info(pocket_pdb_file)) is None:
            raise RuntimeError(
                f"Failed to extract info from {pocket_pdb_file=} for {idx=}"
            )

        ligand_atomic_numbers, ligand_pos = ligand_sdf_info
        ligand_atomic_numbers = torch.from_numpy(
            ligand_atomic_numbers
        ).long()  # n_ligand_atoms
        ligand_pos = torch.from_numpy(ligand_pos).float()  # n_ligand_atoms 3

        pocket_atomic_numbers, pocket_pos = pocket_pdb_info
        pocket_atomic_numbers = torch.from_numpy(
            pocket_atomic_numbers
        ).long()  # n_pocket_atoms
        pocket_pos = torch.from_numpy(pocket_pos).float()  # n_pocket_atoms 3

        atomic_numbers, pos = pack([ligand_atomic_numbers, pocket_atomic_numbers], "*")
        pos, _ = pack([ligand_pos, pocket_pos], "* p")
        tags, _ = pack(
            [
                torch.zeros_like(ligand_atomic_numbers),
                torch.ones_like(pocket_atomic_numbers),
            ],
            "*",
        )

        y = self.y[idx]

        data = Data.from_dict(
            {
                "idx": idx,
                "atomic_numbers": atomic_numbers,
                "pos": pos,
                "tags": tags,
                "y": y,
                # Save the y_mean and y_std in the data object for metrics
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            }
        )
        data = cast(BaseData, data)
        return data

    # Function to extract atomic numbers and coordinates from an SDF file
    @staticmethod
    @lru_cache(maxsize=16)
    def _get_sdf_info(sdf_file: str):
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)
        for mol in suppl:
            if mol is None:
                continue
            atoms = mol.GetAtoms()
            atomic_numbers = np.array([atom.GetAtomicNum() for atom in atoms])
            coords = mol.GetConformer().GetPositions()
            return atomic_numbers, coords
        return None

    @cache
    def _pt():
        return Chem.GetPeriodicTable()

    # Function to extract atomic numbers and coordinates from a PDB file
    @staticmethod
    @lru_cache(maxsize=16)
    def _get_pdb_info(pdb_file: str):
        pt = PDBBindDataset._pt()

        parser = PDB.PDBParser()
        structure = parser.get_structure("structure", pdb_file)
        for model in structure:
            atomic_numbers = []
            coords = []
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        symbol = atom.element.strip()
                        symbol = f"{symbol[0].upper()}{symbol[1:].lower()}"
                        atomic_numbers.append(pt.GetAtomicNumber(symbol))
                        coords.append(atom.get_coord())
            return np.array(atomic_numbers), np.array(coords)
        return None


class PDBBindConfig(TypedConfig):
    task: PDBBindTask = "-logKd/Ki"
    split: Literal["train", "val", "test"]
