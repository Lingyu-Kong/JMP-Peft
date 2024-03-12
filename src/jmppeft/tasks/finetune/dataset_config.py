from pathlib import Path
from typing import Literal, TypeAlias

from .base import FinetuneLmdbDatasetConfig, FinetunePDBBindDatasetConfig
from .matbench import MatbenchDataset
from .md22 import MD22Molecule
from .rmd17 import RMD17Molecule
from .spice import SPICEDataset

Split: TypeAlias = Literal["train", "val", "test"]
MatbenchFold: TypeAlias = Literal[0, 1, 2, 3, 4]


def matbench_config(
    dataset: MatbenchDataset,
    base_path: Path,
    split: Split,
    fold: MatbenchFold,
):
    lmdb_path = base_path / "lmdb" / f"matbench_{dataset}" / f"{fold}" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def rmd17_config(
    molecule: RMD17Molecule,
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{molecule}" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def md22_config(
    molecule: MD22Molecule,
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{molecule}" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def qm9_config(
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def qmof_config(
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def spice_config(
    dataset: SPICEDataset,
    base_path: Path,
    split: Split,
):
    lmdb_path = base_path / "lmdb" / f"{dataset}" / f"{split}"
    assert lmdb_path.exists(), f"{lmdb_path} does not exist"

    config = FinetuneLmdbDatasetConfig(src=lmdb_path)
    return config


def pdbbind_config(split: Split):
    config = FinetunePDBBindDatasetConfig(split=split)
    return config
