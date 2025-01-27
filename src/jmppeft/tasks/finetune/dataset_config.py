from pathlib import Path
from typing import Literal, TypeAlias

from .base import (
    FinetuneLmdbDatasetConfig,
    FinetuneMatbenchDiscoveryDatasetConfig,
    FinetuneMatbenchDiscoveryMegNet133kDatasetConfig,
    FinetunePDBBindDatasetConfig,
)
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


def matbench_discovery_config(
    base_path: Path,
    split: Split,
    use_megnet_133k: bool = True,
    use_atoms_metadata: bool = True,
    use_linref: bool = False,
):
    if use_megnet_133k:
        assert use_atoms_metadata, "use_atoms_metadata must be True for MegNet-133k"

        base_path = base_path
        config = FinetuneMatbenchDiscoveryMegNet133kDatasetConfig(
            base_path=base_path / f"{split}",
            energy_linref_path=base_path / "linrefs.npy" if use_linref else None,
        )
    else:
        config = FinetuneMatbenchDiscoveryDatasetConfig(
            split_csv_path=base_path / "splits" / f"{split}.csv",
            base_path=base_path / "mptrj-gga-ggapu",
            atoms_metadata=base_path / "natoms" / f"{split}.npy"
            if use_atoms_metadata
            else None,
            energy_linref_path=base_path / "energy_linref.npy" if use_linref else None,
        )

    return config
