import bisect
import pickle
from collections.abc import Mapping
from functools import cache
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import lmdb
import numpy as np
from ll import TypedConfig
from torch.utils.data import Dataset
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ..utils.ocp import pyg2_data_transform

log = getLogger(__name__)


class FinetuneDatasetConfig(TypedConfig):
    src: Path
    """Path to the LMDB file or directory containing LMDB files."""

    metadata_path: Path | None = None
    """Path to the metadata npz file containing the number of atoms in each structure."""

    def __post_init__(self):
        super().__post_init__()

        # If metadata_path is not provided, assume it is src/metadata.npz
        if self.metadata_path is None:
            self.metadata_path = self.src / "metadata.npz"


class FinetuneLmdbDataset(Dataset[BaseData]):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    The keys in the LMDB must be integers (stored as ascii objects) starting
    from 0 through the length of the LMDB. For historical reasons any key named
    "length" is ignored since that was used to infer length of many lmdbs in the same
    folder, but lmdb lengths are now calculated directly from the number of keys.
    Args:
            config (dict): Dataset configuration
    """

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    @property
    def atoms_metadata(self) -> np.ndarray:
        if (
            metadata := next(
                (
                    self.metadata[k]
                    for k in ["natoms", "num_nodes"]
                    if k in self.metadata
                ),
                None,
            )
        ) is None:
            raise ValueError(
                f"Could not find atoms metadata key in loaded metadata.\n"
                f"Available keys: {list(self.metadata.keys())}"
            )
        return metadata

    @property
    @cache
    def metadata(self) -> Mapping[str, np.ndarray]:
        metadata_path = self.metadata_path
        if metadata_path and metadata_path.is_file():
            return np.load(metadata_path, allow_pickle=True)

        raise ValueError(f"Could not find atoms metadata in '{self.metadata_path}'")

    def __init__(self, config: FinetuneDatasetConfig) -> None:
        super().__init__()

        self.config = config
        self.path = Path(self.config.src)
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
        else:
            assert self.path.suffix == ".lmdb", f"File '{self.path}' is not an LMDB"
            db_paths = [self.path]

        self.metadata_path = (
            Path(self.config.metadata_path)
            if self.config.metadata_path
            else self.path / "metadata.npz"
        )

        self.keys: list[list[int]] = []
        self.envs: list[lmdb.Environment] = []
        # Open all the lmdb files
        for db_path in db_paths:
            cur_env = lmdb.open(
                str(db_path.absolute()),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=1,
            )
            self.envs.append(cur_env)

            # If "length" encoded as ascii is present, use that
            length_entry = cur_env.begin().get("length".encode("ascii"))
            if length_entry is not None:
                num_entries = pickle.loads(length_entry)
            else:
                # Get the number of stores data from the number of entries
                # in the LMDB
                num_entries = cur_env.stat()["entries"]

            # Append the keys (0->num_entries) as a list
            self.keys.append(list(range(num_entries)))

        keylens = [len(k) for k in self.keys]
        self.keylen_cumulative: list[int] = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

    def __len__(self) -> int:
        return self.num_samples

    @override
    def __getitem__(self, idx: int):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self.keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self.keylen_cumulative[db_idx - 1]
        assert el_idx >= 0, f"{el_idx=} is not a valid index"

        # Return features.
        key = f"{self.keys[db_idx][el_idx]}".encode("ascii")
        env = self.envs[db_idx]
        data_object_pickled = env.begin().get(key, default=None)
        if data_object_pickled is None:
            raise KeyError(
                f"Key {key=} not found in {env=}. {el_idx=} {db_idx=} {idx=}"
            )

        data_object = pyg2_data_transform(pickle.loads(cast(Any, data_object_pickled)))
        data_object.id = f"{db_idx}_{el_idx}"
        return data_object
