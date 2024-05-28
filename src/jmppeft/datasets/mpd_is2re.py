from functools import cache
from logging import getLogger
from typing import Any

import numpy as np
import pandas as pd
import torch
from matbench_discovery import Key
from matbench_discovery.data import DATA_FILES
from pymatgen.core import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override

log = getLogger(__name__)


class MatBenchDiscoveryIS2REDataset(Dataset[Data]):
    def __init__(self):
        super().__init__()

        self.df = pd.read_json(DATA_FILES.wbm_initial_structures).set_index(Key.mat_id)
        self.df_summary = pd.read_csv(DATA_FILES.wbm_summary).set_index(Key.mat_id)

    def __len__(self):
        return len(self.df)

    @override
    def __getitem__(self, idx: Any):
        row = self.df.iloc[idx]
        # Get the ID (`Key.mat_id`) and the initial structure (`Key.init_struct`)
        id_ = row.name

        summary_row = self.df_summary.loc[id_]
        structure = Structure.from_dict(row[Key.init_struct])

        data = Data(
            id=id_,
            pos=torch.tensor(structure.cart_coords, dtype=torch.float),
            atomic_numbers=torch.tensor(structure.atomic_numbers, dtype=torch.long),
            cell=torch.tensor(structure.lattice.matrix, dtype=torch.float).view(
                1, 3, 3
            ),
            y_formation=torch.tensor(summary_row[Key.e_form], dtype=torch.float),
            y_above_hull=torch.tensor(summary_row[Key.each_true], dtype=torch.float),
            natoms=torch.tensor(len(structure), dtype=torch.long),
        )

        return data

    @property
    @cache
    def atoms_metadata(self):
        log.critical("Computing atoms metadata...")
        atoms_metadata = (
            self.df[Key.init_struct].apply(lambda x: len(Structure.from_dict(x))).values
        )
        log.critical("Done computing atoms metadata.")
        return atoms_metadata

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]
