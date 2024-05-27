from functools import cache
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from matbench_discovery import Key
from matbench_discovery.data import DATA_FILES
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override

log = getLogger(__name__)


class MatBenchDiscoveryIS2REDataset(Dataset[Data]):
    def __init__(self, energy_linref_path: Path | None = None):
        super().__init__()

        self.df = pd.read_json(DATA_FILES.wbm_cses_plus_init_structs).set_index(
            Key.mat_id
        )

        self.energy_linref = None
        if energy_linref_path is not None:
            self.energy_linref = np.load(energy_linref_path)

    def __len__(self):
        return len(self.df)

    @override
    def __getitem__(self, idx: Any):
        row = self.df.iloc[idx]
        structure = Structure.from_dict(row[Key.init_struct])
        entry = ComputedStructureEntry.from_dict(row[Key.cse])

        data = Data(
            pos=torch.tensor(structure.cart_coords, dtype=torch.float),
            atomic_numbers=torch.tensor(structure.atomic_numbers, dtype=torch.long),
            cell=torch.tensor(structure.lattice.matrix, dtype=torch.float).view(
                1, 3, 3
            ),
            y_relaxed=torch.tensor(entry.energy, dtype=torch.float),
            natoms=torch.tensor(len(structure), dtype=torch.long),
        )

        if self.energy_linref is not None:
            reference_energy = self.energy_linref[data.atomic_numbers].sum()
            data.y_relaxed -= reference_energy

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
