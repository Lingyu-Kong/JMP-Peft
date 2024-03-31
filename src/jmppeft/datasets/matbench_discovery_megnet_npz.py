from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from ll.typecheck import Float, Int
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override


class _NpzData(TypedDict):
    structure_id: str
    positions: Float[np.ndarray, "n_atoms 3"]
    cell: Float[np.ndarray, "3 3"]
    atomic_numbers: Int[np.ndarray, "n_atoms"]
    energy: Float[np.ndarray, ""]
    stress: Float[np.ndarray, "1 3 3"]
    forces: Float[np.ndarray, "n_atoms 3"]


class MatbenchDiscoveryMegNetNpzDataset(Dataset[Data]):
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __init__(
        self,
        base_path: Path,
        energy_linref_path: Path | None = None,
    ) -> None:
        super().__init__()

        # Load the natoms metadata
        self.atoms_metadata = np.load(base_path / "natoms.npy")
        self.data_base_path = base_path / "data"

        self.energy_linref = None
        if energy_linref_path is not None:
            self.energy_linref = np.load(energy_linref_path)

    def __len__(self) -> int:
        return self.atoms_metadata.shape[0]

    def _get_raw_data(self, idx: int) -> _NpzData:
        p = self.data_base_path / f"{idx}.npz"
        assert p.exists(), f"{p} does not exist"
        return cast(_NpzData, np.load(p))

    @override
    def __getitem__(self, idx: int) -> Data:
        npz_data = self._get_raw_data(idx)

        data_dict: dict[str, np.ndarray] = {
            "atomic_numbers": npz_data["atomic_numbers"],
            "cell": npz_data["cell"],
            "pos": npz_data["positions"],
            "energy": np.array(npz_data["energy"]),
            "force": npz_data["forces"],
        }

        # Convert to PT tensors
        data_dict_pt: dict[str, torch.Tensor] = {}
        for k, v in data_dict.items():
            v_pt = torch.from_numpy(v)
            # Check if v is a floating point numpy array
            if v.dtype.kind == "f":
                v_pt = v_pt.float()
            data_dict_pt[k] = v_pt
        data = Data.from_dict(data_dict_pt)

        if self.energy_linref is not None:
            reference_energy = self.energy_linref[data.atomic_numbers].sum()
            data.energy -= reference_energy

        data.y = data.pop("energy")
        data.cell = data.cell.unsqueeze(dim=0)

        return data
