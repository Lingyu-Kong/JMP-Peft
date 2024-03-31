import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from ll.typecheck import Float, Int
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override


class _JsonData(TypedDict):
    structure_id: str
    positions: list[list[float]]  # (n_atoms, 3)
    cell: list[list[float]]  # (3, 3)
    atomic_numbers: list[int]  # (n_atoms,)
    energy: float
    stress: list[list[list[float]]]  # (1, 3, 3)
    forces: list[list[float]]  # (n_atoms, 3)


@dataclass
class _ParsedData:
    structure_id: str
    positions: Float[np.ndarray, "n_atoms 3"]
    cell: Float[np.ndarray, "3 3"]
    atomic_numbers: Int[np.ndarray, "n_atoms"]
    energy: float
    stress: Float[np.ndarray, "1 3 3"]
    forces: Float[np.ndarray, "n_atoms 3"]


class MatbenchDiscoveryMegNetJsonDataset(Dataset[Data]):
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __init__(
        self,
        json_path: Path,
        energy_linref_path: Path | None = None,
    ) -> None:
        super().__init__()

        original_data_list: list[_JsonData] = json.load(json_path.open("r"))
        # Parse it into a more convenient format
        parsed_data_list: list[_ParsedData] = []
        natoms_list: list[int] = []
        for data in original_data_list:
            parsed_data = _ParsedData(
                structure_id=data["structure_id"],
                positions=np.array(data["positions"]),
                cell=np.array(data["cell"]),
                atomic_numbers=np.array(data["atomic_numbers"]),
                energy=data["energy"],
                stress=np.array(data["stress"]),
                forces=np.array(data["forces"]),
            )
            parsed_data_list.append(parsed_data)
            natoms_list.append(parsed_data.positions.shape[0])

        self.data = parsed_data_list
        self.atoms_metadata = np.array(natoms_list)

        self.energy_linref = None
        if energy_linref_path is not None:
            self.energy_linref = np.load(energy_linref_path)

    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, idx: int) -> Data:
        parsed_data = self.data[idx]

        data_dict: dict[str, np.ndarray] = {
            "atomic_numbers": parsed_data.atomic_numbers,
            "cell": parsed_data.cell,
            "pos": parsed_data.positions,
            "energy": np.array(parsed_data.energy),
            "force": parsed_data.forces,
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
