import json
from abc import ABC, abstractmethod
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
    energy: Float[np.ndarray, ""]
    stress: Float[np.ndarray, "1 3 3"]
    forces: Float[np.ndarray, "n_atoms 3"]

    @classmethod
    def from_json_data(cls, json_data: _JsonData) -> "_ParsedData":
        return cls(
            structure_id=json_data["structure_id"],
            positions=np.array(json_data["positions"], dtype=np.float32),
            cell=np.array(json_data["cell"], dtype=np.float32),
            atomic_numbers=np.array(json_data["atomic_numbers"], dtype=np.int32),
            energy=np.array(json_data["energy"], dtype=np.float32),
            stress=np.array(json_data["stress"], dtype=np.float32),
            forces=np.array(json_data["forces"], dtype=np.float32),
        )


class _MatbenchDiscoveryMegNetJsonDatasetBase(Dataset[Data], ABC):
    def __init__(
        self,
        energy_linref_path: Path | None = None,
    ) -> None:
        super().__init__()

        self.energy_linref = None
        if energy_linref_path is not None:
            self.energy_linref = np.load(energy_linref_path)

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _get_raw_data(self, idx: int) -> _ParsedData: ...

    @override
    def __getitem__(self, idx: int) -> Data:
        parsed_data = self._get_raw_data(idx)

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


class MatbenchDiscoveryMegNetJsonDataset(_MatbenchDiscoveryMegNetJsonDatasetBase):
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __init__(
        self,
        json_path: Path,
        energy_linref_path: Path | None = None,
    ) -> None:
        super().__init__(energy_linref_path=energy_linref_path)

        # Make sure `json_path` exists and is a *.json file
        assert json_path.exists(), f"{json_path} does not exist"
        assert json_path.suffix == ".json", f"{json_path} is not a *.json file"

        original_data_list: list[_JsonData] = json.load(json_path.open("r"))
        # Parse it into a more convenient format
        parsed_data_list: list[_ParsedData] = []
        natoms_list: list[int] = []
        for data in original_data_list:
            parsed_data_list.append(parsed_data := _ParsedData.from_json_data(data))
            natoms_list.append(parsed_data.positions.shape[0])

        self.data = parsed_data_list
        self.atoms_metadata = np.array(natoms_list)

    @override
    def __len__(self) -> int:
        return len(self.data)

    @override
    def _get_raw_data(self, idx: int) -> _ParsedData:
        return self.data[idx]


class MatbenchDiscoveryMegNetNdJsonDataset(_MatbenchDiscoveryMegNetJsonDatasetBase):
    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    def __init__(
        self,
        json_path: Path,
        atoms_metadata: Path,
        energy_linref_path: Path | None = None,
    ) -> None:
        super().__init__(energy_linref_path=energy_linref_path)

        # Make sure `json_path` exists and is a *.ndjson file
        assert json_path.exists(), f"{json_path} does not exist"
        assert json_path.suffix == ".ndjson", f"{json_path} is not a *.ndjson file"
        self.json_path = json_path

        # Load atoms metadata
        assert atoms_metadata.exists(), f"{atoms_metadata} does not exist"
        self.atoms_metadata = np.load(atoms_metadata)

    @override
    def __len__(self) -> int:
        return self.atoms_metadata.shape[0]

    @override
    def _get_raw_data(self, idx: int) -> _ParsedData:
        # Read the idx-th line from the *.ndjson file
        with self.json_path.open("r") as f:
            for i, line in enumerate(f):
                if i == idx:
                    return _ParsedData.from_json_data(json.loads(line))
        raise IndexError(f"Index {idx} out of range")
