from pathlib import Path

import ase
import ase.io
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override


def _get_fixed_atoms(atoms: ase.Atoms):
    fixed = np.zeros(len(atoms), dtype=np.bool_)
    if (constraints := getattr(atoms, "constraints", None)) is None:
        return fixed

    from ase.constraints import FixAtoms

    if (
        constraint := next(
            (c for c in constraints if isinstance(c, FixAtoms)),
            None,
        )
    ) is None:
        return fixed

    fixed[constraint.index] = True
    return fixed


def frame_to_data(
    frame: ase.Atoms,
    fractional_coordinates: bool = False,
) -> Data:
    frame_level_properties: dict[str, np.ndarray] = {}
    # pos: (n, 3)
    # pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
    if fractional_coordinates:
        pos = frame.get_scaled_positions()
    else:
        pos = frame.get_positions()
    pos = np.array(pos, dtype=np.float32)
    frame_level_properties["pos"] = pos

    # energy: ()
    energy = (
        np.array(
            frame.get_potential_energy(apply_constraint=False),
            dtype=np.float32,
        )
    )[np.newaxis]
    frame_level_properties["energy"] = energy

    # forces: (n, 3)
    forces = np.array(frame.get_forces(apply_constraint=False), dtype=np.float32)
    frame_level_properties["force"] = forces

    # the following properties do not change across frames:
    # - atomic_numbers: (n,)
    # - cell: (3, 3)
    # - fixed: (n,)
    # - tags: (n,)
    atomic_numbers = np.array(frame.get_atomic_numbers(), dtype=np.int32)
    cell = np.array(np.array(frame.cell), dtype=np.float32)
    fixed = np.array(_get_fixed_atoms(frame), dtype=np.bool_)
    tags = np.array(frame.get_tags(), dtype=np.int32)

    data_dict = {
        **frame_level_properties,
        "atomic_numbers": atomic_numbers,
        "cell": cell,
        "fixed": fixed,
        "tags": tags,
    }

    data = Data.from_dict(data_dict)
    return data


class MatbenchDiscoveryAseDataset(Dataset[Data]):
    def __init__(
        self,
        split_csv_path: Path,
        base_path: Path,
        energy_linref_path: Path | None = None,
        fractional_coordinates: bool = False,
    ) -> None:
        super().__init__()

        self.df = pd.read_csv(split_csv_path, index_col=False)
        self.base_path = base_path

        self.energy_linref = None
        if energy_linref_path is not None:
            self.energy_linref = np.load(energy_linref_path)

        self.fractional_coordinates = fractional_coordinates

    def __len__(self) -> int:
        return len(self.df)

    @override
    def __getitem__(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        id_ = row["id"]
        traj_idx = row["traj_idx"]

        traj_path = self.base_path / f"{id_}.extxyz"
        frame = ase.io.read(traj_path, index=traj_idx)
        assert isinstance(frame, ase.Atoms), f"Expected ase.Atoms, got {type(frame)}"

        data = frame_to_data(frame, self.fractional_coordinates)

        if self.energy_linref is not None:
            reference_energy = self.energy_linref[data.atomic_numbers].sum()
            data.energy -= reference_energy

        data.y = data.pop("energy")
        return data
