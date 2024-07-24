# %%
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from tqdm.auto import tqdm


class _JsonData(TypedDict):
    structure_id: str
    positions: list[list[float]]  # (n_atoms, 3)
    cell: list[list[float]]  # (3, 3)
    atomic_numbers: list[int]  # (n_atoms,)
    energy: float
    stress: list[list[list[float]]]  # (1, 3, 3)
    forces: list[list[float]]


path = Path("/mnt/datasets/matbench-discovery-traj/megnet-133k/data.ndjson")
# Read the number of lines in the file
with path.open("r") as f:
    n_lines = sum(1 for _ in f)
print(f"Number of lines: {n_lines}")

# %%
# Find the indices for train, val, and test splits
# Do the shuffling here to ensure reproducibility

indices = np.arange(n_lines)
train_ratio, val_ratio, test_ratio = 0.9, 0.05, 0.05
np.random.seed(0)
np.random.shuffle(indices)

train_indices = indices[: int(n_lines * train_ratio)]
val_indices = indices[
    int(n_lines * train_ratio) : int(n_lines * (train_ratio + val_ratio))
]
test_indices = indices[int(n_lines * (train_ratio + val_ratio)) :]
print(
    f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
)

# %%
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, TypeAlias

from nshutils.typecheck import Float, Int

out_base_path = Path("/mnt/datasets/matbench-discovery-traj/megnet-133k-npz")
out_base_path.mkdir(exist_ok=True)

Split: TypeAlias = Literal["train", "val", "test"]
natoms = defaultdict[Split, list[int]](lambda: [])


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

    def save_to_npz(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(
            path,
            structure_id=self.structure_id,
            positions=self.positions,
            cell=self.cell,
            atomic_numbers=self.atomic_numbers,
            energy=self.energy,
            stress=self.stress,
            forces=self.forces,
        )


split_max_indices: dict[Split, int] = {"train": 0, "val": 0, "test": 0}

# Read the file line-by-line:
with path.open("r") as f:
    for i, line in enumerate(tqdm(f, total=n_lines)):
        split: Split
        if i in train_indices:
            split = "train"
        elif i in val_indices:
            split = "val"
        elif i in test_indices:
            split = "test"
        else:
            raise ValueError(f"Unknown index {i}")

        data: _JsonData = json.loads(line)
        parsed_data = _ParsedData.from_json_data(data)
        natoms[split].append(parsed_data.atomic_numbers.shape[0])

        i = split_max_indices[split]
        split_max_indices[split] += 1

        # Save the data
        out_path = out_base_path / f"{split}" / "data" / f"{i}.npz"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        parsed_data.save_to_npz(out_path)

for split in ("train", "val", "test"):
    print(
        f"{split}: {len(natoms[split])} samples, {np.mean(natoms[split]):.2f} atoms/sample"
    )
    # Save natoms to a file
    np.save(out_base_path / f"{split}" / "natoms.npy", np.array(natoms[split]))

# %%
