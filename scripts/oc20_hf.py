# %%
import lzma
from pathlib import Path

import ase
import ase.io
import datasets
import pandas as pd

base_path = Path("/mnt/datasets/s2ef_train_2M/s2ef_train_2M")
extxyz_files_compressed = list(base_path.glob("*.extxyz.xz"))
print(len(extxyz_files_compressed))


# %%
def load_file(file_path: Path):
    # There should be a .txt file with the same name
    txt_file = file_path.parent / file_path.name.replace(".extxyz.xz", ".txt.xz")
    with lzma.open(txt_file, "rt") as f:
        df_metadata = pd.read_csv(
            f,
            # No header, but cols are: "sid", "fid", "reference_energy"
            header=None,
            names=["sid", "fid", "reference_energy"],
        )

    with lzma.open(file_path, "rt") as f:
        atoms = ase.io.read(f, format="extxyz")
        assert isinstance(atoms, list)

    return df_metadata, atoms


df_metadata, atoms_list = load_file(extxyz_files_compressed[0])
print(df_metadata.head())
print(atoms_list)


# %%
import datasets
import numpy as np


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


def generator():
    for i in range(len(df_metadata)):
        row = df_metadata.iloc[i]
        atoms = atoms_list[i]

        yield {
            "sid": str(row.sid),
            "fid": str(row.fid),
            "reference_energy": float(row.reference_energy),
            "atomic_numbers": np.array(atoms.get_atomic_numbers(), dtype=np.int64),
            "pos": np.array(atoms.get_positions(), dtype=np.float32),
            "energy": float(atoms.get_potential_energy(apply_constraint=False)),
            "forces": np.array(
                atoms.get_forces(apply_constraint=False), dtype=np.float32
            ),
            "cell": np.array(atoms.get_cell(), dtype=np.float32),
            "fixed": _get_fixed_atoms(atoms),
            "tags": np.array(atoms.get_tags(), dtype=np.int32),
        }
