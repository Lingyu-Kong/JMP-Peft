# %%
import lzma
from collections.abc import Callable
from functools import partial
from pathlib import Path

import ase
import ase.io
import datasets
import pandas as pd

base_path = Path("/mnt/datasets/s2ef_train_2M/s2ef_train_2M")
metadata_files_compressed = list(base_path.glob("*.txt"))
print(len(metadata_files_compressed))


# %%
def load_metadata(file_path: Path):
    with lzma.open(file_path, "rt") as f:
        return pd.read_csv(
            f,
            # No header, but cols are: "sid", "fid", "reference_energy"
            header=None,
            names=["sid", "fid", "reference_energy"],
        )


df = load_metadata(metadata_files_compressed[0])
df
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


def generator(df_metadata: pd.DataFrame, get_atoms: Callable[[int], ase.Atoms]):
    for i in range(len(df_metadata)):
        row = df_metadata.iloc[i]
        atoms = get_atoms(i)

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


def get_atoms_fn(
    idx: int,
    *,
    txt_path: Path,
):
    fpath = txt_path.with_name(txt_path.name.replace(".txt.xz", ".extxyz.xz"))
    print(f"Reading idx={idx} from {fpath}")
    with lzma.open(fpath, "rt") as f:
        atoms = ase.io.read(f, format="extxyz", index=idx)
        assert isinstance(atoms, ase.Atoms)
    print(f"Read idx={idx} from {fpath}")
    return atoms


df = df.iloc[:10]
list(generator(df, partial(get_atoms_fn, txt_path=metadata_files_compressed[0])))

# %%
