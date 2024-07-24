# %%
from pathlib import Path

import ase
import ase.io
import datasets
import pandas as pd

base_path = Path("/mnt/datasets/s2ef_train_2M/s2ef_train_2M_extracted/")
extxyz_files_compressed = list(base_path.glob("*.extxyz"))
# Order by the name
extxyz_files_compressed = sorted(extxyz_files_compressed, key=lambda p: int(p.stem))
print(len(extxyz_files_compressed))


# %%
def load_file(file_path: Path):
    # There should be a .txt file with the same name
    txt_file = file_path.with_suffix(".txt")
    with txt_file.open("r") as f:
        df_metadata = pd.read_csv(
            f,
            # No header, but cols are: "sid", "fid", "reference_energy"
            header=None,
            names=["sid", "fid", "reference_energy"],
        )

    with file_path.open("r") as f:
        atoms = ase.io.read(f, format="extxyz", index=":")
        assert isinstance(atoms, list), f"{type(atoms)=} is not list"

    return df_metadata, atoms


df_metadata, atoms_list = load_file(extxyz_files_compressed[0])
print(df_metadata.head())
print(len(atoms_list))


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


def generator(df_metadata: pd.DataFrame, atoms_list: list[ase.Atoms]):
    for i in range(len(df_metadata)):
        row = df_metadata.iloc[i]
        atoms = atoms_list[i]

        yield {
            "sid": str(row.sid),
            "fid": str(row.fid),
            "reference_energy": float(row.reference_energy),
            "num_atoms": len(atoms),
            "atomic_numbers": np.array(atoms.get_atomic_numbers(), dtype=np.int64),
            "pos": np.array(atoms.get_positions(), dtype=np.float32),
            "energy": float(atoms.get_potential_energy(apply_constraint=False)),
            "forces": np.array(
                atoms.get_forces(apply_constraint=False), dtype=np.float32
            ),
            "cell": np.array(np.array(atoms.cell), dtype=np.float32),
            "fixed": _get_fixed_atoms(atoms),
            "tags": np.array(atoms.get_tags(), dtype=np.int32),
        }


print(next(generator(df_metadata, atoms_list)))


# %%
def generator_all(shards: list[Path]):
    for file_path in shards:
        df_metadata, atoms_list = load_file(file_path)
        yield from generator(df_metadata, atoms_list)


dataset = datasets.Dataset.from_generator(
    generator_all,
    gen_kwargs={"shards": extxyz_files_compressed},
    num_proc=32,
)
dataset = dataset.cast_column("cell", datasets.Array2D((3, 3), dtype="float32"))
dataset

# %%
import nshtrainer.ll as ll

ll.pretty()
dataset.with_format("torch")[0]

# %%

# Push to the hub
dataset.push_to_hub(
    "nimashoghi/oc20_s2ef_train_2M",
    private=True,
)

# %%
datasets.DatasetDict(
    {
        "train": dataset,
    }
)
