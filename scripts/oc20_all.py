# %%
from pathlib import Path

import ase
import ase.io
import datasets
import numpy as np
import pandas as pd


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


def generator_all(shards: list[Path]):
    for file_path in shards:
        df_metadata, atoms_list = load_file(file_path)
        yield from generator(df_metadata, atoms_list)


def shards(path: str | Path):
    if isinstance(path, str):
        path = Path(path)
    return sorted(list(path.glob("*.extxyz")), key=lambda x: int(x.stem))


def create_dataset(path: str | Path):
    dataset = datasets.Dataset.from_generator(
        generator_all,
        gen_kwargs={"shards": shards(path)},
        num_proc=32,
    )
    dataset = dataset.cast_column("cell", datasets.Array2D((3, 3), dtype="float32"))
    return dataset


# %%
dataset = datasets.DatasetDict(
    {
        "2M": create_dataset("/mnt/datasets/s2ef_train_2M/s2ef_train_2M_extracted/"),
        "val": create_dataset("/mnt/datasets/s2ef_val_id/s2ef_val_id_extracted/"),
    }
)

dataset.push_to_hub("nimashoghi/oc20-s2ef", private=True, set_default=False)
