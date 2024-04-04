import pickle
from collections.abc import Sequence
from pathlib import Path

import ase
import ase.io
import numpy as np
import pandas as pd
from tqdm import tqdm

base_path = Path("/mnt/datasets/matbench-discovery-traj")

natoms_path = base_path / "natoms"
natoms_path.mkdir(exist_ok=True)
pkl_file = natoms_path / "id_to_natoms.pkl"

if not pkl_file.exists():
    extxyz_files = list((base_path / "mptrj-gga-ggapu").glob("*.extxyz"))
    print(len(extxyz_files))

    natoms: dict[str, int] = {}
    for f in tqdm(extxyz_files):
        # Read the first frame
        atoms = ase.io.read(f, 0)
        assert not isinstance(atoms, Sequence)

        # Get the number of atoms in the molecule
        natoms[f.stem] = len(atoms)

    # Save the dict
    with open(pkl_file, "wb") as f:
        pickle.dump(natoms, f)
else:
    print("Pickle file exists, loading it")
    with open(pkl_file, "rb") as f:
        natoms = pickle.load(f)

# For each split, let's make its own dense natoms array
splits = ["train", "val", "test"]
for split in tqdm(splits):
    df = pd.read_csv(base_path / "splits" / f"{split}.csv")
    split_natoms = np.array([natoms[id_] for id_ in df["id"]])

    np.save(natoms_path / f"{split}.npy", split_natoms)