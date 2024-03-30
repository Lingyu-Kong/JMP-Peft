# %%
from pathlib import Path

base_path = Path("/mnt/datasets/matbench-discovery-traj/mptrj-gga-ggapu")
extxyz_files = list(base_path.glob("*.extxyz"))
print(len(extxyz_files))

# %%
from collections.abc import Sequence

import ase.io
import numpy as np
from tqdm.auto import tqdm

energies: list[float] = []
forces: list[np.ndarray] = []

for f in tqdm(extxyz_files):
    traj = ase.io.read(str(f), ":")
    if not isinstance(traj, Sequence):
        traj = [traj]

    for frame in traj:
        energies.append(frame.get_potential_energy(apply_constraint=False))
        forces.append(frame.get_forces(apply_constraint=False))

# %%
energy_array = np.array(energies)
force_array = np.concatenate(forces)

print(energy_array.mean(), energy_array.std())
print(force_array.mean(), force_array.std())
# Also get the mean square of the forces
print((force_array**2).mean())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(energy_array, kde=True)
plt.show()

sns.histplot(force_array.flatten(), kde=True)
plt.show()

# %%
from collections.abc import Sequence

import ase.io
from tqdm.auto import tqdm

traj_lens: dict[int, int] = {}
for i, f in enumerate(tqdm(extxyz_files)):
    traj = ase.io.read(str(f), ":")
    traj_len = len(traj) if isinstance(traj, Sequence) else 1
    traj_lens[i] = traj_len

# %%
import pickle

with (base_path.parent / "traj_lens.pkl").open("wb") as f:
    pickle.dump(traj_lens, f)

# %%
id_to_traj_lens: dict[str, int] = {}
for i, f in enumerate(tqdm(extxyz_files)):
    id_ = f.stem
    id_to_traj_lens[id_] = traj_lens[i]

with (base_path.parent / "id_to_traj_lens.pkl").open("wb") as f:
    pickle.dump(id_to_traj_lens, f)

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Plot histogram of trajectory lengths
sns.histplot(np.array(list(traj_lens.values())), kde=True)

a = np.array(list(traj_lens.values()))
print(a.mean(), a.std())

# %%
import ase.io

traj = ase.io.read(str(extxyz_files[101]), ":")
print(len(traj))

# %%
from collections import defaultdict

import numpy as np
from einops import rearrange
from torch_geometric.data import Data

# def frame_to_data(
#     frame: ase.Atoms,
#     reference_energy: float | None = None,
# ) -> Data:
#     # Frame-level data
#     pos = frame.get_positions()
#     energy = frame.get_potential_energy(apply_constraint=False)
#     if reference_energy is not None:
#         energy -= reference_energy
#     forces = frame.get_forces(apply_constraint=False)

#     # Atom-level data
#     atomic_numbers = frame.get_atomic_numbers()
#     cell = frame.get_cell()
#     pbc = frame.get_pbc()
#     tags = frame.get_tags()


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
    reference_energy: float = 0.0,
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
        - reference_energy
    )[np.newaxis]
    frame_level_properties["energy"] = energy

    # forces: (n, 3)
    forces = np.array(frame.get_forces(apply_constraint=False), dtype=np.float32)
    frame_level_properties["forces"] = forces

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


# %%
from pathlib import Path

from jmppeft.datasets.matbench_discovery_ase import MatbenchDiscoveryAseDataset

dataset = MatbenchDiscoveryAseDataset(
    Path("/mnt/datasets/matbench-discovery-traj/splits/train.csv"),
    Path("/mnt/datasets/matbench-discovery-traj/mptrj-gga-ggapu"),
)
print(len(dataset))
