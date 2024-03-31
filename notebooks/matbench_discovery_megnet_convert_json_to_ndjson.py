# %%
import json
from pathlib import Path
from typing import TypedDict

from tqdm.auto import tqdm


class _JsonData(TypedDict):
    structure_id: str
    positions: list[list[float]]  # (n_atoms, 3)
    cell: list[list[float]]  # (3, 3)
    atomic_numbers: list[int]  # (n_atoms,)
    energy: float
    stress: list[list[list[float]]]  # (1, 3, 3)
    forces: list[list[float]]


path = Path("/mnt/datasets/matbench-discovery-traj/megnet-133k.json")
data_list: list[_JsonData] = json.load(path.open("r"))
print(len(data_list))

# %%
import numpy as np
from sklearn.linear_model import LinearRegression

out_path = Path("/mnt/datasets/matbench-discovery-traj/megnet-133k.ndjson")
assert not out_path.exists(), f"{out_path} already exists"

# Let's also compute the natoms, energy distribution, and linear reference
natoms_list: list[int] = []
energies_list: list[float] = []
MAX_ATOM_NUMBER = 120
atom_counts = np.zeros((len(data_list), MAX_ATOM_NUMBER), dtype=np.int32)

with out_path.open("w") as f:
    for data in tqdm(data_list):
        atom_indices, atom_counts_ = np.unique(
            data["atomic_numbers"], return_counts=True
        )
        atom_counts[len(natoms_list), atom_indices] = atom_counts_

        natoms_list.append(len(data["atomic_numbers"]))
        energies_list.append(data["energy"])
        f.write(json.dumps(data) + "\n")

natoms = np.array(natoms_list)
energies = np.array(energies_list)

# Linear reference
lr = LinearRegression(fit_intercept=False).fit(atom_counts, energies.reshape(-1, 1))
linrefs = lr.coef_[0]

# %%
# Dump everything
np.save("/mnt/datasets/matbench-discovery-traj/megnet-133k/natoms.npy", natoms)
np.save("/mnt/datasets/matbench-discovery-traj/megnet-133k/energies.npy", energies)
np.save("/mnt/datasets/matbench-discovery-traj/megnet-133k/linrefs.npy", linrefs)

# %%
# Visualize energy distribution
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(energies, bins=100, kde=True)
plt.show()

# %%
# Visualize natoms distribution
sns.histplot(natoms, bins=100, kde=True)
plt.show()
# %%
# Visualize linref energy distribution

linref_energy_list: list[float] = []

for data in tqdm(data_list):
    linref_energy = data["energy"] - linrefs[data["atomic_numbers"]].sum()
    linref_energy_list.append(linref_energy)

linref_energies = np.array(linref_energy_list)

sns.histplot(linref_energies, bins=100, kde=True)
plt.show()
# %%
print(
    linref_energies.min(),
    linref_energies.max(),
    linref_energies.mean(),
    linref_energies.std(),
)
print(energies.min(), energies.max(), energies.mean(), energies.std())


# %%
# Also get the force rms values
force_rms_list = []
for data in tqdm(data_list):
    force_rms = np.sqrt(np.mean(np.sum(np.array(data["forces"]) ** 2, axis=1)))
    force_rms_list.append(force_rms)

force_rms = np.array(force_rms_list)
print(force_rms.min(), force_rms.max(), force_rms.mean(), force_rms.std())

sns.histplot(
    force_rms,
)
plt.show()

# %%

# Store those stats in a json file for later use
stats = {
    "linref_energy": {
        "min": float(linref_energies.min()),
        "max": float(linref_energies.max()),
        "mean": float(linref_energies.mean()),
        "std": float(linref_energies.std()),
    },
    "energy": {
        "min": float(energies.min()),
        "max": float(energies.max()),
        "mean": float(energies.mean()),
        "std": float(energies.std()),
    },
    "force_rms": {
        "min": float(force_rms.min()),
        "max": float(force_rms.max()),
        "mean": float(force_rms.mean()),
        "std": float(force_rms.std()),
    },
}

json.dump(
    stats,
    open("/mnt/datasets/matbench-discovery-traj/megnet-133k/stats.json", "w"),
    indent=4,
)

# %%
