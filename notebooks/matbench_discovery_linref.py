from pathlib import Path

import numpy as np
from jmppeft.datasets.matbench_discovery_ase import MatbenchDiscoveryAseDataset
from sklearn.linear_model import LinearRegression as LR
from tqdm import tqdm

dataset = MatbenchDiscoveryAseDataset(
    Path("/mnt/datasets/matbench-discovery-traj/splits/train.csv"),
    Path("/mnt/datasets/matbench-discovery-traj/mptrj-gga-ggapu"),
)

coeffs: dict[str, np.ndarray] = {}
energies = []
X = np.zeros((len(dataset), 120))
for i, data in enumerate(tqdm(dataset, total=len(dataset))):
    atom_indices, atom_counts = np.unique(data.atomic_numbers, return_counts=True)
    X[i, atom_indices] = atom_counts
    energies.append(data.y)

energies = np.array(energies).reshape(-1, 1)
reg = LR(fit_intercept=False).fit(X, energies)

np.save(Path("/mnt/datasets/matbench-discovery-traj/linref.npy"), reg.coef_[0])
