# %%
from pathlib import Path

from jmppeft.tasks.pretrain import module as M

base_dir: Path = Path("/global/cfs/cdirs/m3641/Nima/datasets/")
metadatas_dir: Path = Path("/global/cfs/cdirs/m3641/Nima/metadatas/")

config = M.PretrainDatasetConfig(
    src=base_dir / "oc20/s2ef/2M/train/",
    metadata_path=metadatas_dir / "oc20-2M-train.npz",
    # lin_ref=base_dir / "oc20/coeff_2M.npz",
)
dataset = M.PretrainLmdbDataset(config)
print(len(dataset))
# %%
# import numpy as np
# from tqdm.auto import tqdm

# MAX_ATOMIC_NUMBER = 120

# energies_list: list[float] = []
# atoms_list: list[np.ndarray] = []

# for data in tqdm(dataset, total=len(dataset)):
#     energies_list.append(float(data.energy))

#     atoms = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)
#     # Get the unique atom types and their counts
#     atom_types, counts = np.unique(
#         data.atomic_numbers.long().numpy(), return_counts=True
#     )
#     # Set the atom counts
#     atoms[atom_types] = counts
#     atoms_list.append(atoms)

# energies = np.array(energies_list)
# atoms = np.stack(atoms_list)

# print(energies.shape, atoms.shape)

# %%
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

MAX_ATOMIC_NUMBER = 120


def collate_fn(data_list):
    energies_list = []
    atoms_list = []
    for data in data_list:
        energies_list.append(float(data.energy))

        atoms = np.zeros((MAX_ATOMIC_NUMBER,), dtype=np.float32)
        atom_types, counts = np.unique(
            data.atomic_numbers.long().numpy(), return_counts=True
        )
        atoms[atom_types] = counts
        atoms_list.append(atoms)

    energies = np.array(energies_list)
    atoms = np.stack(atoms_list)
    return energies, atoms


dataloader = DataLoader(
    dataset,
    batch_size=64,
    collate_fn=collate_fn,
    num_workers=16,
)

energies_list_flat = []
atoms_list_flat = []

for energies, atoms in tqdm(dataloader):
    energies_list_flat.append(energies)
    atoms_list_flat.append(atoms)

energies_flat = np.concatenate(energies_list_flat)
atoms_flat = np.concatenate(atoms_list_flat)

print(energies_flat.shape, atoms_flat.shape)
np.savez("lin_ref_input.npz", energies=energies_flat, atoms=atoms_flat)
print(energies_flat.shape, atoms_flat.shape)

# %%
from sklearn.linear_model import LinearRegression

reg = LinearRegression(fit_intercept=False).fit(atoms_flat, energies_flat)
print(reg.coef_)

# %%
np.savez("lin_ref_coeffs.npz", coeff=reg.coef_)
