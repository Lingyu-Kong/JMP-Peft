# %%
import ll

ll.pretty()
# %%
import datasets

datasets.disable_caching()

dataset = datasets.load_dataset("nimashoghi/mptrj", split="train")
assert isinstance(dataset, datasets.Dataset)
dataset.set_format("numpy")
dataset


# %%
import numpy as np
import torch
from jmppeft.utils.goc_graph import get_max_neighbors_mask
from matscipy.neighbours import neighbour_list
from tqdm.auto import tqdm

full_cutoff = 12.0


def map_fn(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray):
    i = neighbour_list(
        "i",
        positions=positions,
        cell=cell,
        pbc=list(pbc),
        cutoff=full_cutoff,
    )

    return int(i.shape[0])


data = dataset[0]
map_fn(data["positions"], data["cell"], data["pbc"])

# %%
# sizes = [map_fn(data["positions"], data["cell"], data["pbc"]) for data in tqdm(dataset)]
# sizes

# %%
import numpy as np
from torch.utils.data import DataLoader

dl = DataLoader(
    dataset,
    batch_size=1024,
    num_workers=16,
    collate_fn=lambda data_list: np.array(
        [map_fn(data["positions"], data["cell"], data["pbc"]) for data in data_list]
    ),
)
dl

# %%
sizes_list = [size for sizes_part in tqdm(dl) for size in sizes_part]

# %%
sizes = np.array(sizes_list)
sizes

np.save("mptrj_edge_sizes.npy", sizes)

# %%
dataset[list(np.argpartition(sizes, -10)[-10:])]
