# %%
import nshtrainer.ll as ll

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

sizes = np.load("mptrj_edge_sizes.npy")
sizes

# %%
idxs = [int(idx) for idx in np.argpartition(sizes, 10)[:10]]
sizes[idxs]

# %%
dataset[idxs[9]]
