# %%
import datasets

dataset = datasets.load_dataset("nimashoghi/mptrj", split="val")
assert isinstance(dataset, datasets.Dataset)
dataset.set_format("numpy")
dataset

# %%
import matscipy
import numpy as np
from matscipy.neighbours import neighbour_list

radius = 8.0
max_neighbors = 15


def map_fn(data: dict[str, np.ndarray]):
    i, j, d, D, S = neighbour_list(
        "ijdDS",
        cutoff=radius,
    )

    data["edge_index_i"] = i
    data["edge_index_j"] = j
    data["edge_distance"] = d
    data["edge_distance_vector"] = D
    data["edge_cell_offset"] = S

    return data


dataset = dataset.map(map_fn, batched=False)

# %%
