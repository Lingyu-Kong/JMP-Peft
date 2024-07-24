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
import torch
from jmppeft.utils.goc_graph import get_max_neighbors_mask
from matscipy.neighbours import neighbour_list


def _subselect_edges(
    edge_index: torch.Tensor,
    cell_offset: torch.Tensor,
    distance: torch.Tensor,
    natoms: torch.Tensor,
    cutoff: float | None,
    max_neighbors: int | None,
):
    if cutoff is not None:
        edge_mask = distance <= cutoff
        edge_index = edge_index[:, edge_mask]
        cell_offset = cell_offset[edge_mask]
        distance = distance[edge_mask]

    if max_neighbors is not None:
        edge_mask, _ = get_max_neighbors_mask(
            natoms=natoms,
            index=edge_index[1],
            atom_distance=distance,
            max_num_neighbors_threshold=max_neighbors,
        )
        edge_index = edge_index[:, edge_mask]
        cell_offset = cell_offset[edge_mask]
        # distance = distance[edge_mask]

    if edge_index.shape[1] == 0:
        raise ValueError("An image has no neighbors")

    return edge_index, cell_offset


full_cutoff = 12.0
configs = {
    "aint": {"cutoff": 8.0, "max_neighbors": 1000},
    "main": {"cutoff": 8.0, "max_neighbors": 15},
    "qint": {"cutoff": 8.0, "max_neighbors": 4},
    "aeaint": {"cutoff": 8.0, "max_neighbors": 10},
}


def map_fn(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray):
    i, j, d, S = neighbour_list(
        "ijdS",
        positions=positions,
        cell=cell,
        pbc=list(pbc),
        cutoff=full_cutoff,
    )

    data = {}
    # data["edges_i"] = i
    # data["edges_j"] = j
    # data["edges_cell_offsets"] = S
    data["neighbors"] = int(i.shape[0])

    if False:
        edge_index = torch.from_numpy(np.vstack((i, j)).astype(np.int64)).long()
        cell_offsets = torch.from_numpy(S.astype(np.float32)).float()
        distances = torch.from_numpy(d.astype(np.float32)).float()
        natoms = torch.tensor([positions.shape[0]], dtype=torch.long)

        for key, config in configs.items():
            edge_index_sub, cell_offsets_sub = _subselect_edges(
                edge_index=edge_index,
                cell_offset=cell_offsets,
                distance=distances,
                natoms=natoms,
                cutoff=config["cutoff"],
                max_neighbors=config["max_neighbors"],
            )
            data[f"edges_{key}_i"] = edge_index_sub[0]
            data[f"edges_{key}_j"] = edge_index_sub[1]
            data[f"edges_{key}_cell_offsets"] = cell_offsets_sub
            data[f"neighbors_{key}"] = int(edge_index_sub.shape[1])

    return data


data = dataset[0]
map_fn(data["positions"], data["cell"], data["pbc"])

# %%
dataset = dataset.map(
    map_fn,
    batched=False,
    num_proc=16,
    input_columns=["positions", "cell", "pbc"],
)

# %%
(dataset["num_atoms"] < 4)
