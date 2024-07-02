# %%
import ll

ll.pretty()
# %%
import datasets

dataset = datasets.load_dataset("nimashoghi/mptrj", split="val")
assert isinstance(dataset, datasets.Dataset)
dataset.set_format("torch")
dataset

# %%
from functools import partial
from typing import cast

import torch
from jmppeft.utils.goc_graph import (
    Cutoffs,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from matscipy.neighbours import neighbour_list
from torch_geometric.data.data import BaseData, Data

# radius = 8.0
cutoffs = Cutoffs.from_constant(8.0)
# max_neighbors = 15
max_neighbors = MaxNeighbors.from_goc_base_proportions(15)
pbc = True
qint_tags = [0, 1, 2]


def map_fn(data_dict: dict[str, torch.Tensor]):
    data = Data(
        pos=data_dict["positions"],
        cell=data_dict["cell"].unsqueeze(0),
        natoms=data_dict["num_atoms"],
        tags=torch.zeros_like(data_dict["numbers"], dtype=torch.long),
    )
    data = cast(BaseData, data)

    aint_graph = generate_graph(
        data, cutoff=cutoffs.aint, max_neighbors=max_neighbors.aint, pbc=pbc
    )
    subselect = partial(
        subselect_graph,
        data,
        aint_graph,
        cutoff_orig=cutoffs.aint,
        max_neighbors_orig=max_neighbors.aint,
    )
    main_graph = subselect(cutoffs.main, max_neighbors.main)
    aeaint_graph = subselect(cutoffs.aeaint, max_neighbors.aeaint)
    qint_graph = subselect(cutoffs.qint, max_neighbors.qint)

    # We can't do this at the data level: This is because the batch collate_fn doesn't know
    # that it needs to increment the "id_swap" indices as it collates the data.
    # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
    # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
    qint_graph = tag_mask(data, qint_graph, tags=qint_tags)

    graphs = {
        "main": main_graph,
        "a2a": aint_graph,
        "a2ee2a": aeaint_graph,
        "qint": qint_graph,
    }

    for graph_type, graph in graphs.items():
        for key, value in graph.items():
            # setattr(data, f"{graph_type}_{key}", value)
            data_dict[f"{graph_type}_{key}"] = value

    return data_dict


map_fn(dataset[0])

# %%
dataset = dataset.map(map_fn, batched=False)

# %%
