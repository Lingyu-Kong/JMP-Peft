# %%
import ll

ll.pretty()
# %%
import datasets

datasets.disable_caching()

dataset = datasets.load_dataset("nimashoghi/mptrj", split="train")
assert isinstance(dataset, datasets.Dataset)
# dataset.set_format("torch")
dataset

# %%
from functools import partial
from typing import Any, cast

import torch
from jmppeft.utils.goc_graph import (
    Cutoffs,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData, Data
from torch_geometric.utils import cumsum, degree, unbatch, unbatch_edge_index


def map_fn(
    data_dict: dict[str, Any],
    # rank: int,
    *,
    device: int | str | torch.device | None,
    cutoffs: Cutoffs,
    max_neighbors: MaxNeighbors,
    pbc: bool,
    qint_tags: tuple[int, ...],
):
    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)

    def _to_data_single(data_dict: dict[str, Any]) -> BaseData:
        data = Data(
            pos=torch.tensor(data_dict["positions"], dtype=torch.float, device=device),
            cell=torch.tensor(
                data_dict["cell"], dtype=torch.float, device=device
            ).unsqueeze(0),
            natoms=torch.tensor(
                data_dict["num_atoms"], dtype=torch.long, device=device
            ),
            tags=torch.zeros(data_dict["num_atoms"], dtype=torch.long, device=device),
        ).to(device)
        return cast(BaseData, data)

    def _to_data(data_dict: dict[str, Any]) -> BaseData:
        if isinstance(data_dict["num_atoms"], int):
            return _to_data_single(data_dict)

        batch_size = len(data_dict["num_atoms"])
        batch = Batch.from_data_list(
            [
                _to_data_single({key: value[i] for key, value in data_dict.items()})
                for i in range(batch_size)
            ]
        ).to(device)
        return cast(BaseData, batch)

    data = _to_data(data_dict)
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
    qint_graph = tag_mask(data, qint_graph, tags=list(qint_tags))

    graphs = {
        "main": main_graph,
        "a2a": aint_graph,
        "a2ee2a": aeaint_graph,
        "qint": qint_graph,
    }

    def _pythonify(value):
        if not torch.is_tensor(value):
            return value

        value = value.detach().cpu()
        # If scalar, return scalar
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    if isinstance(data_dict["num_atoms"], int):
        for graph_type, graph in graphs.items():
            for key, value in graph.items():
                if key in ("cutoff", "max_neighbors"):
                    continue

                # if key == "edge_index":
                #     data_dict[f"{graph_type}_{key}_i"] = _pythonify(value[0])
                #     data_dict[f"{graph_type}_{key}_j"] = _pythonify(value[1])
                #     continue

                data_dict[f"{graph_type}_{key}"] = _pythonify(value)
    else:
        batch_size = len(data_dict["num_atoms"])

        for graph_type, graph in graphs.items():
            assert (
                len(graph["num_neighbors"]) == batch_size
            ), f"{len(graph['num_neighbors'])=} != {batch_size=}"

            edge_index = graph["edge_index"]
            deg = degree(data.batch, batch_size, dtype=torch.long)
            ptr = cumsum(deg)

            edge_batch = data.batch[edge_index[0]]
            edge_index = edge_index - ptr[edge_batch]
            sizes = degree(edge_batch, batch_size, dtype=torch.long).cpu().tolist()
            # return edge_index.split(sizes, dim=1)

            for key, value in graph.items():
                if value.ndim == 0:
                    value = [_pythonify(value) for _ in range(batch_size)]
                elif key == "edge_index":
                    value = [_pythonify(v) for v in value.split(sizes, dim=1)]
                elif value.shape[0] == edge_index.shape[1]:
                    value = [_pythonify(v) for v in value.split(sizes, dim=0)]
                else:
                    assert (
                        value.shape[0] == batch_size
                    ), f"{value.shape=} != {batch_size=}"
                    value = _pythonify(value)

                data_dict[f"{graph_type}_{key}"] = value

    return data_dict


kwargs = dict(
    cutoffs=Cutoffs.from_constant(8.0),
    max_neighbors=MaxNeighbors.from_goc_base_proportions(15),
    pbc=True,
    qint_tags=(0, 1, 2),
    device="cuda",
)
map_fn(dataset[0:100], **kwargs)

# %%
dataset = dataset.map(
    map_fn,
    batched=True,
    batch_size=4096,
    fn_kwargs=kwargs,
    # num_proc=8,
    # with_rank=True,
    # num_proc=torch.cuda.device_count(),
).with_format("torch")

dataset[0]
