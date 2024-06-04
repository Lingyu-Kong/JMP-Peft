from collections.abc import Callable, Sequence
from typing import TypeAlias, TypedDict, cast

import ll.typecheck as tc
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data.batch import Batch as TorchGeoBatch
from torch_geometric.data.data import BaseData as TorchGeoData

cell_offsets = torch.tensor(
    [
        [-1, -1, 0],
        [-1, 0, 0],
        [-1, 1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [1, -1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ],
).float()
n_cells = cell_offsets.size(0)


class DenseData(TypedDict):
    atoms: tc.Int[torch.Tensor, "n"] | tc.Int[torch.Tensor, "b n"]
    pos: tc.Float[torch.Tensor, "n 3"] | tc.Float[torch.Tensor, "b n 3"]
    real_mask: tc.Bool[torch.Tensor, "n"] | tc.Bool[torch.Tensor, "b n"]


Data: TypeAlias = TorchGeoData
Batch: TypeAlias = TorchGeoBatch


def pbc_graph_transformer(
    pos: tc.Float[torch.Tensor, "n 3"],
    cell: tc.Float[torch.Tensor, "3 3"],
    atoms: tc.Int[torch.Tensor, "n"],
    tags: tc.Int[torch.Tensor, "n"],
    *,
    cutoff: float,
    filter_src_pos_by_tag: int | None = None,  # should be 2 for oc20
    no_copy_tag: int | None = None,  # should be 2 for oc20 (no copy ads)
):
    global cell_offsets, n_cells
    offsets = torch.matmul(cell_offsets, cell).view(n_cells, 1, 3)
    expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets).view(-1, 3)

    src_pos = pos
    if filter_src_pos_by_tag is not None:
        src_pos = src_pos[tags == filter_src_pos_by_tag]

    dist: torch.Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
    used_mask = (dist < cutoff).any(dim=0)
    if no_copy_tag is not None:
        used_mask = used_mask & (tags.ne(no_copy_tag).repeat(n_cells))

    used_expand_pos = expand_pos[used_mask]
    used_expand_atoms = atoms.repeat(n_cells)[used_mask]

    real_mask = torch.cat(
        [
            torch.ones_like(atoms, dtype=torch.bool),
            torch.zeros_like(used_expand_atoms, dtype=torch.bool),
        ]
    )
    pos = torch.cat([pos, used_expand_pos], dim=0)
    atoms = torch.cat([atoms, used_expand_atoms])

    return pos, atoms, real_mask


def _pad(
    data_list: Sequence[DenseData],
    attr: str,
    batch_first: bool = True,
    padding_value: float = 0,
):
    return pad_sequence(
        [d[attr] for d in data_list],
        batch_first=batch_first,
        padding_value=padding_value,
    )


def collate_fn(
    data_list: list[Data],
    torch_geo_collate_fn: Callable[[list[TorchGeoData]], TorchGeoBatch],
) -> Batch:
    dense_data_list: list[DenseData] = []

    # Remove `jmp_dense_data` from data_list and collect it
    for d in data_list:
        # Make sure `jmp_dense_data` is present and pop it
        assert hasattr(d, "jmp_dense_data")
        dense_data_list.append(d.jmp_dense_data)
        del d.jmp_dense_data

    # Collate the rest normally
    torch_geo_batch = torch_geo_collate_fn(data_list)

    # Collate dense data separately and add it back as `jmp_dense_data`
    dense_batch = cast(
        DenseData, {k: _pad(dense_data_list, k) for k in dense_data_list[0].keys()}
    )
    torch_geo_batch.jmp_dense_data = dense_batch

    return torch_geo_batch


def data_transform(
    data: TorchGeoData,
    *,
    cutoff: float,
    filter_src_pos_by_tag: int | None = None,  # should be 2 for oc20
    no_copy_tag: int | None = None,  # should be 2 for oc20 (no copy ads)
) -> Data:
    atoms = data.atomic_numbers.long()
    pos = data.pos
    cell = data.cell.view(3, 3)
    tags = data.tags.long()

    pos, atoms, real_mask = pbc_graph_transformer(
        pos,
        cell,
        atoms,
        tags,
        cutoff=cutoff,
        filter_src_pos_by_tag=filter_src_pos_by_tag,
        no_copy_tag=no_copy_tag,
    )

    dense_data: DenseData = {"atoms": atoms, "pos": pos, "real_mask": real_mask}

    # Add a "jmp_dense_data" attribute to the data object
    assert not hasattr(data, "jmp_dense_data")
    data.jmp_dense_data = dense_data

    return data
