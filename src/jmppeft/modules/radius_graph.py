from typing import (
    TypeVarTuple,
    Unpack,
    cast,
)

import torch
from nshtrainer.ll import TypedConfig
from torch_geometric.data.data import BaseData
from torch_geometric.nn import radius_graph

from ._radius_graph_util import (
    compute_neighbors,
    get_pbc_distances,
    radius_graph_pbc,
)


class RadiusGraphConfig(TypedConfig):
    radius: float
    """The cutoff radius for the graph"""

    max_neighbors: int
    """Maximum number of neighbors per node"""

    radius_lower: float | None = None
    """Lower cutoff radius for interatomic interactions"""

    pbc_dims: tuple[bool, bool, bool] = (False, False, False)
    """What dimensions are periodic?"""

    enforce_max_neighbors_strictly: bool = False
    """
    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.
    """

    mask_out_self_loops: bool = False
    """Mask out self-loops in the graph"""

    def compute_inplace_(self, batch: BaseData):
        return compute_radius_graph_inplace_(batch, self)


TAttrs = TypeVarTuple("TAttrs")


def _apply_edge_mask(
    edge_index: torch.Tensor,  # 2 E
    mask: torch.Tensor,  # E
    edge_attrs: tuple[Unpack[TAttrs]] = (),
) -> tuple[torch.Tensor, tuple[Unpack[TAttrs]]]:
    edge_index = edge_index[:, mask]
    edge_attrs = cast(
        tuple[Unpack[TAttrs]],
        tuple(cast(torch.Tensor, attr)[mask] for attr in edge_attrs),
    )

    return edge_index, edge_attrs


def _no_self_loops_mask(
    edge_index: torch.Tensor,  # 2 E
    cell_offsets: torch.Tensor | None,  # E 3
):
    mask = edge_index[0] != edge_index[1]
    if cell_offsets is not None:
        # Also make sure the source/target cell are the same
        same_cell_mask = torch.all(cell_offsets == 0, dim=1)
        # ^ Shape: E
        mask = mask & same_cell_mask

    return mask


def compute_radius_graph_inplace_(batch: BaseData, config: RadiusGraphConfig):
    # PBC graphs use `radius_graph_pbc`
    if any(config.pbc_dims):
        assert "cell" in batch, (
            "BaseData does not contain `cell` property, necessary for PBC radius graph computation. "
            "Are you sure this is a PBC dataset?"
        )

        edge_index, edge_cell_offsets, _ = radius_graph_pbc(
            batch.pos,
            batch.natoms,
            batch.cell,
            config.radius,
            config.max_neighbors,
            config.enforce_max_neighbors_strictly,
        )
        # Mask out self loops if necessary
        if config.mask_out_self_loops:
            edge_index, (edge_cell_offsets,) = _apply_edge_mask(
                edge_index,
                _no_self_loops_mask(edge_index, edge_cell_offsets),
                edge_attrs=(edge_cell_offsets,),
            )
        # Compute the number of neighbors for each node
        neighbors = compute_neighbors(
            batch.pos.shape[0],
            batch.natoms,
            edge_index,
        )

        out = get_pbc_distances(
            batch.pos,
            edge_index,
            batch.cell,
            edge_cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )
        edge_index = out["edge_index"]
        edge_distances = out["distances"]
        edge_cell_offset_distances = out["offsets"]
        edge_displacement_vectors = out["edge_displacement_vectors"]
    # Non-PBC graphs use `radius_graph`
    else:
        edge_index = radius_graph(
            batch.pos,
            r=config.radius,
            batch=batch.batch,
            max_num_neighbors=config.max_neighbors,
        )
        if config.mask_out_self_loops:
            edge_index, () = _apply_edge_mask(
                edge_index,
                _no_self_loops_mask(edge_index, None),
            )

        j, i = edge_index
        edge_displacement_vectors = batch.pos[j] - batch.pos[i]

        edge_distances = edge_displacement_vectors.norm(dim=-1)
        edge_cell_offsets = torch.zeros(edge_index.shape[1], 3, device=batch.pos.device)
        edge_cell_offset_distances = torch.zeros_like(
            edge_cell_offsets, device=batch.pos.device
        )

        # Compute the number of neighbors for each node
        neighbors = compute_neighbors(
            batch.pos.shape[0],
            batch.natoms,
            edge_index,
        )

    if config.radius_lower:
        num_edges_prev = edge_index.shape[1]
        (
            edge_index,
            (
                edge_displacement_vectors,
                edge_distances,
                edge_cell_offsets,
                edge_cell_offset_distances,
            ),
        ) = _apply_edge_mask(
            edge_index,
            edge_distances >= config.radius_lower,  # Shape: E
            edge_attrs=(
                edge_displacement_vectors,
                edge_distances,
                edge_cell_offsets,
                edge_cell_offset_distances,
            ),
        )

        # Recompute neighbors if edges were removed
        if edge_index.shape[1] < num_edges_prev:
            neighbors = compute_neighbors(batch.pos.shape[0], batch.natoms, edge_index)

    batch.edge_index = edge_index
    batch.edge_displacement_vectors = edge_displacement_vectors
    batch.edge_distances = edge_distances
    batch.edge_cell_offsets = edge_cell_offsets
    batch.edge_cell_offset_distances = edge_cell_offset_distances
    batch.neighbors = neighbors

    return batch
