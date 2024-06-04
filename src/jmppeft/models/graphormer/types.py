from collections.abc import Sequence
from typing import NamedTuple

import ll
import ll.typecheck as tc
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data as TorchGeoData

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
cutoff = 8.0


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

    pos = torch.cat([pos, used_expand_pos], dim=0)
    atoms = torch.cat([atoms, used_expand_atoms])
    real_mask = torch.cat(
        [
            torch.ones_like(tags, dtype=torch.bool),
            torch.zeros_like(used_expand_atoms, dtype=torch.bool),
        ]
    )


class Data(NamedTuple):
    pos: torch.Tensor
    atoms: torch.Tensor
    tags: torch.Tensor
    real_mask: torch.Tensor
    deltapos: torch.Tensor
    y_relaxed: torch.Tensor
    fixed: torch.Tensor
    natoms: torch.Tensor
    sid: torch.Tensor
    cell: torch.Tensor

    def to(self, device):
        return Data(
            pos=self.pos.to(device),
            atoms=self.atoms.to(device),
            tags=self.tags.to(device),
            real_mask=self.real_mask.to(device),
            deltapos=self.deltapos.to(device),
            y_relaxed=self.y_relaxed.to(device),
            fixed=self.fixed.to(device),
            natoms=self.natoms.to(device),
            sid=self.sid.to(device),
            cell=self.cell.to(device),
        )

    @classmethod
    def from_torch_geometric_data(
        cls,
        # data: TorchGeoData,
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

        dist: torch.Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(
            dim=-1
        )
        used_mask = (dist < cutoff).any(dim=0)
        if no_copy_tag is not None:
            used_mask = used_mask & (tags.ne(no_copy_tag).repeat(n_cells))

        used_expand_pos = expand_pos[used_mask]
        used_expand_atoms = atoms.repeat(n_cells)[used_mask]

        return cls(
            pos=torch.cat([pos, used_expand_pos], dim=0),
            atoms=torch.cat([atoms, used_expand_atoms]),
            tags=torch.cat([tags, used_expand_tags]),
            real_mask=torch.cat(
                [
                    torch.ones_like(tags, dtype=torch.bool),
                    torch.zeros_like(used_expand_atoms, dtype=torch.bool),
                ]
            ),
            cell=cell.squeeze(dim=0),
        )


def _pad(
    data_list: Sequence[Data],
    attr: str,
    batch_first: bool = True,
    padding_value: float = 0,
):
    return pad_sequence(
        [getattr(d, attr) for d in data_list],
        batch_first=batch_first,
        padding_value=padding_value,
    )


class Batch(NamedTuple):
    pos: torch.Tensor
    atoms: torch.Tensor
    tags: torch.Tensor
    real_mask: torch.Tensor
    deltapos: torch.Tensor
    y_relaxed: torch.Tensor
    fixed: torch.Tensor
    natoms: torch.Tensor
    sid: torch.Tensor
    cell: torch.Tensor

    def to(self, device):
        return Batch(
            pos=self.pos.to(device),
            atoms=self.atoms.to(device),
            tags=self.tags.to(device),
            real_mask=self.real_mask.to(device),
            deltapos=self.deltapos.to(device),
            y_relaxed=self.y_relaxed.to(device),
            fixed=self.fixed.to(device),
            natoms=self.natoms.to(device),
            sid=self.sid.to(device),
            cell=self.cell.to(device),
        )

    @classmethod
    def from_data_list(cls, data_list: Sequence[Data]):
        batch = cls(
            pos=_pad(data_list, "pos"),
            atoms=_pad(data_list, "atoms"),
            tags=_pad(data_list, "tags"),
            real_mask=_pad(data_list, "real_mask"),
            deltapos=_pad(data_list, "deltapos"),
            y_relaxed=_pad(data_list, "y_relaxed"),
            fixed=_pad(data_list, "fixed"),
            natoms=_pad(data_list, "natoms"),
            sid=_pad(data_list, "sid"),
            cell=_pad(data_list, "cell"),
        )
        return [batch]
