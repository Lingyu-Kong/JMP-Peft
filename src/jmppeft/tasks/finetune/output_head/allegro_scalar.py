from logging import getLogger
from typing import Literal

import ll.typecheck as tc
import torch
import torch.nn as nn
from einops import rearrange
from jmppeft.modules.torch_scatter_polyfill import scatter
from ll.nn import MLP
from torch_geometric.data.data import BaseData
from typing_extensions import TypedDict, override

from ....models.gemnet.backbone import GOCBackboneOutput
from ....modules.loss import LossConfig
from ...config import OutputConfig
from ._base import BaseTargetConfig

log = getLogger(__name__)


class AllegroScalarTargetConfig(BaseTargetConfig):
    kind: Literal["allegro_scalar"] = "allegro_scalar"

    reduction: Literal["mean", "sum"] = "sum"
    """The reduction to use for the output."""

    loss: LossConfig
    """The loss configuration for the target."""

    max_atomic_number: int
    """The max atomic number in the dataset."""

    edge_level_energies: bool = False
    """Whether to use edge level energies."""

    @override
    def is_classification(self) -> bool:
        return False

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return AllegroScalarOutputHead(
            self,
            output_config,
            d_model_node,
            d_model_edge,
            activation_cls,
        )


class OutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput


class AllegroScalarOutputHead(nn.Module):
    @override
    def __init__(
        self,
        target_config: AllegroScalarTargetConfig,
        output_config: OutputConfig,
        d_model: int,
        d_model_edge: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        self.target_config = target_config
        self.output_config = output_config
        self.out_mlp_node = MLP(
            ([d_model] * self.output_config.num_mlps) + [1],
            activation=activation_cls,
        )

        self.per_atom_scales = nn.Embedding(
            self.target_config.max_atomic_number + 1,
            1,
            padding_idx=0,
        )
        nn.init.ones_(self.per_atom_scales.weight)

        self.per_atom_shifts = nn.Embedding(
            self.target_config.max_atomic_number + 1,
            1,
            padding_idx=0,
        )
        nn.init.zeros_(self.per_atom_shifts.weight)

        if self.target_config.edge_level_energies:
            self.out_mlp_edge = MLP(
                ([d_model_edge] * self.output_config.num_mlps) + [1],
                activation=activation_cls,
            )

            num_atom_pairs = (self.target_config.max_atomic_number + 1) ** 2
            self.pairwise_scales = nn.Embedding(num_atom_pairs, 1, padding_idx=0)
            nn.init.ones_(self.pairwise_scales.weight)

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        atomic_numbers = data.atomic_numbers
        tc.tassert(tc.Int[torch.Tensor, "n"], atomic_numbers)

        # Compute node-level energies from node embeddings
        per_atom_energies = backbone_output["energy"]
        tc.tassert(tc.Float[torch.Tensor, "n d_model"], per_atom_energies)
        per_atom_energies = self.out_mlp_node(per_atom_energies)
        tc.tassert(tc.Float[torch.Tensor, "n 1"], per_atom_energies)

        if self.target_config.edge_level_energies:
            # Compute edge-level energies from edge embeddings
            per_edge_energies = backbone_output["forces"]
            tc.tassert(tc.Float[torch.Tensor, "e d_model_edge"], per_edge_energies)
            per_edge_energies = self.out_mlp_edge(per_edge_energies)
            tc.tassert(tc.Float[torch.Tensor, "e 1"], per_edge_energies)

            # Multiply edge energies by pairwise scales
            # Compute the pairwise indices
            idx_s, idx_t = backbone_output["idx_s"], backbone_output["idx_t"]
            tc.tassert(tc.Int[torch.Tensor, "e"], (idx_s, idx_t))
            pair_idx = (
                atomic_numbers[idx_s] * (self.target_config.max_atomic_number + 1)
                + atomic_numbers[idx_t]
            )
            tc.tassert(tc.Int[torch.Tensor, "e"], pair_idx)

            # Get the pairwise scales
            pairwise_scales = self.pairwise_scales(pair_idx)
            tc.tassert(tc.Float[torch.Tensor, "e 1"], pairwise_scales)

            # Multiply edge energies by pairwise scales
            per_edge_energies = per_edge_energies * pairwise_scales

            # Add to node energies
            per_atom_energies_per_edge = scatter(
                per_edge_energies,
                idx_t,
                dim=0,
                dim_size=atomic_numbers.shape[0],
                reduce=self.target_config.reduction,
            )
            tc.tassert(tc.Float[torch.Tensor, "n 1"], per_atom_energies_per_edge)
            per_atom_energies = per_atom_energies + per_atom_energies_per_edge

        per_atom_scales = self.per_atom_scales(atomic_numbers)
        per_atom_shifts = self.per_atom_shifts(atomic_numbers)
        tc.tassert(tc.Float[torch.Tensor, "n 1"], (per_atom_scales, per_atom_shifts))

        per_atom_energies = per_atom_energies * per_atom_scales + per_atom_shifts
        tc.tassert(tc.Float[torch.Tensor, "n 1"], per_atom_energies)

        per_system_energies = scatter(
            per_atom_energies,
            data.batch,
            dim=0,
            dim_size=data.num_graphs,
            reduce=self.target_config.reduction,
        )
        tc.tassert(tc.Float[torch.Tensor, "b 1"], per_system_energies)

        per_system_energies = rearrange(per_system_energies, "b 1 -> b")
        return per_system_energies
