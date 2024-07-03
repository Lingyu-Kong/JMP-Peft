from logging import getLogger
from typing import Literal

import torch
import torch.nn as nn
from jmppeft.modules.torch_scatter_polyfill import scatter
from ll.nn import MLP
from torch_geometric.data.data import BaseData
from typing_extensions import TypedDict, override

from ....models.gemnet.backbone import GOCBackboneOutput
from ....modules.loss import L2MAELossConfig, LossConfig
from ...config import OutputConfig
from ._base import BaseTargetConfig

log = getLogger(__name__)


class NodeVectorTargetConfig(BaseTargetConfig):
    kind: Literal["vector"] = "vector"

    reduction: Literal["sum", "mean"] = "sum"
    """
    The reduction method for the target. This refers to how the target is computed.
    For example, for graph scalar targets, this refers to how the scalar targets are
    computed from each node's scalar prediction.
    """

    loss: LossConfig = L2MAELossConfig()
    """The loss function to use for the target"""

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return NodeVectorOutputHead(
            self,
            output_config,
            d_model_edge,
            activation_cls,
        )


class NodeVectorOutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput


class NodeVectorOutputHead(nn.Module):
    @override
    def __init__(
        self,
        target_config: NodeVectorTargetConfig,
        output_config: OutputConfig,
        d_model: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        self.target_config = target_config
        self.output_config = output_config
        self.d_model = d_model
        self.out_mlp = MLP(
            ([self.d_model] * self.output_config.num_mlps) + [1],
            activation=activation_cls,
        )

    @override
    def forward(self, input: NodeVectorOutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_atoms = data.atomic_numbers.shape[0]

        output = self.out_mlp(backbone_output["forces"])
        output = output * backbone_output["V_st"]  # (n_edges, 3)
        output = scatter(
            output,
            backbone_output["idx_t"],
            dim=0,
            dim_size=n_atoms,
            reduce=self.target_config.reduction,
        )
        return output
