from logging import getLogger
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from jmppeft.modules.torch_scatter_polyfill import scatter
from ll.nn import MLP
from torch_geometric.data.data import BaseData
from typing_extensions import TypedDict, override

from ....models.gemnet.backbone import GOCBackboneOutput
from ....modules.loss import LossConfig, MAELossConfig
from ...config import OutputConfig
from ._base import BaseTargetConfig

log = getLogger(__name__)


class GraphScalarTargetConfig(BaseTargetConfig):
    kind: Literal["scalar"] = "scalar"

    loss: LossConfig = MAELossConfig()

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return GraphScalarOutputHead(
            self,
            output_config,
            d_model_node,
            activation_cls,
        )


class GraphBinaryClassificationTargetConfig(BaseTargetConfig):
    kind: Literal["binary"] = "binary"

    num_classes: int
    """The number of classes for the target"""

    pos_weight: float | None = None
    """The positive weight for the target"""

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.num_classes != 2:
            raise ValueError(
                f"Binary classification target {self.name} has {self.num_classes} classes"
            )

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return GraphBinaryClassificationOutputHead(
            self,
            output_config,
            d_model_node,
            activation_cls,
        )


class GraphMulticlassClassificationTargetConfig(BaseTargetConfig):
    kind: Literal["multiclass"] = "multiclass"

    num_classes: int
    """The number of classes for the target"""

    class_weights: list[float] | None = None
    """The class weights for the target"""

    dropout: float | None = None
    """The dropout probability to use before the output layer"""

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return GraphMulticlassClassificationOutputHead(
            self,
            output_config,
            d_model_node,
            activation_cls,
        )


class OutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput


class GraphScalarOutputHead(nn.Module):
    @override
    def __init__(
        self,
        target_config: GraphScalarTargetConfig,
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
    def forward(
        self,
        input: OutputHeadInput,
        *,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_molecules = int(torch.max(data.batch).item() + 1)

        output = self.out_mlp(backbone_output["energy"])  # (n_atoms, 1)
        if scale is not None:
            output = output * scale
        if shift is not None:
            output = output + shift

        output = scatter(
            output,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.target_config.reduction,
        )  # (bsz, 1)
        output = rearrange(output, "b 1 -> b")
        return output


class GraphBinaryClassificationOutputHead(nn.Module):
    @override
    def __init__(
        self,
        target_config: GraphBinaryClassificationTargetConfig,
        output_config: OutputConfig,
        d_model: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        assert target_config.num_classes == 2, "Only binary classification supported"

        self.target_config = target_config
        self.output_config = output_config
        self.d_model = d_model
        self.out_mlp = MLP(
            ([self.d_model] * self.output_config.num_mlps) + [1],
            activation=activation_cls,
        )

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        n_molecules = int(torch.max(data.batch).item() + 1)

        output = self.out_mlp(backbone_output["energy"])  # (n, num_classes)
        output = scatter(
            output,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.target_config.reduction,
        )  # (bsz, num_classes)
        output = rearrange(output, "b 1 -> b")
        return output


class GraphMulticlassClassificationOutputHead(nn.Module):
    @override
    def __init__(
        self,
        target_config: GraphMulticlassClassificationTargetConfig,
        output_config: OutputConfig,
        d_model: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        self.target_config = target_config
        self.output_config = output_config
        self.d_model = d_model
        self.out_mlp = MLP(
            ([self.d_model] * self.output_config.num_mlps)
            + [self.target_config.num_classes],
            activation=activation_cls,
        )

        self.dropout = None
        if self.target_config.dropout:
            self.dropout = nn.Dropout(self.target_config.dropout)

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        n_molecules = int(torch.max(data.batch).item() + 1)

        x = input["backbone_output"]["energy"]
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.out_mlp(x)  # (n, num_classes)
        x = scatter(
            x,
            data.batch,
            dim=0,
            dim_size=n_molecules,
            reduce=self.target_config.reduction,
        )  # (bsz, num_classes)
        return x
