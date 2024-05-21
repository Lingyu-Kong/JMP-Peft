from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from logging import getLogger
from typing import Generic, Literal, cast

import ll
import torch
import torch.nn as nn
from ll.nn import TypedModuleDict
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData, Data
from typing_extensions import TypeVar, assert_never, final, override

from ...models.gemnet.backbone import GOCBackboneOutput
from ...modules.relaxer import LightningModuleRelaxerMixin, RelaxerConfig
from .base import FinetuneConfigBase, FinetuneModelBase
from .output_head import (
    GradientForcesTargetConfig,
    GradientOutputHeadInput,
    GraphScalarTargetConfig,
    GraphTargetConfig,
    NodeTargetConfig,
    NodeVectorTargetConfig,
    OutputHeadInput,
)

log = getLogger(__name__)


class EnergyForcesConfigBase(FinetuneConfigBase):
    graph_targets: list[GraphTargetConfig] = [
        GraphScalarTargetConfig(name="y", loss_coefficient=1.0),
    ]
    node_targets: list[NodeTargetConfig] = [
        GradientForcesTargetConfig(
            name="force",
            energy_name="y",
            loss_coefficient=100.0,
        ),
    ]

    relaxer: RelaxerConfig | None = None
    """Relaxer configuration. If None, relaxer is disabled."""

    def energy_config_(self):
        self.graph_targets = [
            GraphScalarTargetConfig(name="y", loss_coefficient=1.0),
        ]
        self.node_targets = []

    def forces_config_(self, *, gradient: bool, coefficient: float = 100.0):
        self.graph_targets = (
            [
                GraphScalarTargetConfig(name="y", loss_coefficient=0.0),
            ]
            if gradient
            else []
        )
        self.node_targets = [
            GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=coefficient,
            )
            if gradient
            else NodeVectorTargetConfig(name="force", loss_coefficient=coefficient),
        ]

        if gradient:
            self.trainer.inference_mode = False

    def energy_forces_config_(
        self,
        *,
        gradient: bool,
        energy_coefficient: float = 1.0,
        force_coefficient: float = 100.0,
        force_loss: Literal["mae", "l2mae"] = "l2mae",
    ):
        self.graph_targets = [
            GraphScalarTargetConfig(name="y", loss_coefficient=energy_coefficient),
        ]
        self.node_targets = [
            GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=force_coefficient,
            )
            if gradient
            else NodeVectorTargetConfig(
                name="force", loss_coefficient=force_coefficient, loss=force_loss
            ),
        ]

        if gradient:
            self.trainer.inference_mode = False

    def supports_inference_mode(self):
        return all(t.supports_inference_mode() for t in self.node_targets)

    @property
    def gradient_force_target(self):
        return next(
            (
                target
                for target in self.node_targets
                if isinstance(target, GradientForcesTargetConfig)
            ),
            None,
        )

    @override
    def __post_init__(self):
        super().__post_init__()

        if not self.supports_inference_mode():
            assert not self.trainer.inference_mode, "`config.trainer.inference_mode` is True, but the model does not support inference mode."

        if any(
            isinstance(target, NodeVectorTargetConfig) for target in self.node_targets
        ):
            assert (
                self.trainer.inference_mode
            ), "NodeVectorTargetConfig requires inference mode to be enabled."
            assert (
                self.backbone.regress_forces
            ), "NodeVectorTargetConfig requires `backbone.regress_forces` to be True."
            assert (
                self.backbone.direct_forces
            ), "NodeVectorTargetConfig requires `backbone.direct_forces` to be True."


TConfig = TypeVar("TConfig", bound=EnergyForcesConfigBase, infer_variance=True)


class EnergyForcesModelBase(
    LightningModuleRelaxerMixin[TConfig],
    FinetuneModelBase[TConfig],
    nn.Module,
    ABC,
    Generic[TConfig],
):
    @override
    def construct_output_heads(self):
        self.graph_outputs = TypedModuleDict(
            {
                target.name: target.construct_output_head(
                    self.config.output,
                    self.config.backbone.emb_size_atom,
                    self.config.backbone.emb_size_edge,
                    self.config.activation_cls,
                )
                for target in self.config.graph_targets
            },
            key_prefix="ft_mlp_",
        )
        self.node_outputs = TypedModuleDict(
            {
                target.name: target.construct_output_head(
                    self.config.output,
                    self.config.backbone.emb_size_atom,
                    self.config.backbone.emb_size_edge,
                    self.config.activation_cls,
                )
                for target in self.config.node_targets
            },
            key_prefix="ft_mlp_",
        )

    @override
    def forward(self, data: BaseData):
        preds: dict[str, torch.Tensor] = {}
        with ExitStack() as stack:
            # Enter all the necessary contexts for output heads.
            # Right now, this is only for gradient forces, which
            #   requires torch.inference_mode(False), torch.enable_grad,
            #   and data.pos.requires_grad_(True).
            for target in self.config.targets:
                stack.enter_context(target.model_forward_context(data))

            # Generate graphs on the GPU
            data = self.generate_graphs_transform(data)

            # Run the backbone
            atomic_numbers = data.atomic_numbers - 1
            h = self.embedding(atomic_numbers)  # (N, d_model)
            out = cast(GOCBackboneOutput, self.backbone(data, h=h))

            output_head_input: OutputHeadInput = {
                "backbone_output": out,
                "data": data,
            }
            graph_preds = {
                target: module(output_head_input)
                for target, module in self.graph_outputs.items()
            }
            preds.update(graph_preds)

            # We need to send the graph predictions to the node output heads
            #   in case they need to use the graph predictions to compute the
            #   node predictions.
            # Currently, this is only the case for the gradient forces target,
            #   which needs to use the energy to compute the forces.
            node_output_head_input: GradientOutputHeadInput = {
                "backbone_output": out,
                "data": data,
                "graph_preds": graph_preds,
            }
            node_preds = {
                target: module(node_output_head_input)
                for target, module in self.node_outputs.items()
            }
            preds.update(node_preds)

        return preds

    @abstractmethod
    def generate_graphs_transform(self, data: BaseData) -> BaseData: ...

    @override
    def relaxer_config(self) -> RelaxerConfig | None:
        return self.config.relaxer

    @override
    def relaxer_collate_fn(self, data_list: list[BaseData]) -> Batch:
        return self.collate_fn(data_list)
