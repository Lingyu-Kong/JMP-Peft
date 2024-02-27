from abc import ABC, abstractmethod
from contextlib import ExitStack
from logging import getLogger
from typing import Generic, cast

import torch
import torch.nn as nn
from ll.nn import TypedModuleDict
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar, override

from ...models.gemnet.backbone import GOCBackboneOutput
from ...modules.dataset import dataset_transform as DT
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

    def energy_config_(self):
        self.graph_targets = [
            GraphScalarTargetConfig(name="y", loss_coefficient=1.0),
        ]
        self.node_targets = []

    def forces_config_(self, *, gradient: bool):
        self.graph_targets = [
            GraphScalarTargetConfig(name="y", loss_coefficient=0.0),
        ]
        self.node_targets = [
            GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=100.0,
            )
            if gradient
            else NodeVectorTargetConfig(name="force", loss_coefficient=100.0),
        ]

    def energy_forces_config_(self, *, gradient: bool):
        self.graph_targets = [
            GraphScalarTargetConfig(name="y", loss_coefficient=1.0),
        ]
        self.node_targets = [
            GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=100.0,
            )
            if gradient
            else NodeVectorTargetConfig(name="force", loss_coefficient=100.0),
        ]

    def should_compute_graph_in_forward(self):
        return any(t.should_compute_graph_in_forward() for t in self.node_targets)

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
            assert (
                not self.trainer.inference_mode
            ), "`config.trainer.inference_mode` is True, but the model does not support inference mode."


TConfig = TypeVar("TConfig", bound=EnergyForcesConfigBase, infer_variance=True)


class EnergyForcesModelBase(
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

            if self.config.should_compute_graph_in_forward():
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
    def generate_graphs_transform(self, data: BaseData) -> BaseData:
        ...

    def _generate_graphs_transform(self, data: BaseData):
        if self.config.should_compute_graph_in_forward():
            # We need to compute the graphs in the forward method
            # so that we can compute the forces using the energy
            # and the positions.
            return data
        return self.generate_graphs_transform(data)

    @override
    def train_dataset(self):
        if (dataset := super().train_dataset()) is None:
            return None

        dataset = DT.transform(dataset, transform=self._generate_graphs_transform)
        return dataset

    @override
    def val_dataset(self):
        if (dataset := super().val_dataset()) is None:
            return None

        dataset = DT.transform(dataset, transform=self._generate_graphs_transform)
        return dataset

    @override
    def test_dataset(self):
        if (dataset := super().test_dataset()) is None:
            return None

        dataset = DT.transform(dataset, transform=self._generate_graphs_transform)
        return dataset
