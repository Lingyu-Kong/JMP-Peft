from abc import ABC, abstractmethod
from contextlib import ExitStack
from logging import getLogger
from typing import Generic, Literal, cast

import ll
import torch
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar, override

from ...models.gemnet.backbone import GOCBackboneOutput
from ...modules.loss import L2MAELossConfig, LossConfig, MAELossConfig
from . import output_head
from .base import FinetuneConfigBase, FinetuneModelBase, SkipBatch

log = getLogger(__name__)


class EnergyForcesConfigBase(FinetuneConfigBase):
    graph_targets: list[output_head.GraphTargetConfig] = []
    node_targets: list[output_head.NodeTargetConfig] = []

    ignore_graph_generation_errors: bool = True
    """If True, ignore errors that occur during graph generation and skip the batch."""

    def energy_config_(self):
        self.graph_targets = [
            output_head.GraphScalarTargetConfig(name="y", loss_coefficient=1.0),
        ]
        self.node_targets = []

    def forces_config_(self, *, gradient: bool, coefficient: float = 100.0):
        self.graph_targets = (
            [
                output_head.GraphScalarTargetConfig(name="y", loss_coefficient=0.0),
            ]
            if gradient
            else []
        )
        self.node_targets = [
            output_head.GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=coefficient,
            )
            if gradient
            else output_head.NodeVectorTargetConfig(
                name="force", loss_coefficient=coefficient
            ),
        ]

        if gradient:
            self.trainer.inference_mode = False

    def energy_forces_config_(
        self,
        *,
        gradient: bool,
        energy_coefficient: float = 1.0,
        energy_loss: LossConfig = MAELossConfig(),
        energy_pooling: Literal["mean", "sum"] = "mean",
        force_coefficient: float = 100.0,
        force_loss: LossConfig = L2MAELossConfig(),
    ):
        self.graph_targets = [
            output_head.GraphScalarTargetConfig(
                name="y",
                loss_coefficient=energy_coefficient,
                loss=energy_loss,
                reduction=energy_pooling,
            ),
        ]
        self.node_targets = [
            output_head.GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=force_coefficient,
                loss=force_loss,
            )
            if gradient
            else output_head.NodeVectorTargetConfig(
                name="force", loss_coefficient=force_coefficient, loss=force_loss
            ),
        ]

        if gradient:
            self.trainer.inference_mode = False

    def energy_forces_stress_config_(
        self,
        *,
        gradient: bool,
        energy_coefficient: float = 1.0,
        energy_loss: LossConfig = MAELossConfig(),
        energy_pooling: Literal["mean", "sum"] = "mean",
        force_coefficient: float = 100.0,
        force_loss: LossConfig = L2MAELossConfig(),
        stress_coefficient: float = 1.0,
        stress_loss: LossConfig = MAELossConfig(),
        stress_pooling: Literal["mean", "sum"] = "mean",
    ):
        self.graph_targets = [
            output_head.GraphScalarTargetConfig(
                name="y",
                loss_coefficient=energy_coefficient,
                loss=energy_loss,
                reduction=energy_pooling,
            ),
            output_head.GradientStressTargetConfig(
                name="stress",
                energy_name="y",
                loss_coefficient=stress_coefficient,
                loss=stress_loss,
                forces=True,
            )
            if gradient
            else output_head.DirectStressTargetConfig(
                name="stress",
                loss_coefficient=stress_coefficient,
                loss=stress_loss,
                reduction=stress_pooling,
            ),
        ]
        self.node_targets = [
            output_head.GradientForcesTargetConfig(
                name="force",
                energy_name="y",
                loss_coefficient=force_coefficient,
                loss=force_loss,
                use_stress_forces=True,
            )
            if gradient
            else output_head.NodeVectorTargetConfig(
                name="force",
                loss_coefficient=force_coefficient,
                loss=force_loss,
            )
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
                if isinstance(target, output_head.GradientForcesTargetConfig)
            ),
            None,
        )

    @override
    def __post_init__(self):
        super().__post_init__()

        if not self.supports_inference_mode():
            assert not self.trainer.inference_mode, "`config.trainer.inference_mode` is True, but the model does not support inference mode."

        if any(
            isinstance(target, output_head.NodeVectorTargetConfig)
            for target in self.node_targets
        ):
            # assert (
            #     self.trainer.inference_mode
            # ), "output_head.NodeVectorTargetConfig requires inference mode to be enabled."
            assert self.backbone.regress_forces, "output_head.NodeVectorTargetConfig requires `backbone.regress_forces` to be True."
            assert self.backbone.direct_forces, "output_head.NodeVectorTargetConfig requires `backbone.direct_forces` to be True."


TConfig = TypeVar("TConfig", bound=EnergyForcesConfigBase, infer_variance=True)


class EnergyForcesModelBase(FinetuneModelBase[TConfig], ABC, Generic[TConfig]):
    @override
    def construct_output_heads(self):
        self.graph_outputs = ll.nn.TypedModuleDict(
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
        self.node_outputs = ll.nn.TypedModuleDict(
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
            if self.config.ignore_graph_generation_errors:
                try:
                    data = self.generate_graphs_transform(data)
                except Exception as e:
                    # If this is a CUDA error, rethrow it
                    if "CUDA" in str(data):
                        raise

                    # Otherwise, log the error and skip the batch
                    log.error(f"Error generating graphs: {e}", exc_info=True)
                    raise SkipBatch()
            else:
                data = self.generate_graphs_transform(data)

            # Run the backbone
            atomic_numbers = data.atomic_numbers - 1
            h = self.embedding(atomic_numbers)  # (N, d_model)
            out = cast(GOCBackboneOutput, self.backbone(data, h=h))

            graph_preds: dict[str, torch.Tensor] = {}
            node_preds: dict[str, torch.Tensor] = {}
            output_head_input = {
                "backbone_output": out,
                "data": data,
                "graph_preds": graph_preds,
                "node_preds": node_preds,
            }
            for target, module in self.graph_outputs.items():
                graph_preds[target] = module(output_head_input)
            preds.update(graph_preds)

            for target, module in self.node_outputs.items():
                node_preds[target] = module(output_head_input)
            preds.update(node_preds)

        return preds

    @abstractmethod
    def generate_graphs_transform(self, data: BaseData) -> BaseData: ...
