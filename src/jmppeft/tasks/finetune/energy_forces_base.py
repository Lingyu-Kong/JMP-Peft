from abc import ABC, abstractmethod
from contextlib import ExitStack
from logging import getLogger
from typing import Generic, Literal, cast

import ll
import torch
import torch.nn as nn
from ll.nn import TypedModuleDict
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar, override

from ...models.gemnet.backbone import GOCBackboneOutput
from ...modules.relaxer import Relaxer, RelaxerConfig
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


class RelaxationConfig(ll.TypedConfig):
    validation: RelaxerConfig | None = None
    """Relaxer configuration for validation. If None, relaxer is disabled for validation."""

    test: RelaxerConfig | None = None
    """Relaxer configuration for testing. If None, relaxer is disabled for testing."""


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

    relaxation: RelaxationConfig = RelaxationConfig()
    """Relaxation configuration for validation and testing."""

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

    def relaxation_y_relaxed(self, batch: BaseData):
        if (y_relaxed := getattr(batch, "y_relaxed", None)) is None:
            raise AttributeError(
                "Batch does not have `y_relaxed` attribute. Please either set it or override `relaxation_y_relaxed`."
            )

        return y_relaxed

    def _relaxer_forward(self, batch: Batch):
        out = self(batch)

    @override
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()

        self.val_relaxer = None
        if self.config.relaxation.validation is not None:
            self.val_relaxer = Relaxer(
                self.config.relaxation.validation,
                self._relaxer_forward,
                self.collate_fn,
                self.device,
            )

    @override
    def validation_step(self, batch: BaseData, batch_idx: int):
        if self.config.relaxation.validation is None:
            return super().validation_step(batch, batch_idx)

        assert self.val_relaxer is not None, "Relaxer must be initialized"
        with self.log_context(prefix=f"val/{self.metric_prefix()}/"):
            self.val_relaxer.step(
                batch,
                self.relaxation_y_relaxed(batch),
            )

    @override
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        if self.config.relaxation.validation is not None:
            assert self.val_relaxer is not None, "Relaxer must be initialized"

            # Compute the relaxation metrics and report them.
            with self.log_context(prefix=f"val/relax/{self.metric_prefix()}/"):
                self.log_dict(self.val_relaxer.compute_metrics(self), on_epoch=True)

    @override
    def on_test_epoch_start(self):
        super().on_test_epoch_start()

        self.test_relaxer = None
        if self.config.relaxation.test is not None:
            self.test_relaxer = Relaxer(
                self.config.relaxation.test,
                self,
                self.collate_fn,
                self.device,
            )

    @override
    def test_step(self, batch: BaseData, batch_idx: int):
        if self.config.relaxation.test is None:
            return super().test_step(batch, batch_idx)

        assert self.test_relaxer is not None, "Relaxer must be initialized"
        with self.log_context(prefix=f"test/{self.metric_prefix()}/"):
            self.test_relaxer.step(
                batch,
                self.relaxation_y_relaxed(batch),
            )

    @override
    def on_test_epoch_end(self):
        super().on_test_epoch_end()

        if self.config.relaxation.test is not None:
            assert self.test_relaxer is not None, "Relaxer must be initialized"

            # Compute the relaxation metrics and report them.
            with self.log_context(prefix=f"test/relax/{self.metric_prefix()}/"):
                self.log_dict(self.test_relaxer.compute_metrics(self), on_epoch=True)
