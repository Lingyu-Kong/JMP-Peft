from abc import ABC, abstractmethod
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
from typing_extensions import TypeVar, assert_never, override

from ...models.gemnet.backbone import GOCBackboneOutput
from ...modules.relaxer import GraphConverter, Potential, Relaxer
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


class RelaxerConfig(ll.TypedConfig):
    enabled: bool = False

    energy_key: str = "y"
    """Key for the energy in the graph data."""

    force_key: str = "force"
    """Key for the forces in the node data."""

    stress_key: str | None = None
    """Key for the stress in the graph data (or `None` if stress is not computed)."""

    optimizer: Literal[
        "FIRE",
        "BFGS",
        "LBFGS",
        "LBFGSLineSearch",
        "MDMin",
        "SciPyFminCG",
        "SciPyFminBFGS",
        "BFGSLineSearch",
    ] = "FIRE"
    """Optimizer to use for relaxation."""

    relax_cell: bool = False
    """Whether to relax the cell."""

    stress_weight: float = 0.01
    """Weight for the stress loss."""

    @property
    def optimizer_cls(self):
        match self.optimizer:
            case "FIRE":
                from ase.optimize.fire import FIRE

                return FIRE
            case "BFGS":
                from ase.optimize.bfgs import BFGS

                return BFGS
            case "LBFGS":
                from ase.optimize.lbfgs import LBFGS

                return LBFGS
            case "LBFGSLineSearch":
                from ase.optimize.lbfgs import LBFGSLineSearch

                return LBFGSLineSearch
            case "MDMin":
                from ase.optimize.mdmin import MDMin

                return MDMin
            case "SciPyFminCG":
                from ase.optimize.sciopt import SciPyFminCG

                return SciPyFminCG
            case "SciPyFminBFGS":
                from ase.optimize.sciopt import SciPyFminBFGS

                return SciPyFminBFGS
            case "BFGSLineSearch":
                from ase.optimize.bfgslinesearch import BFGSLineSearch

                return BFGSLineSearch
            case _:
                assert_never(self.optimizer)

    def __bool__(self):
        return self.enabled


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

    def relaxer_graph_converter(self):
        return RelaxerGraphConverter(self)

    def relaxer_potential(self):
        return RelaxerPotential(self)

    def relaxer(self):
        assert self.config.relaxer, "Relaxer is not enabled in the config."

        return Relaxer(
            potential=self.relaxer_potential(),
            graph_converter=self.relaxer_graph_converter(),
            optimizer_cls=self.config.relaxer.optimizer_cls,
            relax_cell=self.config.relaxer.relax_cell,
            stress_weight=self.config.relaxer.stress_weight,
        )


@dataclass
class RelaxerGraphConverter(GraphConverter, Generic[TConfig]):
    model: EnergyForcesModelBase[TConfig]

    @property
    def config(self):
        return self.model.config

    @override
    def __call__(self, atoms):
        assert self.config.relaxer, "Relaxer is not enabled in the config."

        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float)
        tags = (2 * torch.ones_like(atomic_numbers)).long()
        fixed = torch.zeros_like(tags, dtype=torch.bool)
        natoms = len(atoms)

        data = Data.from_dict(
            {
                "atomic_numbers": atomic_numbers,
                "pos": pos,
                "cell": cell,
                "tags": tags,
                "fixed": fixed,
                "natoms": natoms,
            }
        )
        batch = self.model.collate_fn([data])

        return batch


@dataclass
class RelaxerPotential(Potential, Generic[TConfig]):
    model: EnergyForcesModelBase[TConfig]

    @property
    def config(self):
        return self.model.config

    @override
    def get_efs_tensor(self, graph: Batch, include_stresses: bool = True):
        assert (c := self.config.relaxer), "Relaxer is not enabled in the config."

        # First, move the graph to the model's device
        graph = graph.to(self.model.device)

        # Compute the energy and forces
        out: dict[str, torch.Tensor] = self.model(graph)
        energy = out[c.energy_key].detach().float().cpu().numpy()
        forces = out[c.force_key].detach().float().cpu().numpy()

        if include_stresses:
            assert (
                c.stress_key is not None
            ), "Stress key is not set in the relaxer config."

            stress = out[c.stress_key].detach().float().cpu().numpy()
            return energy, forces, stress
        else:
            return energy, forces
