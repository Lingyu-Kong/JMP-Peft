from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from logging import getLogger
from typing import TYPE_CHECKING, Literal

import ll
import ll.typecheck as tc
import torch
from einops import pack
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter
from typing_extensions import override

from .config import TorchMDNetBackboneConfig

if TYPE_CHECKING:
    from ...tasks.pretrain.module import TaskConfig
    from .backbone import TorchMDNetBackboneOutput

log = getLogger(__name__)


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        activation_cls: Callable[[], nn.Module],
        intermediate_channels: int | None = None,
        scalar_activation: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            activation_cls(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = activation_cls() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    @override
    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        # detach zero-entries to avoid NaN gradients during force loss backpropagation
        vec1 = torch.zeros(
            vec1_buffer.size(0),
            vec1_buffer.size(2),
            device=vec1_buffer.device,
            dtype=torch.float,
        )
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        if not mask.all():
            log.warn(
                (
                    f"Skipping gradients for {(~mask).sum()} atoms due to vector features being zero. "
                    "This is likely due to atoms being outside the cutoff radius of any other atom. "
                    "These atoms will not interact with any other atom unless you change the cutoff."
                )
            )
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)

        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class _OutputModel(nn.Module, metaclass=ABCMeta):
    """Base class for output models.

    Derive this class to make custom output models.
    As an example, have a look at the :py:mod:`torchmdnet.output_modules.Scalar` output model.
    """

    def __init__(self, allow_prior_model, reduce_op):
        super().__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op
        self.dim_size = 0

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x: torch.Tensor, v: torch.Tensor):
        return

    def reduce(self, x: torch.Tensor, batch: torch.Tensor):
        is_capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
        if not x.is_cuda or not is_capturing:
            self.dim_size = int(batch.max().item() + 1)
        if is_capturing:
            assert (
                self.dim_size > 0
            ), "Warming up is needed before capturing the model into a CUDA graph"
            log.warn(
                "CUDA graph capture will lock the batch to the current number of samples ({}). Changing this will result in a crash".format(
                    self.dim_size
                )
            )
        return scatter(x, batch, dim=0, dim_size=self.dim_size, reduce=self.reduce_op)

    def post_reduce(self, x: torch.Tensor):
        return x


class _EquivariantScalar(_OutputModel):
    def __init__(
        self,
        hidden_channels: int,
        activation_cls: Callable[[], nn.Module],
        allow_prior_model: bool = True,
        reduce_op: Literal["sum", "mean"] = "sum",
    ):
        super().__init__(allow_prior_model=allow_prior_model, reduce_op=reduce_op)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation_cls=activation_cls,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2, 1, activation_cls=activation_cls
                ),
            ]
        )

        self.reset_parameters()

    @override
    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    @override
    def pre_reduce(self, x: torch.Tensor, v: torch.Tensor):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0


class GatedEquivariantBlockPredictionHeadSingle(_EquivariantScalar):
    def __init__(self, backbone_config: TorchMDNetBackboneConfig):
        super().__init__(
            backbone_config.hidden_channels,
            backbone_config.activation_cls,
            allow_prior_model=False,
            reduce_op="sum",
        )

        self.out_energy = nn.Sequential(
            nn.Linear(
                backbone_config.hidden_channels,
                backbone_config.outhead_hidden_size,
            ),
            backbone_config.activation_cls(),
            nn.Linear(backbone_config.outhead_hidden_size, 1, bias=False),
        )

    @override
    def pre_reduce(self, x: torch.Tensor, v: torch.Tensor):
        for layer in self.output_network:
            x, v = layer(x, v)

        return x, v

    @override
    def forward(self, x: torch.Tensor, v: torch.Tensor, batch: torch.Tensor):
        x, v = self.pre_reduce(x, v)

        energy = self.reduce(x, batch)

        forces = v
        if forces.shape[-1] == 1:
            forces = forces.squeeze(-1)

        return energy, forces


class Output(nn.Module):
    @override
    def __init__(
        self,
        task_configs: "list[TaskConfig]",
        backbone_config: TorchMDNetBackboneConfig,
    ):
        super().__init__()

        # Make sure all node task energy reductions are sum
        for task in task_configs:
            assert (
                task.node_energy_reduction == "sum"
            ), "Only sum reduction is supported"

        self.task_configs = task_configs
        self.backbone_config = backbone_config

        self.per_task_output = ll.nn.TypedModuleList(
            [
                GatedEquivariantBlockPredictionHeadSingle(backbone_config)
                for _ in task_configs
            ]
        )

    @override
    def forward(self, data: Data, backbone_out: "TorchMDNetBackboneOutput"):
        energy_list: list[tc.Float[torch.Tensor, "b"]] = []
        forces_list: list[tc.Float[torch.Tensor, "n 3"]] = []

        for output_head, task in zip(self.per_task_output, self.task_configs):
            energy, forces = output_head(
                backbone_out["x"], backbone_out["v"], data.batch
            )

            energy_list.append(energy)
            forces_list.append(forces)

        E, _ = pack(energy_list, "bsz *")
        F, _ = pack(forces_list, "n_atoms p *")

        return E, F
