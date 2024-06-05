from typing import ClassVar, Literal, TypeAlias, final

import torch
from ase.data import atomic_masses
from einops import rearrange
from jmppeft.modules.torch_scatter_polyfill import scatter
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .base import FinetuneConfigBase, FinetuneModelBase, OutputHeadInput
from .output_head import GraphScalarOutputHead, GraphScalarTargetConfig

QM9Target: TypeAlias = Literal[
    "mu",  # dipole_moment
    "alpha",  # isotropic_polarizability
    "eps_HOMO",  # hOMO
    "eps_LUMO",  # lumo
    "delta_eps",  # homo_lumo_gap
    "R_2_Abs",  # electronicspatial_extent
    "ZPVE",  # zpve
    "U_0",  # energy_U0
    "U",  # energy_U
    "H",  # enthalpy_H
    "G",  # free_energy
    "c_v",  # heat_capacity
    "U_0_ATOM",  # atomization_energy_U0
    "U_ATOM",  # atomization_energy_U
    "H_ATOM",  # atomization_enthalpy_H
    "G_ATOM",  # atomization_free_energy
    "A",  # rotational_constant_A
    "B",  # rotational_constant_B
    "C",  # rotational_constant_C
]


class GraphSpatialExtentScalarTargetConfig(GraphScalarTargetConfig):
    kind: Literal["spatial_extent_scalar"] = "spatial_extent_scalar"  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return GraphSpatialExtentScalarOutputHead(
            self,
            output_config,
            d_model_node,
            activation_cls,
        )


class GraphSpatialExtentScalarOutputHead(GraphScalarOutputHead):
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

        # Run the MLP
        x = self.out_mlp(backbone_output["energy"])  # (n_atoms, 1)
        if scale is not None:
            x = x * scale
        if shift is not None:
            x = x + shift

        batch_size = int(torch.max(data.batch).item() + 1)

        # Get the center of mass
        masses = self.atomic_masses()[data.atomic_numbers]  # n
        center_of_mass = scatter(
            masses.unsqueeze(-1) * data.pos,  # n 3
            data.batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        ) / scatter(
            masses.unsqueeze(-1),
            data.batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )  # b 3

        # Get the squared norm of each position vector
        pos_norm_sq = (
            torch.linalg.vector_norm(
                data.pos - center_of_mass[data.batch],
                dim=-1,
                keepdim=True,
                ord=2,
            )
            ** 2
        )  # n 1
        x = x * pos_norm_sq  # n 1

        # Apply the reduction
        x = scatter(
            x,
            data.batch,
            dim=0,
            dim_size=batch_size,
            reduce=self.reduction,
        )  # (bsz, 1)

        x = rearrange(x, "b 1 -> b")
        return x


class QM9Config(FinetuneConfigBase):
    QM9_TARGETS: ClassVar[list[QM9Target]] = [
        "mu",
        "alpha",
        "eps_HOMO",
        "eps_LUMO",
        "delta_eps",
        "R_2_Abs",
        "ZPVE",
        "U_0",
        "U",
        "H",
        "G",
        "c_v",
        "U_0_ATOM",
        "U_ATOM",
        "H_ATOM",
        "G_ATOM",
        "A",
        "B",
        "C",
    ]

    max_neighbors: int = 30

    @override
    def __post_init__(self):
        super().__post_init__()

        for target in self.targets:
            assert (
                target.name in self.QM9_TARGETS
            ), f"{target=} is not a valid QM9 target"


@final
class QM9Model(FinetuneModelBase[QM9Config]):
    atomic_masses: torch.Tensor

    @override
    def __init__(self, hparams: QM9Config):
        super().__init__(hparams)

        self.register_buffer(
            "atomic_masses",
            torch.from_numpy(atomic_masses).float(),
            persistent=False,
        )

    @classmethod
    @override
    def config_cls(cls):
        return QM9Config

    @override
    def metric_prefix(self) -> str:
        return "qm9"

    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    @override
    def forward(self, data: BaseData):
        # Generate graphs on the GPU
        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(8.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
        )

        return super().forward(data)

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)
        return data
