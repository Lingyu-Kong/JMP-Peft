from typing import Literal, TypeAlias, final

import torch
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .energy_forces_base import EnergyForcesConfigBase, EnergyForcesModelBase

MD22Molecule: TypeAlias = Literal[
    "Ac-Ala3-NHMe",
    "DHA",
    "stachyose",
    "AT-AT",
    "AT-AT-CG-CG",
    "buckyball-catcher",
    "double-walled_nanotube",
]


class MD22Config(EnergyForcesConfigBase):
    molecule: MD22Molecule


@final
class MD22Model(EnergyForcesModelBase[MD22Config]):
    @classmethod
    @override
    def config_cls(cls):
        return MD22Config

    @override
    def metric_prefix(self) -> str:
        return f"md22/{self.config.molecule}"

    @override
    def generate_graphs_transform(self, data: BaseData, training: bool):
        return self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
            training=training,
        )

    @override
    def process_aint_graph(self, graph: Graph, *, training: bool):
        graph = super().process_aint_graph(graph, training=training)
        return graph

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        data.y = data.pop("y").view(-1).float()
        data.atomic_numbers = data.pop("atomic_numbers").long()
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.cell = (torch.eye(3) * 1000.0).unsqueeze(dim=0)
        return data
