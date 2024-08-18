import copy
from typing import Literal, TypeAlias

import torch
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .energy_forces_base import EnergyForcesConfigBase, EnergyForcesModelBase

RMD17Molecule: TypeAlias = Literal[
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
]


class RMD17Config(EnergyForcesConfigBase):
    molecule: RMD17Molecule

    cutoff: float = 7.0
    max_neighbors: int = 100


class RMD17Model(EnergyForcesModelBase[RMD17Config]):
    @classmethod
    @override
    def config_cls(cls):
        return RMD17Config

    @override
    def metric_prefix(self) -> str:
        return f"md17/{self.config.molecule}"

    @override
    def generate_graphs_transform(self, data: BaseData, training: bool):
        return self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(self.config.cutoff),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(
                self.config.max_neighbors
            ),
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

        data = copy.deepcopy(data)

        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y, dtype=torch.float)
        data.y = data.y.view(-1).float()
        if hasattr(data, "z"):
            data.atomic_numbers = data.pop("z")
        data.atomic_numbers = data.atomic_numbers.long()
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        # data.cell = (torch.eye(3) * 1000.0).unsqueeze(dim=0)
        # data.cell = torch.tensor(
        # [
        #     [
        #         [8, 0.0000, -0.0000],
        #         [-0.0000, 12.7363, -0.0000],
        #         [0.0000, 0.0000, 47.3956],
        #     ]
        # ]
        # )
        return data
