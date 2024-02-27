from typing import Literal, TypeAlias, final

import torch
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .energy_forces_base import EnergyForcesConfigBase, EnergyForcesModelBase

SPICEDataset: TypeAlias = Literal["solvated_amino_acids", "dipeptides"]


class SPICEConfig(EnergyForcesConfigBase):
    dataset: SPICEDataset


@final
class SPICEModel(EnergyForcesModelBase[SPICEConfig]):
    @classmethod
    @override
    def config_cls(cls):
        return SPICEConfig

    @override
    def metric_prefix(self) -> str:
        return f"spice/{self.config.dataset}"

    @override
    def process_aint_graph(self, aint_graph: Graph):
        return aint_graph

    @override
    def generate_graphs_transform(self, data: BaseData):
        return self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
            pbc=False,
        )

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        data.y = data.pop("formation_energy").view(-1).float()
        data.atomic_numbers = data.pop("atomic_numbers").long()
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.cell = (torch.eye(3) * 1000.0).unsqueeze(dim=0)
        return data
