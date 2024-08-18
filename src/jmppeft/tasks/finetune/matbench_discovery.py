import torch
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .energy_forces_base import EnergyForcesConfigBase, EnergyForcesModelBase


class MatbenchDiscoveryConfig(EnergyForcesConfigBase):
    max_neighbors: MaxNeighbors = MaxNeighbors(
        main=20,
        aeaint=20,
        aint=1000,
        qint=8,
    )
    cutoffs: Cutoffs = Cutoffs.from_constant(12.0)


class MatbenchDiscoveryModel(EnergyForcesModelBase[MatbenchDiscoveryConfig]):
    @classmethod
    @override
    def config_cls(cls):
        return MatbenchDiscoveryConfig

    @override
    def metric_prefix(self) -> str:
        return "matbench_discovery"

    @override
    def process_aint_graph(self, graph: Graph, *, training: bool):
        graph = super().process_aint_graph(graph, training=training)
        return graph

    @override
    def generate_graphs_transform(self, data: BaseData, training: bool):
        # Generate graphs
        data = self.generate_graphs(
            data,
            cutoffs=self.config.cutoffs,
            max_neighbors=self.config.max_neighbors,
            pbc=True,
            training=training,
        )
        return data

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        if getattr(data, "y", None) is not None:
            if not torch.is_tensor(data.y):
                data.y = torch.tensor(data.y)
            data.y = data.y.view(-1)

        data.atomic_numbers = data.atomic_numbers.long()
        assert data.num_nodes is not None
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.pos = data.pos.float()

        return data
