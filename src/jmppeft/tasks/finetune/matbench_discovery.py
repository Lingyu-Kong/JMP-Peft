import torch
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, MaxNeighbors
from .energy_forces_base import EnergyForcesConfigBase, EnergyForcesModelBase


class MatbenchDiscoveryConfig(EnergyForcesConfigBase):
    conditional_max_neighbors: bool = True
    """
    If True, the number of neighbors will be set based on the number of atoms in the system.

    This is necessary when running the model on GPUs with limited memory, as the number of neighbors
    is a major factor in the memory consumption of the model.
    """


class MatbenchDiscoveryModel(EnergyForcesModelBase[MatbenchDiscoveryConfig]):
    @classmethod
    @override
    def config_cls(cls):
        return MatbenchDiscoveryConfig

    @override
    def metric_prefix(self) -> str:
        return "matbench_discovery"

    @override
    def generate_graphs_transform(self, data: BaseData):
        # Generate graphs
        max_neighbors = 30
        if self.config.conditional_max_neighbors:
            if (data.natoms > 300).any():
                max_neighbors = 5
            elif (data.natoms > 200).any():
                max_neighbors = 10
            elif (data.natoms > 100).any():
                max_neighbors = 20
            else:
                max_neighbors = 30

        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
            pbc=True,
        )
        return data

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

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
