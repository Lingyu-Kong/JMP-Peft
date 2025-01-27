from typing import ClassVar, Literal, TypeAlias, final

import torch
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ...utils.goc_graph import Cutoffs, Graph, MaxNeighbors
from .base import FinetuneConfigBase, FinetuneModelBase
from .output_head import GraphScalarTargetConfig

MatbenchDataset: TypeAlias = Literal[
    "jdft2d",
    "phonons",
    "dielectric",
    "log_gvrh",
    "log_kvrh",
    "perovskites",
    "mp_gap",
    "mp_e_form",
    "mp_is_metal",
]


class MatbenchConfig(FinetuneConfigBase):
    ALL_MATBENCH_DATASETS: ClassVar[list[MatbenchDataset]] = [
        "jdft2d",
        "phonons",
        "dielectric",
        "log_gvrh",
        "log_kvrh",
        "perovskites",
        "mp_gap",
        "mp_e_form",
        "mp_is_metal",
    ]

    dataset: MatbenchDataset

    fold: int = 0
    mp_e_form_dev: bool = True

    conditional_max_neighbors: bool = False

    def default_target_(self):
        self.graph_targets = [GraphScalarTargetConfig(name="y", loss_coefficient=1.0)]
        self.node_targets = []

    @override
    def __post_init__(self):
        super().__post_init__()

        assert (
            self.dataset in self.ALL_MATBENCH_DATASETS
        ), f"{self.dataset=} is not valid"


@final
class MatbenchModel(FinetuneModelBase[MatbenchConfig]):
    @classmethod
    @override
    def config_cls(cls):
        return MatbenchConfig

    @override
    def metric_prefix(self) -> str:
        return f"matbench/{self.config.dataset}"

    @override
    def process_aint_graph(self, graph: Graph, *, training: bool):
        graph = super().process_aint_graph(graph, training=training)
        return graph

    @override
    def forward(self, data: BaseData):
        # Generate graphs
        max_neighbors = 30
        if self.config.conditional_max_neighbors:
            if data.natoms > 300:
                max_neighbors = 5
            elif data.natoms > 200:
                max_neighbors = 10
            else:
                max_neighbors = 30

        data = self.generate_graphs(
            data,
            cutoffs=Cutoffs.from_constant(12.0),
            max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
            pbc=True,
            training=self.training,
        )

        return super().forward(data)

    @override
    def data_transform(self, data: BaseData):
        data = super().data_transform(data)

        if not torch.is_tensor(data.y):
            data.y = torch.tensor(data.y)
        data.y = data.y.view(-1)

        if self.config.dataset == "mp_is_metal":
            data.y = data.y.bool()

        data.atomic_numbers = data.atomic_numbers.long()
        assert data.num_nodes is not None
        data.natoms = data.num_nodes

        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()

        data.fixed = torch.zeros(data.natoms, dtype=torch.bool)

        data.pos = data.pos.float()

        return data
