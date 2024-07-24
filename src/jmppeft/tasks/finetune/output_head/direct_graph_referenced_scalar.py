from abc import ABC, abstractmethod
from logging import getLogger
from typing import Annotated, Literal, TypeAlias

import nshtrainer.ll as ll
import nshutils.typecheck as tc
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from jmppeft.modules.torch_scatter_polyfill import scatter
from nshtrainer.ll.nn import MLP
from torch_geometric.data.data import BaseData
from typing_extensions import TypedDict, override

from ....models.gemnet.backbone import GOCBackboneOutput
from ....modules.loss import LossConfig, MAELossConfig
from ...config import OutputConfig
from ._base import BaseTargetConfig

log = getLogger(__name__)


class ReferenceInitializationConfigBase(ll.TypedConfig, ABC):
    @abstractmethod
    def initialize(
        self, max_atomic_number: int
    ) -> tc.Float[torch.Tensor, "max_atomic_number 1"]: ...


class ZerosReferenceInitializationConfig(ReferenceInitializationConfigBase):
    name: Literal["zeros"] = "zeros"

    @override
    def initialize(
        self, max_atomic_number: int
    ) -> tc.Float[torch.Tensor, "max_atomic_number 1"]:
        return torch.zeros((max_atomic_number + 1, 1))


class RandomReferenceInitializationConfig(ReferenceInitializationConfigBase):
    name: Literal["random"] = "random"

    @override
    def initialize(
        self, max_atomic_number: int
    ) -> tc.Float[torch.Tensor, "max_atomic_number 1"]:
        return torch.randn((max_atomic_number + 1, 1))


class MPElementalReferenceInitializationConfig(ReferenceInitializationConfigBase):
    name: Literal["mp_elemental"] = "mp_elemental"

    @override
    def initialize(
        self, max_atomic_number: int
    ) -> tc.Float[torch.Tensor, "max_atomic_number 1"]:
        from matbench_discovery.data import DATA_FILES
        from pymatgen.core import Element
        from pymatgen.entries.computed_entries import ComputedEntry

        references = torch.zeros((max_atomic_number + 1, 1))

        for elem_str, entry in (
            pd.read_json(DATA_FILES.mp_elemental_ref_entries, typ="series")
            .map(ComputedEntry.from_dict)
            .to_dict()
            .items()
        ):
            references[Element(elem_str).Z] = round(entry.energy_per_atom, 4)

        return references


ReferenceInitializationConfig: TypeAlias = Annotated[
    ZerosReferenceInitializationConfig
    | RandomReferenceInitializationConfig
    | MPElementalReferenceInitializationConfig,
    ll.Field(discriminator="name"),
]


class ReferencedScalarTargetConfig(BaseTargetConfig):
    kind: Literal["referenced_scalar"] = "referenced_scalar"

    reduction: Literal["mean", "sum"] = "sum"
    """The reduction to use for the output."""

    loss: LossConfig = MAELossConfig()
    """The loss configuration for the target."""

    max_atomic_number: int
    """The max atomic number in the dataset."""

    initialization: ReferenceInitializationConfig
    """The initialization configuration for the references."""

    trainable_references: bool = True
    """Whether to train the references. If False, the references must be initialized."""

    @override
    def is_classification(self) -> bool:
        return False

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return ReferencedScalarOutputHead(
            self,
            output_config,
            d_model_node,
            activation_cls,
        )


class OutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput


class ReferencedScalarOutputHead(nn.Module):
    @override
    def __init__(
        self,
        target_config: ReferencedScalarTargetConfig,
        output_config: OutputConfig,
        d_model: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        self.target_config = target_config
        self.output_config = output_config
        self.d_model = d_model
        self.out_mlp = MLP(
            ([self.d_model] * self.output_config.num_mlps) + [1],
            activation=activation_cls,
        )

        self.references = nn.Embedding.from_pretrained(
            target_config.initialization.initialize(target_config.max_atomic_number),
            freeze=not target_config.trainable_references,
        )
        assert (
            self.references.weight.shape == (target_config.max_atomic_number + 1, 1)
        ), f"{self.references.weight.shape=} != {(target_config.max_atomic_number + 1, 1)}"

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        per_atom_energies = self.out_mlp(backbone_output["energy"])  # (n_atoms, 1)
        per_atom_references = self.references(data.atomic_numbers)  # (n_atoms, 1)
        per_atom_energies = per_atom_energies + per_atom_references  # (n_atoms, 1)

        per_system_energies = scatter(
            per_atom_energies,
            data.batch,
            dim=0,
            dim_size=data.num_graphs,
            reduce=self.target_config.reduction,
        )  # (bsz, 1)
        per_system_energies = rearrange(per_system_energies, "b 1 -> b")
        return per_system_energies
