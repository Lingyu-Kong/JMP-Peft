from typing import TYPE_CHECKING, Literal

from typing_extensions import override

from .._config import AtomEmbeddingTableInfo, BackboneConfigBase

if TYPE_CHECKING:
    from ...tasks.pretrain.module import TaskConfig


class TorchMDNetBackboneConfig(BackboneConfigBase):
    name: Literal["torchmdnet"] = "torchmdnet"

    @override
    def d_atom(self) -> int:
        return self.hidden_channels

    @override
    def handles_atom_embedding(self) -> bool:
        return True

    @override
    def atom_embedding_table_info(self) -> AtomEmbeddingTableInfo:
        return {
            "num_embeddings": self.max_z,
            "embedding_dim": self.hidden_channels,
        }

    hidden_channels: int = 128
    num_layers: int = 8
    num_rbf: int = 64
    rbf_type: str = "expnorm"
    trainable_rbf: bool = True
    activation: str = "silu"
    neighbor_embedding: bool = True
    num_heads: int = 8
    distance_influence: str = "both"
    cutoff_lower: float = 0.0
    cutoff_upper: float = 5.0
    max_z: int = 100
    max_num_neighbors: int = 32
    output_layernorm_trainable: bool = True
    layernorm_trainable: bool = True
    outhead_hidden_size: int = 128

    @property
    def activation_cls(self):
        match self.activation:
            case "silu":
                from torch.nn import SiLU

                return SiLU
            case _:
                raise ValueError(f"Unknown activation: {self.activation}")

    def apply_1_5M_param_config_(self):
        self.num_layers = 6
        self.num_rbf = 32
        self.hidden_channels = 128
        return self

    def create_backbone(self):
        from .backbone import TorchMD_ET

        return TorchMD_ET(
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            num_rbf=self.num_rbf,
            rbf_type=self.rbf_type,
            trainable_rbf=self.trainable_rbf,
            activation=self.activation,
            neighbor_embedding=self.neighbor_embedding,
            num_heads=self.num_heads,
            distance_influence=self.distance_influence,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            max_z=self.max_z,
            max_num_neighbors=self.max_num_neighbors,
            output_layernorm_trainable=self.output_layernorm_trainable,
            layernorm_trainable=self.layernorm_trainable,
        )

    def create_output_head(self, tasks: "list[TaskConfig]"):
        from .output import Output

        return Output(tasks, self)
