from typing import Literal

from typing_extensions import override

from .._config import AtomEmbeddingTableInfo, BackboneConfigBase


class Graphormer3DConfig(BackboneConfigBase):
    """Graphormer-3D model arguments."""

    @override
    def d_atom(self) -> int:
        return self.embed_dim

    @override
    def handles_atom_embedding(self) -> bool:
        return True

    @override
    def atom_embedding_table_info(self) -> AtomEmbeddingTableInfo:
        return {
            "num_embeddings": self.num_elements,
            "embedding_dim": self.embed_dim,
        }

    name: Literal["graphormer3d"] = "graphormer3d"

    layers: int
    """Number of encoder layers."""

    blocks: int
    """Number of blocks."""

    embed_dim: int
    """Encoder embedding dimension."""

    ffn_embed_dim: int
    """Encoder embedding dimension for Feed-Forward Network (FFN)."""

    attention_heads: int
    """Number of encoder attention heads."""

    dropout: float
    """Dropout probability."""

    attention_dropout: float
    """Dropout probability for attention weights."""

    activation_dropout: float
    """Dropout probability after activation in FFN."""

    node_loss_weight: float
    """Loss weight for node fitting."""

    min_node_loss_weight: float
    """Minimum loss weight for node fitting."""

    num_kernel: int
    """Number of kernels."""

    input_dropout: float = 0.0
    """Input dropout probability."""

    energy_output_dropout: float = 0.1
    """Dropout probability for energy output."""

    eng_loss_weight: float = 1.0
    """Loss weight for energy prediction."""

    num_elements: int = 120
    """Number of elements."""

    @property
    def activation(self):
        return "gelu"

    def graphormer_base_(self):
        """Default base architecture configuration."""
        self.layers = 12
        self.blocks = 4
        self.embed_dim = 768
        self.ffn_embed_dim = 3072
        self.attention_heads = 32
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.1
        self.node_loss_weight = 15.0
        self.min_node_loss_weight = 1.0
        self.num_kernel = 128
        self.input_dropout = 0.0
        self.eng_loss_weight = 1.0

    def graphormer_large_(self):
        self.graphormer_base_()

        self.layers = 24
        self.embed_dim = 1024
        self.attention_heads = 32

    def graphormer_extra_large_(self):
        self.graphormer_large_()

        self.layers = 48
        self.embed_dim = 2048
        self.attention_heads = 32

    def create_model(self):
        from .model import Graphormer3D

        return Graphormer3D(self)
