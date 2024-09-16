from typing import Literal
import nshtrainer.ll as ll
from typing_extensions import override
from .._config import AtomEmbeddingTableInfo, BackboneConfigBase


class BackboneConfig(BackboneConfigBase):
    @override
    def d_atom(self) -> int:
        return self.hidden_dim

    @override
    def handles_atom_embedding(self) -> bool:
        return True

    @override
    def atom_embedding_table_info(self) -> AtomEmbeddingTableInfo:
        return {
            "num_embeddings": self.max_z,
            "embedding_dim": self.hidden_dim,
        }

    name: Literal["m3gnet"] = "m3gnet"
    num_targets: int
    num_blocks: int
    hidden_dim: int
    max_l: int
    max_n: int
    cutoff: float
    threebody_cutoff: float
    max_z: int
    emb_size_atom: int = 0
    emb_size_edge: int = 0
    activation: str = "swish"
    regress_forces: bool = True
    regress_energy: bool = True
    direct_forces: bool = False
    use_pbc: bool = True
    
    @classmethod
    def base(cls):
        """
        The original configuration for M3GNet
        https://www.nature.com/articles/s43588-022-00349-3
        """
        config = cls(
            num_targets=1,
            num_blocks=3,
            hidden_dim=64,
            max_l=3,
            max_n=3,
            cutoff=5.0,
            threebody_cutoff=4.0,
            max_z=120,
        )
        config.emb_size_atom = config.hidden_dim
        config.emb_size_edge = config.hidden_dim
        return config
    
    @classmethod
    def large(cls):
        """
        The increased-parameter M3GNet, used in MatterSim for 3M dataset
        https://arxiv.org/pdf/2405.04967, Page 25
        """
        config = cls(
            num_targets=1,
            num_blocks=4,
            hidden_dim=256,
            max_l=4,
            max_n=4,
            cutoff=5.0,
            threebody_cutoff=4.0,
            max_z=120,
        )
        config.emb_size_atom = config.hidden_dim
        config.emb_size_edge = config.hidden_dim
        return config
    
    