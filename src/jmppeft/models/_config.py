from abc import ABC, abstractmethod
from typing import TypedDict

import nshtrainer.ll as ll


class AtomEmbeddingTableInfo(TypedDict):
    num_embeddings: int
    embedding_dim: int


class BackboneConfigBase(ll.TypedConfig, ABC):
    @abstractmethod
    def d_atom(self) -> int: ...

    @abstractmethod
    def handles_atom_embedding(self) -> bool: ...

    @abstractmethod
    def atom_embedding_table_info(self) -> AtomEmbeddingTableInfo: ...
