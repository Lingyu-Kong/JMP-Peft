from functools import partial
from typing import TypedDict

from ll import TypedConfig


class _LoraKwargs(TypedDict):
    r: int
    lora_alpha: int
    lora_dropout: float
    merge_weights: bool


class LoraConfig(TypedConfig):
    enabled: bool = True

    r: int
    alpha: int = 1
    dropout: float = 0.0
    merge_weights: bool = False

    def as_kwargs(self):
        return _LoraKwargs(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            merge_weights=self.merge_weights,
        )

    def __bool__(self):
        return self.enabled

    @staticmethod
    def gemnet_basis_embedding_cls(config: "LoraConfig | None"):
        if config is None:
            from ...models.gemnet.layers.efficient import BasisEmbedding

            return BasisEmbedding
        else:
            from ...models.gemnet.layers.lora_efficient import LoRABasisEmbedding

            return partial(LoRABasisEmbedding, lora=config)
