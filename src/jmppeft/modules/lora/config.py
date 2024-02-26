from functools import partial
from typing import Any, cast, TypedDict

from ll import TypedConfig


class _LoraKwargs(TypedDict):
    r: int
    lora_alpha: int
    lora_dropout: float
    merge_weights: bool


class LoraConfig(TypedConfig):
    enabled: bool = True

    # Base Settings
    r: int
    alpha: int = 1
    dropout: float = 0.0
    merge_weights: bool = False

    # Specialized Settings for Children
    _children_config: dict[str, Any] = {}

    def as_kwargs(self):
        return _LoraKwargs(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            merge_weights=self.merge_weights,
        )

    def __bool__(self):
        return self.enabled and self.r > 0

    @property
    def gemnet_basis_embedding_cls(self):
        if not self:
            from ...models.gemnet.layers.efficient import BasisEmbedding

            return BasisEmbedding
        else:
            from ...models.gemnet.layers.lora_efficient import LoRABasisEmbedding

            return partial(LoRABasisEmbedding, lora=self)

    def __call__(self, module: str):
        update: dict[str, Any] = {}

        # Recursively update kwargs for children
        for part in module.split("."):
            update.update(self._children_config.get(part, {}))

        # Update kwargs for the module
        return cast(LoraConfig, self.pydantic_model().model_copy(update=update))

    @classmethod
    def disabled(cls):
        return cls(enabled=False, r=0)
