from functools import partial
from typing import Any, ClassVar, TypeAlias, TypedDict

from ll import TypedConfig


class _LoraKwargs(TypedDict):
    r: int
    lora_alpha: int
    lora_dropout: float
    merge_weights: bool


_PathTree: TypeAlias = dict[str, "_PathTree"]


class LoraConfig(TypedConfig):
    enabled: bool = True

    # Base Settings
    r: int = 0
    alpha: int = 1
    dropout: float = 0.0
    merge_weights: bool = False

    # Specialized Settings for Children
    _path: list[str] = []
    _children_config: dict[str, Any] = {}

    _all_paths: ClassVar[_PathTree] = {}

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
        updated_path = list(self._path)

        # Recursively update kwargs for children
        for part in module.split("."):
            update.update(self._children_config.get(part, {}))
            updated_path.append(part)

        update["_path"] = updated_path
        self._register_path_inplace_(updated_path)

        # Update kwargs for the module
        return self.model_copy(update=update)

    @classmethod
    def disabled(cls):
        return cls(enabled=False, r=0)

    @classmethod
    def _register_path_inplace_(cls, path: list[str]):
        tree = cls._all_paths
        for part in path:
            tree = tree.setdefault(part, {})

    @classmethod
    def pprint_path_tree(cls):
        def pprint_tree(tree: _PathTree, depth=0):
            for k, v in tree.items():
                print(f"{'   ' * depth}{k}")
                pprint_tree(v, depth + 1)

        pprint_tree(cls._all_paths)
