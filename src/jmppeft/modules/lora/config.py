from functools import partial
from logging import getLogger
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from ll import PrivateAttr, TypedConfig

log = getLogger(__name__)


class _LoraKwargs(TypedDict):
    r: int
    lora_alpha: int
    lora_dropout: float
    merge_weights: bool


_PathTree: TypeAlias = dict[str, Any]


class LoraRootConfig(TypedConfig):
    enabled: bool = True
    """Should LoRA be enabled?"""

    enabled_by_default: bool = False
    """Should LoRA be automatically enabled for all Dense layers?"""

    freeze_non_lora_backbone: bool = True
    """Should non-LoRA layers in the backbone be frozen?"""

    children: dict[str, Any] = {}
    """Configuration for children modules."""

    # Default Settings
    r: int = 0
    alpha: int = 1
    dropout: float = 0.0
    merge_weights: bool = False

    # Tracking children
    all_children_paths: _PathTree = {}

    def create_lora_config(self):
        return LoraConfig(
            enabled=self.enabled,
            r=self.r,
            alpha=self.alpha,
            dropout=self.dropout,
            merge_weights=self.merge_weights,
            _root=self,
        )

    @classmethod
    def disabled(cls):
        return cls(enabled=False, r=0)

    def pprint_path_tree(self):
        def pprint_tree(tree: _PathTree, depth=0):
            for k, v in tree.items():
                print(f"{'   ' * depth}{k}")
                pprint_tree(v, depth + 1)

        pprint_tree(self.all_children_paths)


class LoraConfig(TypedConfig):
    enabled: bool
    """Should LoRA be enabled for this module?"""

    # Base Settings
    r: int
    alpha: int
    dropout: float
    merge_weights: bool

    # Specialized Settings for Children
    path: list[str] = []

    _root: LoraRootConfig = PrivateAttr()
    """Root configuration for LoRA."""

    if not TYPE_CHECKING:

        def __init__(self, *args, _root, **kwargs):
            super().__init__(*args, **kwargs)
            self._root = _root

    def as_kwargs(self):
        if not self.enabled:
            raise ValueError(f"LoRA is not enabled. Path: {self.path}")

        assert self.r > 0, f"Invalid r={self.r} for LoRA. Must be > 0."

        return _LoraKwargs(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            merge_weights=self.merge_weights,
        )

    @property
    def gemnet_basis_embedding_cls(self):
        if not self.enabled:
            from ...models.gemnet.layers.efficient import BasisEmbedding

            return BasisEmbedding
        else:
            from ...models.gemnet.layers.lora_efficient import LoRABasisEmbedding

            return partial(LoRABasisEmbedding, lora=self)

    def __call__(self, module: str):
        update: dict[str, Any] = {"enabled": self._root.enabled_by_default}

        # Recursively update kwargs for children
        updated_path: list[str] = []
        for part in [*self.path, *module.split(".")]:
            updated_path.append(part)
            update.update(self._root.children.get(".".join(updated_path), {}))

        update["path"] = updated_path
        self._register_path_inplace_(updated_path)

        # Update kwargs for the module
        updated = self.model_copy(update=update)
        return updated

    @classmethod
    def disabled(cls):
        return LoraRootConfig.disabled().create_lora_config()

    def _register_path_inplace_(self, path: list[str]):
        tree = self._root.all_children_paths
        for part in path:
            tree = tree.setdefault(part, {})
