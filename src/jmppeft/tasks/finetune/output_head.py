import copy
import fnmatch
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    TypeAlias,
    assert_never,
    cast,
    final,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from ll import Base, BaseConfig, Field, LightningModuleBase, TypedConfig
from ll.data.balanced_batch_sampler import BalancedBatchSampler, DatasetWithSizes
from ll.nn import MLP
from ll.util.typed import TypedModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import TypedDict, TypeVar, override

from ...datasets.finetune_lmdb import FinetuneDatasetConfig as FinetuneDatasetConfigBase
from ...datasets.finetune_lmdb import FinetuneLmdbDataset
from ...datasets.finetune_pdbbind import PDBBindConfig, PDBBindDataset
from ...models.gemnet.backbone import GemNetOCBackbone, GOCBackboneOutput
from ...models.gemnet.config import BackboneConfig
from ...models.gemnet.layers.base_layers import ScaledSiLU
from ...modules import transforms as T
from ...modules.dataset import dataset_transform as DT
from ...modules.dataset.common import CommonDatasetConfig, wrap_common_dataset
from ...modules.early_stopping import EarlyStoppingWithMinLR
from ...modules.ema import EMAConfig
from ...modules.lora import LoraConfig
from ...modules.scheduler.linear_warmup_cos_rlp import (
    PerParamGroupLinearWarmupCosineAnnealingRLPLR,
)
from ...modules.transforms.normalize import NormalizationConfig
from ...utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from ...utils.state_dict import load_state_dict
from ..config import (
    EmbeddingConfig,
    OptimizerConfig,
    OutputConfig,
    optimizer_from_config,
)
from .metrics import FinetuneMetrics, MetricPair, MetricsConfig

log = getLogger(__name__)


class BaseTargetConfig(TypedConfig, ABC):
    name: str
    """The name of the target"""

    loss_coefficient: float = 1.0
    """The loss coefficient for the target"""

    reduction: Literal["sum", "mean", "max"] = "sum"
    """
    The reduction method for the target. This refers to how the target is computed.
    For example, for graph scalar targets, this refers to how the scalar targets are
    computed from each node's scalar prediction.
    """

    @abstractmethod
    def construct_output_head(self) -> nn.Module:
        ...


@final
class GraphScalarTargetConfig(BaseTargetConfig):
    kind: Literal["scalar"] = "scalar"

    @override
    def construct_output_head(self) -> nn.Module:
        raise NotImplementedError


@final
class GraphBinaryClassificationTargetConfig(BaseTargetConfig):
    kind: Literal["binary"] = "binary"

    num_classes: int
    """The number of classes for the target"""

    pos_weight: float | None = None
    """The positive weight for the target"""

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.num_classes != 2:
            raise ValueError(
                f"Binary classification target {self.name} has {self.num_classes} classes"
            )

    @override
    def construct_output_head(self) -> nn.Module:
        raise NotImplementedError


@final
class GraphMulticlassClassificationTargetConfig(BaseTargetConfig):
    kind: Literal["multiclass"] = "multiclass"

    num_classes: int
    """The number of classes for the target"""

    class_weights: list[float] | None = None
    """The class weights for the target"""

    dropout: float | None = None
    """The dropout probability to use before the output layer"""

    @override
    def construct_output_head(self) -> nn.Module:
        raise NotImplementedError


GraphTargetConfig: TypeAlias = Annotated[
    GraphScalarTargetConfig
    | GraphBinaryClassificationTargetConfig
    | GraphMulticlassClassificationTargetConfig,
    Field(discriminator="kind"),
]


@final
class NodeVectorTargetConfig(BaseTargetConfig):
    kind: Literal["vector"] = "vector"

    @override
    def construct_output_head(self) -> nn.Module:
        raise NotImplementedError


NodeTargetConfig: TypeAlias = Annotated[
    NodeVectorTargetConfig, Field(discriminator="kind")
]
