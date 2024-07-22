import contextlib
import copy
import fnmatch
import math
import time
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from functools import cache, cached_property, partial
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, cast

import datasets
import ll
import ll.typecheck as tc
import numpy as np
import rich
import rich.console
import rich.markdown
import rich.table
import rich.tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
)
from ll import (
    ActSave,
    AllowMissing,
    BaseConfig,
    Field,
    LightningModuleBase,
)
from ll.data.balanced_batch_sampler import BalancedBatchSampler, DatasetWithSizes
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar, assert_never, override


class FocalMAEConfig(ll.TypedConfig):
    freq_ratios_path: Path
    """Path to the file with the frequency ratios"""


class AtomTypeBasedLossMultiplier(nn.Module):
    freq_ratios: tc.Float[torch.Tensor, "atom_type"]

    @override
    def __init__(self, config: AtomTypeBasedLossMultiplierConfig):
        super().__init__()

        match config.atom_type_occurrences:
            case Mapping():
                atom_type_occurrences = torch.zeros(
                    (max(config.atom_type_occurrences.keys()) + 1,),
                    dtype=torch.int64,
                )
                for atom_type, occurrences in config.atom_type_occurrences.items():
                    atom_type_occurrences[atom_type] = occurrences
            case np.ndarray():
                atom_type_occurrences = torch.from_numpy(
                    config.atom_type_occurrences
                ).to(torch.int64)
            case torch.Tensor():
                atom_type_occurrences = config.atom_type_occurrences.to(torch.int64)
            case _:
                assert_never(config.atom_type_occurrences)

        self.register_buffer("atom_type_occurrences", atom_type_occurrences)

    @override
    def forward(
        self,
        atomic_numbers: tc.Int[torch.Tensor, "num_nodes"],
        batch_idx: tc.Int[torch.Tensor, "num_nodes"],
    ) -> torch.Tensor:
        pass
