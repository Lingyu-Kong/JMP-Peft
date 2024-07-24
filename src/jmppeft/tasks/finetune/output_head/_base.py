import contextlib
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Literal

import nshtrainer.ll as ll
import torch.nn as nn
from torch_geometric.data.data import BaseData

from ...config import OutputConfig

log = getLogger(__name__)


class BaseTargetConfig(ll.TypedConfig, ABC):
    name: str
    """The name of the target"""

    loss_coefficient: float = 1.0
    """The loss coefficient for the target"""

    reduction: Literal["sum", "mean"] = "sum"
    """
    The reduction method for the target. This refers to how the target is computed.
    For example, for graph scalar targets, this refers to how the scalar targets are
    computed from each node's scalar prediction.
    """

    @abstractmethod
    def construct_output_head(
        self,
        output_config: OutputConfig,
        d_model_node: int,
        d_model_edge: int,
        activation_cls: type[nn.Module],
    ) -> nn.Module: ...

    @abstractmethod
    def is_classification(self) -> bool: ...

    @contextlib.contextmanager
    def model_forward_context(self, data: BaseData):
        yield

    def supports_inference_mode(self) -> bool:
        return True
