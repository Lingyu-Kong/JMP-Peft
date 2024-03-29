from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import ll.nn
import numpy as np
import torch
import torch.nn as nn
from ll.typecheck import Bool, Float

from ._layers import run_mlps_in_parallel

if TYPE_CHECKING:
    from ._config import DLoraConfig

AdapterMLP: TypeAlias = nn.Sequential | ll.nn.ResidualSequential


@dataclass(frozen=True)
class AdapterState:
    layerdrop_mask: Bool[np.ndarray, "num_layers"] | None = None

    @classmethod
    def initialize(cls, config: "DLoraConfig"):
        layerdrop_mask = None
        if (dlc := config.droplayer) is not None:
            layerdrop_mask = np.random.rand(config.num_heads) <= dlc.rate

        return cls(layerdrop_mask=layerdrop_mask)


def _run_original_mlp(
    x: Float[torch.Tensor, "... d_in"],
    original_mlp: nn.ModuleList,
) -> Float[torch.Tensor, "... d_out"]:
    for layer in original_mlp:
        x = layer(x)
    return x


def run_adapters(
    x: Float[torch.Tensor, "... d_in"],
    *,
    adapters: Sequence[AdapterMLP],
    original_mlp: nn.ModuleList,
    state: AdapterState,
) -> tuple[Float[torch.Tensor, "num_adapters ... d_out"], AdapterState]:
    # Filter out layers based on the layerdrop mask.
    if state.layerdrop_mask is not None:
        adapters = [
            adapter for adapter, mask in zip(adapters, state.layerdrop_mask) if not mask
        ]

    original_out: Float[torch.Tensor, "... d_out"] = _run_original_mlp(x, original_mlp)

    # Run the adapters in parallel.
    adapter_outs: Float[torch.Tensor, "num_adapters ... d_out"] = run_mlps_in_parallel(
        adapters, x
    )

    # Add the original output to the adapter outputs.
    output = original_out.unsqueeze(dim=0) + adapter_outs

    return output, state
