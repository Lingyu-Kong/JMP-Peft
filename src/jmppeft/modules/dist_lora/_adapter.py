from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self, TypeAlias

import ll.nn
import torch
import torch.nn as nn
from einops import pack
from ll.typecheck import Bool, Float, tassert

from ._layers import run_mlps_in_parallel

if TYPE_CHECKING:
    from ._config import DLoraConfig

AdapterMLP: TypeAlias = nn.Sequential | ll.nn.ResidualSequential


def _run_original_mlp(
    x: Float[torch.Tensor, "... d_in"],
    module: nn.ModuleList,
) -> Float[torch.Tensor, "... d_out"]:
    for layer in module:
        x = layer(x)
    return x


@dataclass(frozen=True)
class AdapterOutput:
    output: Float[torch.Tensor, "num_adapters ... d_out"]
    layerdrop_mask: Bool[torch.Tensor, "num_adapters"]
    """
    Layerdrop mask for each adapter.

    NOTE: This mask is true for layers that are kept and false for layers that are dropped.
    """

    @property
    def device(self):
        return self.output.device

    @property
    def dtype(self):
        return self.output.dtype

    def map_output(
        self,
        fn: Callable[
            [Float[torch.Tensor, "num_adapters ... d_out"]],
            Float[torch.Tensor, "num_adapters ... d_out"],
        ],
    ) -> Self:
        return replace(self, output=fn(self.output))

    def vmap_output(
        self,
        fn: Callable[
            [Float[torch.Tensor, "... d_out"]],
            Float[torch.Tensor, "... d_out"],
        ],
    ) -> Self:
        return replace(self, output=torch.vmap(fn)(self.output))

    @classmethod
    def stack(cls, outputs: Sequence[Self]):
        output, _ = pack([o.output for o in outputs], "* ... d_out")
        tassert(Float[torch.Tensor, "num_adapters ... d_out"], output)

        layerdrop_mask, _ = pack([o.layerdrop_mask for o in outputs], "*")
        tassert(Bool[torch.Tensor, "num_adapters"], layerdrop_mask)

        return cls(output=output, layerdrop_mask=layerdrop_mask)

    @classmethod
    def from_single_output(cls, output: Float[torch.Tensor, "... d_out"]):
        return cls(
            output=output.unsqueeze(dim=0),
            layerdrop_mask=torch.tensor([True], device=output.device, dtype=torch.bool),
        )


def run_adapters(
    x: Float[torch.Tensor, "... d_in"] | AdapterOutput,
    *,
    config: "DLoraConfig",
    adapters: Iterable[AdapterMLP],
    original_mlp: nn.ModuleList,
):
    # Compute the layerdrop mask.
    if isinstance(x, AdapterOutput):
        # If we are given an AdapterOutput, we use its layerdrop mask.
        layerdrop_mask = x.layerdrop_mask
    else:
        # Otherwise, we compute the layerdrop mask.
        if config.layerdrop is not None:
            layerdrop_mask = (
                torch.rand(len(original_mlp), device=x.device, dtype=x.dtype)
                < config.layerdrop.rate
            )
        else:
            layerdrop_mask = torch.ones(
                len(original_mlp), device=x.device, dtype=torch.bool
            )

    x_original = x.output if isinstance(x, AdapterOutput) else x
    original_out: Float[torch.Tensor, "... d_out"] = _run_original_mlp(
        x_original, original_mlp
    )

    # Run the adapters in parallel.
    adapter_outs: Float[torch.Tensor, "num_adapters ... d_out"] = run_mlps_in_parallel(
        adapters,
        x_original,
    )

    # Add the original output to the adapter outputs.
    output = original_out.unsqueeze(dim=0) + adapter_outs
    tassert(Float[torch.Tensor, "num_adapters ... d_out"], output)

    return AdapterOutput(output, layerdrop_mask)
