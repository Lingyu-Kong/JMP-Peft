import functools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import nshtrainer.ll as ll
import torch
import torch.nn as nn
from einops import einsum, pack, reduce
from nshtrainer.ll.typecheck import Bool, Float, tassert
from typing_extensions import override

from ._config import AdapterLayerConfig, DLoraConfig
from ._layers import run_mlps_in_parallel


class AdapterLayer(nn.Module):
    def __init__(self, config: AdapterLayerConfig):
        super().__init__()

        self.config = config

        self.mlp = ll.nn.MLP(
            [self.config.in_dim, self.config.bottleneck_dim, self.config.out_dim],
            activation=self.config.nonlinearity,
            bias=self.config.bias,
            dropout=self.config.dropout,
            residual=self.config.residual,
        )
        self.config.initialization.initialize_(self.mlp)

    @override
    def forward(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_out"]:
        return self.mlp(x)


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
    def num_adapters(self):
        return self.output.shape[0]

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
    ):
        return replace(self, output=fn(self.output))

    def vmap_output(
        self,
        fn: Callable[
            [Float[torch.Tensor, "... d_out"]],
            Float[torch.Tensor, "... d_out"],
        ],
    ):
        return replace(self, output=torch.vmap(fn)(self.output))

    @classmethod
    def from_single_output(cls, output: Float[torch.Tensor, "... d_out"]):
        return cls(
            output=output.unsqueeze(dim=0),
            layerdrop_mask=torch.ones(1, device=output.device, dtype=torch.bool),
        )

    @classmethod
    def concatenate(cls, outputs: Sequence["AdapterOutput"]):
        output, _ = pack([o.output for o in outputs], "* ... d_out")
        tassert(Float[torch.Tensor, "num_adapters ... d_out"], output)

        layerdrop_mask, _ = pack([o.layerdrop_mask for o in outputs], "*")
        tassert(Bool[torch.Tensor, "num_adapters"], layerdrop_mask)

        return cls(output=output, layerdrop_mask=layerdrop_mask)

    def reduce(
        self, reduction: Literal["sum", "mean", "max"]
    ) -> Float[torch.Tensor, "... d_out"]:
        mask = self.layerdrop_mask
        output = self.output

        # Apply the layerdrop mask.
        output = einsum(
            output,
            mask,
            "num_adapters ..., num_adapters -> num_adapters ...",
        )

        # output = reduce(self.output)
        match reduction:
            case "mean":
                sum = reduce(output, "num_adapters ... -> ...", "sum")
                num_valid_adapters = reduce(mask, "num_adapters -> ", "sum")
                output = sum / num_valid_adapters
            case _:
                output = reduce(
                    output,
                    "num_adapters ... -> ...",
                    "sum",
                )

        return output

    @classmethod
    def concatenate_and_reduce(
        cls,
        outputs: Sequence["AdapterOutput"],
        reduction: Literal["sum", "mean", "max"],
    ) -> Float[torch.Tensor, "... d_out"]:
        stacked = cls.concatenate(outputs)

        # FAST PATH: If there is only one output, we can skip the stack and reduce.
        if stacked.num_adapters == 1:
            return stacked.output[0]

        return stacked.reduce(reduction)


def run_adapters(
    x: Float[torch.Tensor, "... d_in"],
    *,
    config: "DLoraConfig",
    adapters: Iterable[AdapterLayer],
    original_mlp: nn.ModuleList,
):
    # Compute the layerdrop mask.
    if config.layerdrop is not None:
        layerdrop_mask = (
            torch.rand(config.num_heads, device=x.device, dtype=x.dtype)
            < config.layerdrop.rate
        )
    else:
        layerdrop_mask = torch.ones(config.num_heads, device=x.device, dtype=torch.bool)

    original_out: Float[torch.Tensor, "... d_out"] = _run_original_mlp(x, original_mlp)

    # Run the adapters in parallel.
    adapter_outs: Float[torch.Tensor, "num_adapters ... d_out"] = run_mlps_in_parallel(
        adapters,
        x,
    )

    # Add the original output to the adapter outputs.
    output = original_out.unsqueeze(dim=0) + adapter_outs
    tassert(Float[torch.Tensor, "num_adapters ... d_out"], output)

    return AdapterOutput(output, layerdrop_mask)


def run_adapters_existing_output(
    x: AdapterOutput,
    *,
    config: "DLoraConfig",
    adapters: Iterable[AdapterLayer],
    original_mlp: nn.ModuleList,
):
    # If we are given an AdapterOutput, we use its layerdrop mask.
    layerdrop_mask = x.layerdrop_mask

    # Vmap the original MLP over the stacked dimension.
    original_out: Float[torch.Tensor, "num_adapters ... d_out"] = torch.vmap(
        functools.partial(_run_original_mlp, module=original_mlp)
    )(x.output)

    # Run the adapters in parallel.
    adapter_outs: Float[torch.Tensor, "num_adapters ... d_out"] = run_mlps_in_parallel(
        adapters,
        x.output,
        is_x_stacked=True,
    )

    # Add the original output to the adapter outputs.
    output = original_out + adapter_outs
    tassert(Float[torch.Tensor, "num_adapters ... d_out"], output)

    return AdapterOutput(output, layerdrop_mask)
