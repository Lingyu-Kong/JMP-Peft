"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import torch
import torch.nn as nn
from jaxtyping import Float

from ..config import LoraConfig
from ..initializers import he_orthogonal_init

try:
    import loralib
except ImportError:
    loralib = None


class Dense(nn.Module):
    """
    Combines dense layer with scaling for silu activation.

    Arguments
    ---------
    in_features: int
        Input embedding size.
    out_features: int
        Output embedding size.
    bias: bool
        True if use bias.
    activation: str
        Name of the activation function to use.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        activation=None,
        scale_dim: bool = False,
        *,
        lora: LoraConfig | None,
        dropout: float | None,
    ):
        super().__init__()

        self.scale_dim = scale_dim
        self.in_features = in_features
        self.out_features = out_features

        if lora:
            assert loralib is not None, "Loralib is not installed."

            self.linear = loralib.Linear(
                in_features,
                out_features,
                bias=bias,
                **lora.as_kwargs(),
            )
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["scaled_silu", "scaled_swish"]:
            self.activation = ScaledSiLU()
        elif activation in ["silu", "swish"]:
            self.activation = nn.SiLU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()

    def reset_parameters(
        self,
        initializer=he_orthogonal_init,
    ):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            _ = self.linear.bias.data.fill_(0)

    def forward(
        self, x: Float[torch.Tensor, "... {self.in_features}"]
    ) -> Float[torch.Tensor, "... {self.out_features}"]:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.scale_dim:
            x = x * (self.linear.weight.shape[1] ** -0.5)
        return x


class ScaledSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class ResidualLayer(nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Arguments
    ---------
    units: int
        Input and output embedding size.
    nLayers: int
        Number of dense layers.
    layer: nn.Module
        Class for the layers inside the residual block.
    layer_kwargs: str
        Keyword arguments for initializing the layers.
    """

    def __init__(
        self,
        units: int,
        nLayers: int = 2,
        layer: type[Dense] = Dense,
        *,
        lora: LoraConfig | None,
        **layer_kwargs,
    ):
        super().__init__()

        assert layer is Dense, "Only Dense layers are supported for now."

        self.dense_mlp = nn.Sequential(
            *[
                layer(
                    in_features=units,
                    out_features=units,
                    bias=False,
                    lora=lora,
                    **layer_kwargs,
                )
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x
