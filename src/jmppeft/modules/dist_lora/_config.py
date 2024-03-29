import math
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import ll.nn
import torch.nn as nn
from ll import Field, TypedConfig

if TYPE_CHECKING:
    from ...tasks.finetune.base import FinetuneConfigBase


class AdapterLayerHoulsbyInitializationConfig(TypedConfig):
    """
    Initializes the parameters wrt. the adapter layer paper: https://arxiv.org/pdf/1902.00751.pdf

    From the paper:
        With the skip-connection, if the parameters of the projection layers
        are initialized to near-zero, the module is initialized to an
        approximate identity function.
    """

    name: Literal["houlsby"] = "houlsby"
    std: float = 0.02

    def initialize_(self, mlp: nn.Sequential):
        for layer in mlp.modules():
            if not isinstance(layer, nn.Linear):
                continue

            nn.init.normal_(layer.weight, mean=0, std=0.02)


class AdapterLayerLoRAInitializationConfig(TypedConfig):
    """
    Initializes the parameters wrt. the LoRA paper: https://arxiv.org/pdf/2106.09623.pdf

    The LoRA paper initializes the downprojection matrix, A,
    in a similar fashion to any other layer (N(0, sigma^2)), and
    the upprojection matrix, B, is initialized to zero. The end-result
    is that the adapter layer MLP does not have any effect on the input
    (i.e., outputs zero) until the upprojection matrix is trained
    to be non-zero.
    """

    name: Literal["lora"] = "lora"
    A_std: float = math.sqrt(5.0)
    B_std: float = 0.0

    def initialize_(self, mlp: nn.Sequential):
        linear_layers = [
            layer for layer in mlp.modules() if isinstance(layer, nn.Linear)
        ]
        # There should be exactly two linear layers in the MLP.
        assert (
            len(linear_layers) == 2
        ), f"Expected 2 linear layers, got {len(linear_layers)}"

        # Figure out which is A and B by looking at the in_dim, out_dim.
        # Important note: The middle dimension is the bottleneck dimension
        #   and should be the same for both linear layers.
        first, second = linear_layers
        if first.in_features == second.out_features:
            A, B = first, second
        elif first.out_features == second.in_features:
            A, B = second, first
        else:
            raise ValueError("Unexpected linear layer dimensions")

        nn.init.normal_(A.weight, mean=0.0, std=self.A_std)
        if self.B_std:
            nn.init.normal_(B.weight, mean=0.0, std=self.B_std)
        else:
            nn.init.zeros_(B.weight)


AdapterLayerInitializationConfig: TypeAlias = Annotated[
    AdapterLayerHoulsbyInitializationConfig | AdapterLayerLoRAInitializationConfig,
    Field(discriminator="name"),
]


class AdapterLayerConfig(TypedConfig):
    in_dim: int
    bottleneck_dim: int
    out_dim: int
    nonlinearity: ll.nn.NonlinearityConfig
    bias: bool = True
    dropout: float | None = None
    residual: bool = False

    initialization: AdapterLayerInitializationConfig = (
        AdapterLayerHoulsbyInitializationConfig()
    )


class LayerDropConfig(TypedConfig):
    rate: float


class DLoraConfig(TypedConfig):
    num_heads: int

    seq_energy2_output_block: AdapterLayerConfig
    seq_energy_pre_output_block: AdapterLayerConfig | None
    seq_forces_output_block: AdapterLayerConfig | None

    adapter_reduction: Literal["sum", "mean", "max"]

    layerdrop: LayerDropConfig | None = None

    def disable_lora_for_dlora_(self, config: "FinetuneConfigBase"):
        if config.lora is not None:
            config.lora.children["out_mlp_E"] = {"enabled": False}
            config.lora.children["out_mlp_F"] = {"enabled": False}

            for i in range(config.backbone.num_blocks + 1):
                config.lora.children[f"out_blocks_{i}"] = {"enabled": False}

        # Ensure the final layers are being trained.
        config.freeze.ensure_non_frozen_parameter_patterns.append(
            "backbone.out_mlp_E.*"
        )
        config.freeze.ensure_non_frozen_parameter_patterns.append(
            "backbone.out_mlp_F.*"
        )
