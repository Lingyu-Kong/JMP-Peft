import copy
from typing import Any, cast

from jmppeft.tasks.finetune.base import (
    FinetuneConfigBase,
    ParamSpecificOptimizerConfig,
    WarmupCosRLPConfig,
)
from typing_extensions import TypeVar


def PARAMETER_PATTERNS(num_blocks: int):
    return {
        "embedding": ["embedding.*"],
        "additional_embedding": ["additional_embedding.*"],
        "bases": ["backbone.bases.*"],
        # "all_int_blocks": ["backbone.int_blocks.*"],
        **{
            f"int_blocks_{i}": [f"backbone.int_blocks.{i}.*"] for i in range(num_blocks)
        },
        # "all_out_blocks": ["backbone.out_blocks.*"],
        **{
            f"out_blocks_{i}": [f"backbone.out_blocks.{i}.*"]
            for i in range(num_blocks + 1)
        },
        **{
            f"blocks_{i}": [
                f"backbone.int_blocks.{i}.*",
                f"backbone.out_blocks.{i+1}.*",
                *(["backbone.out_blocks.0.*"] if i == 0 else []),
            ]
            for i in range(num_blocks)
        },
        "out_mlp_E": ["backbone.out_mlp.E.*"],
    }


TConfig = TypeVar("TConfig", infer_variance=True)


def make_parameter_specific_optimizer_config(
    config: FinetuneConfigBase,
    num_blocks: int,
    max_lr_scales: dict[str, float],
    lora_lr_scale: float | None = None,
    include_dlora_in_lora_patterns: bool = False,
):
    """
    Create a list of parameter specific optimizers based on the max_lr_scales.

    Args:
        config: The finetune config.
        num_blocks: The number of blocks in the backbone.
        max_lr_scales: A dictionary of max_lr_scales for each parameter group.
        lora_lr_scale:
            Scale the learning rate (as a multiplier of the base learning rate) for LoRA weights.
            `None` means no scaling (i.e. use the base learning rate).

    """
    base_lr = config.optimizer.lr

    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] = []
    max_lr_scales = cast(dict[str, Any], max_lr_scales)

    # If LoRA is enabled, we create two copies of each config: One for LoRA and one for non-LoRA.
    if lora_lr_scale is not None:
        assert (
            config.lora is not None
        ), "config.lora must be set if lora_lr_scale is not None"
        assert (
            config.lora.enabled
        ), "config.lora.enabled must be True if lora_lr_scale is not None"

    for name, lr_scale in max_lr_scales.items():
        assert isinstance(lr_scale, float), f"max_lr_scales[{name}] must be float"

        optimizer = copy.deepcopy(config.optimizer)
        optimizer.lr = base_lr * lr_scale

        lrs = None
        match config.lr_scheduler:
            case WarmupCosRLPConfig():
                lrs = copy.deepcopy(config.lr_scheduler)
                # We now scale down the cos annealing min LR factor
                #   so that the final LR is the same as the original config.
                lrs.min_lr_factor = lrs.min_lr_factor / lr_scale
                lrs.min_lr_factor = max(0.01, min(0.99, lrs.min_lr_factor))
            case _:
                raise ValueError(
                    "You must set config.lr_scheduler to WarmupCosRLPConfig to use parameter specific optimizers."
                )

        assert (
            (parameter_patterns := PARAMETER_PATTERNS(num_blocks).get(name)) is not None
        ), f"PARAMETER_PATTERNS[{name}] is None. You must set PARAMETER_PATTERNS[{name}]"
        parameter_specific_optimizers.append(
            ParamSpecificOptimizerConfig(
                name=name,
                paremeter_patterns=parameter_patterns,
                optimizer=optimizer,
                lr_scheduler=lrs,
            )
        )

    # If no LoRA, we return the parameter specific optimizers.
    if lora_lr_scale is None:
        return parameter_specific_optimizers

    # Otherwise, create those for LoRA as well.
    parameter_specific_optimizers_lora: list[ParamSpecificOptimizerConfig] = []
    for name, lr_scale in max_lr_scales.items():
        assert isinstance(lr_scale, float), f"max_lr_scales[{name}] must be float"

        optimizer = copy.deepcopy(config.optimizer)
        optimizer.lr = base_lr * lr_scale

        # LoRA: Multiply the learning rate by the lora_lr_scale
        optimizer.lr *= lora_lr_scale

        lrs = None
        match config.lr_scheduler:
            case WarmupCosRLPConfig():
                lrs = copy.deepcopy(config.lr_scheduler)
                # We now scale down the cos annealing min LR factor
                #   so that the final LR is the same as the original config.
                lrs.min_lr_factor = lrs.min_lr_factor / lr_scale
                lrs.min_lr_factor = max(0.01, min(0.99, lrs.min_lr_factor))
            case _:
                raise ValueError(
                    "You must set config.lr_scheduler to WarmupCosRLPConfig to use parameter specific optimizers."
                )

        assert (
            (parameter_patterns := PARAMETER_PATTERNS(num_blocks).get(name)) is not None
        ), f"PARAMETER_PATTERNS[{name}] is None. You must set PARAMETER_PATTERNS[{name}]"

        # LoRA: Update the parameter patterns for LoRA
        parameter_patterns_new: list[str] = []
        for pattern in parameter_patterns:
            assert pattern.endswith("*"), f"pattern must end with '*' but got {pattern}"
            parameter_patterns_new.append(f"{pattern}.lora_*")

            if include_dlora_in_lora_patterns:
                parameter_patterns_new.append(f"{pattern}*_adapters.*")

        parameter_specific_optimizers_lora.append(
            ParamSpecificOptimizerConfig(
                name=f"{name}_lora",
                paremeter_patterns=parameter_patterns_new,
                optimizer=optimizer,
                lr_scheduler=lrs,
            )
        )

    # We want the LoRA parameter specific optimizers to come first, as they are more specific.
    return parameter_specific_optimizers_lora + parameter_specific_optimizers
