from pathlib import Path

from ll import GradientClippingConfig
from ll.model import EarlyStoppingConfig

from ...models.gemnet.config import BackboneConfig
from ...tasks.config import AdamWConfig
from ...tasks.finetune import FinetuneConfigBase
from ...tasks.finetune.base import RLPConfig, WarmupCosRLPConfig
from ...utils.param_specific_util import make_parameter_specific_optimizer_config


def jmp_s_ft_config_(
    config: FinetuneConfigBase,
    ckpt_path: Path | None = None,
    ema_backbone: bool = True,
    disable_force_output_heads: bool = True,
    use_bf16: bool = False,
):
    # Set the model trainer settings for maximum performance
    config.trainer.precision = "bf16-mixed" if use_bf16 else "fp16-mixed"
    config.trainer.set_float32_matmul_precision = "medium"

    # Set backbone config
    config.backbone = BackboneConfig.base()
    config.backbone.scale_basis = False

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=8.0e-5,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.gradient_clipping = GradientClippingConfig(
        value=1.0,
        algorithm="value",
    )
    # LR Scheduler settings
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=5,
        warmup_start_lr_factor=1.0e-1,
        should_restart=False,
        max_epochs=32,
        min_lr_factor=0.1,
        rlp=RLPConfig(patience=3, factor=0.8),
    )
    # LLRD Settings
    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
        config,
        config.backbone.num_blocks,
        {
            "embedding": 0.3,
            "blocks_0": 0.30,
            "blocks_1": 0.40,
            "blocks_2": 0.55,
            "blocks_3": 0.625,
        },
    )

    # Checkpoint loading settings
    # We want to use EMA weights from pretraining
    if ckpt_path is not None:
        config.meta["ckpt_path"] = ckpt_path
        config.meta["ema_backbone"] = ema_backbone

    # Set data config
    config.num_workers = 8

    # Base early stopping settings
    config.trainer.max_epochs = 500
    config.trainer.max_time = "07:00:00:00"
    config.trainer.early_stopping = EarlyStoppingConfig(
        patience=50,
        min_delta=1.0e-8,
        min_lr=1.0e-8,
    )

    # If we are not using force output heads, we need to disable them
    if disable_force_output_heads:
        config.backbone.regress_forces = False
        config.backbone.direct_forces = False
