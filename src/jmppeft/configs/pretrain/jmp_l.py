from ll import GradientClippingConfig

from ...modules.dataset.concat_dataset import MTDatasetConfig
from ...modules.ema import EMAConfig
from ...tasks.config import AdamWConfig
from ...tasks.pretrain.module import (
    LinearWarmupCosineAnnealingSchedulerConfig,
    PretrainConfig,
)


def jmp_l_pt_config_(config: PretrainConfig):
    # Set the model trainer settings for maximum performance
    config.trainer.precision = "16-mixed-auto"
    config.trainer.set_float32_matmul_precision = "medium"

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=3.0e-4,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.gradient_clipping = GradientClippingConfig(
        value=1.0,
        algorithm="norm",
    )
    # LR Scheduler settings
    config.lr_scheduler = LinearWarmupCosineAnnealingSchedulerConfig(
        warmup_steps=2000,
        warmup_start_lr_factor=0.2,
        min_lr_factor=0.1,
        max_epochs=2,
    )
    # Regularization settings
    config.edge_dropout = 0.1
    # EMA settings
    config.ema = EMAConfig(decay=0.99)

    # Set data config
    config.num_workers = 8

    # Set up the JMP MT dataset config and tasks
    config.mt_dataset = MTDatasetConfig(
        sample_type="temperature",
        sample_temperature=2.0,
    )
