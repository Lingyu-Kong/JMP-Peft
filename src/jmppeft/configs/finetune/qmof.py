from pathlib import Path

from ll.model import PrimaryMetricConfig

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import QMOFConfig
from ...tasks.finetune import dataset_config as DC

STATS: dict[str, NC] = {
    "y": NC(mean=2.1866251527, std=1.175752521125648),
}


def jmp_l_qmof_config_(config: QMOFConfig, base_path: Path):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Set up dataset
    config.train_dataset = DC.qmof_config(base_path, "train")
    config.val_dataset = DC.qmof_config(base_path, "val")
    config.test_dataset = DC.qmof_config(base_path, "test")

    # Set up normalization
    if (normalization_config := STATS.get("y")) is None:
        raise ValueError(f"Normalization for {'y'} not found")
    config.normalization = {"y": normalization_config}

    # QMOF specific settings
    config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")
