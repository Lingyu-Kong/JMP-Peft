from ll.model import PrimaryMetricConfig

from ...tasks.config import AdamWConfig
from ...tasks.finetune import PDBBindConfig
from ...tasks.finetune import dataset_config as DC


def jmp_l_pdbbind_config_(config: PDBBindConfig):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Set up dataset
    config.train_dataset = DC.pdbbind_config("train")
    config.val_dataset = DC.pdbbind_config("val")
    config.test_dataset = DC.pdbbind_config("test")

    config.batch_size = 1

    # PDBBind specific settings
    config.pdbbind_task = "-logKd/Ki"
    config.metrics.report_rmse = True
    config.primary_metric = PrimaryMetricConfig(
        name=f"pdbbind/{config.pdbbind_task}/y_mae", mode="min"
    )
