# %%
from pathlib import Path

import ll
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.pretrain import module as M


def base_config_(config: M.PretrainConfig):
    # Set the model trainer settings for maximum performance
    config.trainer.precision = "16-mixed-auto"
    config.trainer.set_float32_matmul_precision = "medium"
    config.trainer.supports_parameter_hooks = False
    config.trainer.supports_skip_batch_exception = False

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=3.0e-4,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.gradient_clipping = ll.GradientClippingConfig(
        value=1.0,
        algorithm="norm",
    )
    # LR Scheduler settings
    config.lr_scheduler = M.LinearWarmupCosineAnnealingSchedulerConfig(
        warmup_steps=2000,
        warmup_start_lr_factor=0.2,
        min_lr_factor=0.1,
        max_epochs=2,
    )
    # Regularization settings
    config.edge_dropout = 0.1
    # EMA settings
    config.ema = M.EMAConfig(decay=0.99)

    # Set data config
    config.batch_size = 8
    config.num_workers = 8

    # Set up the JMP MT dataset config and tasks
    config.mt_dataset = M.MTDatasetConfig(
        sample_type="temperature",
        sample_temperature=2.0,
    )


def tasks_config_(config: M.PretrainConfig):
    config.tasks = [
        M.TaskConfig(
            name="oc20",
            train_dataset=M.PretrainDatasetConfig(
                src=Path("/mnt/datasets/s2ef/2M/train/"),
                metadata_path=Path("/mnt/datasets/s2ef/2M/train_metadata.npz"),
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path("/mnt/datasets/s2ef/all/val_id/"),
                metadata_path=Path("/mnt/datasets/s2ef/all/val_id_metadata.npz"),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=73.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=24.901469505465872),
                "force": M.NormalizationConfig(mean=0.0, std=0.5111534595489502),
            },
        ),
        M.TaskConfig(
            name="oc22",
            train_dataset=M.PretrainDatasetConfig(
                src=Path("/mnt/shared/pre-training-datasets/oc22/s2ef-total/train/"),
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path("/mnt/shared/pre-training-datasets/oc22/s2ef-total/val_id/"),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=80.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=25.229595396538468),
                "force": M.NormalizationConfig(mean=0.0, std=0.25678861141204834),
            },
        ),
        M.TaskConfig(
            name="ani1x",
            train_dataset=M.PretrainDatasetConfig(
                src=Path("/mnt/shared/pre-training-datasets/ani1x/train/"),
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path("/mnt/shared/pre-training-datasets/ani1x/val/"),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=15.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=2.8700712783472118),
                "force": M.NormalizationConfig(mean=0.0, std=2.131422996520996),
            },
        ),
        M.TaskConfig(
            name="transition1x",
            train_dataset=M.PretrainDatasetConfig(
                src=Path("/mnt/shared/pre-training-datasets/trans1x/train/"),
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path("/mnt/shared/pre-training-datasets/trans1x/val/"),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=14.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=1.787466168382901),
                "force": M.NormalizationConfig(mean=0.0, std=0.3591422140598297),
            },
        ),
    ]


def backbone_config_(config: M.PretrainConfig):
    config.dropout = None
    config.edge_dropout = None

    config.backbone = M.TorchMDNetBackboneConfig()
    config.backbone.apply_1_5M_param_config_()

    config.exclude_keys.remove("edge_index")


def fsdp_config_(config: M.PretrainConfig):
    config.fsdp = True


configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []

config = M.PretrainConfig.draft()
base_config_(config)
tasks_config_(config)
backbone_config_(config)
fsdp_config_(config)
config = config.finalize()
configs.append((config, M.PretrainModel))


# %%
def run(config: M.PretrainConfig, model_cls: type[M.PretrainModel]):
    model = model_cls(config)

    trainer = ll.Trainer(config, **model.fsdp_trainer_kwargs())
    trainer.fit(model)


# %%
runner = ll.Runner(run)
runner.fast_dev_run(configs, n_batches=128)

# %%
runner = ll.Runner(run)
runner.session(configs, snapshot=True, env={"CUDA_VISIBLE_DEVICES": "1"})
