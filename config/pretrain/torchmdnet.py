# %%
import math

import ll
from jmppeft.configs.pretrain.tasks import tasks_config_frontier_
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


def backbone_config_(config: M.PretrainConfig):
    config.dropout = None
    config.edge_dropout = None

    config.backbone = M.TorchMDNetBackboneConfig()
    config.backbone.apply_1_5M_param_config_()

    config.exclude_keys.remove("edge_index")


def fsdp_config_(config: M.PretrainConfig):
    config.fsdp = True


def profiling_config_(config: M.PretrainConfig):
    config.trainer.python_logging.pretty_()

    config.global_train_sample_ratio = M.DatasetSampleRatioConfig(
        sample_ratio=0.00045,
        seed=0,
    )
    config.global_val_sample_ratio = M.DatasetSampleRatioConfig(
        sample_ratio=0.02,
        seed=0,
    )

    config.trainer.max_epochs = 1

    config.trainer.callbacks.append(ll.callbacks.EpochTimerConfig())
    config.trainer.callbacks.append(ll.callbacks.FiniteChecksConfig())


def _print_dataset_sizes(config: M.PretrainConfig):
    print()
    total_size = 0
    print("Train")
    print()
    sizes = [2_000_000, 8_000_000, 2_000_000, 10_000_000]
    for task, size in zip(config.tasks, sizes):
        ratio = 1.0
        if task.train_dataset.sample_ratio is not None:
            ratio = task.train_dataset.sample_ratio.sample_ratio
        final_size = math.ceil(size * ratio)
        total_size += final_size
        print(f"{task.name}: {final_size:_}")

    print()
    print(f"Total: {total_size:_}")
    print()
    print()

    total_size = 0
    print("Val")
    print()
    sizes = [20_000, 10_000, 5_000, 10_000]
    for task, size in zip(config.tasks, sizes):
        ratio = 1.0
        if task.val_dataset.sample_ratio is not None:
            ratio = task.val_dataset.sample_ratio.sample_ratio
        final_size = math.ceil(size * ratio)
        total_size += final_size
        print(f"{task.name}: {final_size:_}")

    print()
    print(f"Total: {total_size:_}")


def frontier_config_(config: M.PretrainConfig):
    if wandb := config.trainer.logging.wandb:
        wandb.disable_()


configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []

config = M.PretrainConfig.draft()
base_config_(config)
tasks_config_frontier_(config)
backbone_config_(config)
# fsdp_config_(config)
profiling_config_(config)
frontier_config_(config)
config = config.finalize()
_print_dataset_sizes(config)
configs.append((config, M.PretrainModel))


# %%
def run(config: M.PretrainConfig, model_cls: type[M.PretrainModel]):
    model = model_cls(config)

    trainer = ll.Trainer(config, **model.fsdp_trainer_kwargs())
    trainer.fit(model)


# %%
runner = ll.Runner(run)
runner.session(configs, snapshot=False, env={"CUDA_VISIBLE_DEVICES": "0"})
