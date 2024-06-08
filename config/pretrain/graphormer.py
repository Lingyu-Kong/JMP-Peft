# %%
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
        value=2.0,
        algorithm="value",
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
    config.num_workers = 4

    # Set up the JMP MT dataset config and tasks
    config.mt_dataset = M.MTDatasetConfig(
        sample_type="temperature",
        sample_temperature=2.0,
    )


def frontier_compute_node_config_(config: M.PretrainConfig):
    if config.trainer.logging.wandb:
        config.trainer.logging.wandb.offline = True


def backbone_config_(config: M.PretrainConfig):
    backbone = M.Graphormer3DConfig.draft()
    backbone.graphormer_large_()
    backbone.layers *= 3
    config.backbone = backbone.finalize()


def fsdp_config_(config: M.PretrainConfig):
    config.fsdp = M.FSDPConfig(
        gradient_checkpointing=True,
        cpu_offload=False,
    )


def gradient_checkpointing_config_(config: M.PretrainConfig):
    config.gradient_checkpointing = True


def multi_head_loss_trick_config_(config: M.PretrainConfig):
    config.multi_head_loss_trick = True

    config.trainer.optimizer.gradient_clipping = None


configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []

config = M.PretrainConfig.draft()
base_config_(config)
frontier_compute_node_config_(config)
tasks_config_frontier_(config)
backbone_config_(config)
fsdp_config_(config)
# gradient_checkpointing_config_(config)
# multi_head_loss_trick_config_(config)

config.batch_size = 8
config = config.finalize()
configs.append((config, M.PretrainModel))


# %%
def run(config: M.PretrainConfig, model_cls: type[M.PretrainModel]):
    model = model_cls(config)

    trainer = ll.Trainer(config, **model.fsdp_trainer_kwargs())
    trainer.fit(model)


# %%
runner = ll.Runner(run)
runner.fast_dev_run_session(
    configs,
    n_batches=128,
    # gpus=[0, 1, 2, 3],
    # env={
    #     "HSA_XNACK": "1",
    # },
    setup_commands=[
        "source /lustre/orion/mat265/world-shared/nimashoghi/repositories/jmp-peft/rocm60.sh"
    ],
)


# %%
runner = ll.Runner(run)
runner.fast_dev_run(configs, n_batches=128)

# %%
runner = ll.Runner(run)
runner.session(configs, snapshot=False)
