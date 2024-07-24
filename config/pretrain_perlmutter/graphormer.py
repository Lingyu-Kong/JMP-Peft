# %%
from typing import Literal

import nshtrainer.ll as ll
from jmppeft.configs.pretrain.tasks import tasks_config_perlmutter_
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.pretrain import module as M

PROJECT_ROOT = "/global/cfs/cdirs/m3641/Nima/projdata/jmp-pretrain"


def base_config_(config: M.PretrainConfig):
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
    config.trainer.optimizer.gradient_clipping = ll.GradientClippingConfig(
        value=2.0,
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
    config.num_workers = 4

    # Set up the JMP MT dataset config and tasks
    config.mt_dataset = M.MTDatasetConfig(
        sample_type="temperature",
        sample_temperature=2.0,
    )


def graphormer_backbone_config_(
    config: M.PretrainConfig,
    variant: Literal["base", "large", "xl"],
):
    backbone = M.Graphormer3DConfig.draft()
    config.name_parts.extend(("graphormer", variant))
    match variant:
        case "base":
            backbone.graphormer_base_()
        case "large":
            backbone.graphormer_large_()
        case "xl":
            backbone.graphormer_extra_large_()
        case _:
            raise ValueError(f"Unknown variant: {variant}")
    config.backbone = backbone.finalize()


def goc_backbone_config_(
    config: M.PretrainConfig,
    variant: Literal["base", "large", "xl"],
):
    # Set backbone config
    config.name_parts.append("gemnet")
    match variant:
        case "base":
            config.backbone = M.GOCBackboneConfig.base()
        case "large":
            config.backbone = M.GOCBackboneConfig.large()
        case "xl":
            config.backbone = M.GOCBackboneConfig.xl()
        case _:
            raise ValueError(f"Invalid variant: {variant}")
    config.backbone.scale_basis = False


def fsdp_config_(config: M.PretrainConfig):
    config.fsdp = M.FSDPConfig(
        gradient_checkpointing=False,
    )


def gradient_checkpointing_config_(config: M.PretrainConfig):
    config.gradient_checkpointing = True
    config.name_parts.append("gc")


def multi_head_loss_trick_config_(config: M.PretrainConfig):
    raise NotImplementedError("Multi-head loss trick is not implemented yet.")
    config.multi_head_loss_trick = True

    config.trainer.optimizer.gradient_clipping = None


def profiling_config_(config: M.PretrainConfig):
    config.trainer.callbacks.append(ll.callbacks.EpochTimerConfig())
    config.trainer.callbacks.append(
        ll.callbacks.ThroughputMonitorConfig(batch_size=config.batch_size)
    )
    config.perf_metrics = True


configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []

variants = ("xl",)
for variant in variants:
    config = M.PretrainConfig.draft()
    base_config_(config)
    tasks_config_perlmutter_(config)
    graphormer_backbone_config_(config, variant)
    # fsdp_config_(config)
    gradient_checkpointing_config_(config)
    profiling_config_(config)
    config.with_project_root_(PROJECT_ROOT)
    config.project = "jmp-pretrain-perlmutter"

    config.batch_size = 2
    config.num_workers = 4
    config = config.finalize()
    configs.append((config, M.PretrainModel))


# %%
def run(config: M.PretrainConfig, model_cls: type[M.PretrainModel]):
    model = model_cls(config)

    trainer = ll.Trainer(config, **model.fsdp_trainer_kwargs())
    trainer.fit(model)


setup_commands = []
env = {"LL_DISABLE_TYPECHECKING": "1"}

# %%
runner = ll.Runner(run)
runner.session(
    configs,
    snapshot=False,
    setup_commands=setup_commands,
    env=env,
)
# %%
runner = ll.Runner(run)
runner.fast_dev_run_session(
    configs,
    n_batches=128,
    setup_commands=setup_commands,
    env=env,
)


# %%
from datetime import timedelta


def frontier_nodes_to_max_walltime(nodes: int) -> timedelta:
    if 1 <= nodes <= 91:
        return timedelta(hours=2.0)
    elif 92 <= nodes <= 183:
        return timedelta(hours=6.0)
    else:
        return timedelta(hours=12.0)


def compute_cpus_per_task(
    configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]],
):
    # Max `num_workers` + 1 for the main process
    max_num_workers = max(config.num_workers for config, _ in configs)
    return max_num_workers + 1


nodes_list = [1, 8, 64, 128]
nodes_list = [1, 8]

commands: list[str] = []
for nodes in nodes_list:
    configs_copy: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []
    for config, model_cls in configs:
        config_copy = config.clone()
        config_copy.name_parts.append(f"nodes_{nodes}")
        configs_copy.append((config_copy, model_cls))

    runner = ll.Runner(run)
    commands.append(
        runner.submit_slurm(
            configs_copy,
            snapshot=True,
            account="mat265",
            partition="batch",
            # qos="debug",
            nodes=nodes,
            tasks_per_node=8,  # frontier has 8 GPUs per node
            cpus_per_task=compute_cpus_per_task(configs_copy),
            walltime=frontier_nodes_to_max_walltime(nodes),
            setup_commands=setup_commands,
            name=f"graphormer_n{nodes}",
            print_command=False,
        ).command
    )

print("; ".join(commands))
