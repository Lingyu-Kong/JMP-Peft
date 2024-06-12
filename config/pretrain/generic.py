# %%
import itertools
from pathlib import Path
from typing import Literal

import ll
from jmppeft.configs.pretrain.tasks import (
    tasks_config_frontier_,
    tasks_config_perlmutter_,
)
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.pretrain import module as M

perlmutter = True
if perlmutter:
    PROJECT_ROOT = Path("/global/cfs/cdirs/m3641/Nima/projdata/jmp-pretrain")
    setup_commands = []
    env = {"LL_DISABLE_TYPECHECKING": "1"}
else:
    PROJECT_ROOT = Path(
        "/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft-realruns-6_10"
    )
    PROJECT_ROOT.mkdir(exist_ok=True, parents=True)

    setup_commands = [
        "source /lustre/orion/mat265/world-shared/nimashoghi/repositories/jmp-peft/rocm53.sh"
    ]
    env = {"LL_DISABLE_TYPECHECKING": "1"}


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

    if not perlmutter and config.trainer.logging.wandb:
        config.trainer.logging.wandb.offline = True


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
    config.name_parts.extend(("goc", variant))
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
    config.generate_graphs_on_gpu = True


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
    config.trainer.callbacks.append(ll.callbacks.ThroughputMonitorConfig())
    config.perf_metrics = True

    config.trainer.profiler = ll.model.config.PyTorchProfilerConfig(emit_nvtx=True)

    config.trainer.max_steps = 1000
    config.lr_scheduler = None


def no_metrics_config_(config: M.PretrainConfig):
    config.disable_metrics = True
    config.log_task_losses = False


configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []

variants = ("base", "large", "xl")
variants = ("base",)
backbone_config_fns = (graphormer_backbone_config_, goc_backbone_config_)
backbone_config_fns = (graphormer_backbone_config_,)

for variant, backbone_config_ in itertools.product(variants, backbone_config_fns):
    # Testing
    # if (variant, backbone_config_) != ("base", graphormer_backbone_config_):
    #     continue

    config = M.PretrainConfig.draft()
    base_config_(config)
    if perlmutter:
        tasks_config_perlmutter_(config)
    else:
        tasks_config_frontier_(config)
    # graphormer_backbone_config_(config, variant)
    backbone_config_(config, variant)
    # fsdp_config_(config)
    profiling_config_(config)
    config.with_project_root_(PROJECT_ROOT)
    config.project = "jmp-pretrain"
    if perlmutter:
        config.project += "-perlmutter"
    else:
        config.project += "-frontier"
    config.project += "6_10"

    if variant == "base":
        config.batch_size = 6
    elif variant == "large":
        config.batch_size = 3
    elif variant == "xl":
        config.batch_size = 3
    gradient_checkpointing_config_(config)
    no_metrics_config_(config)

    config.num_workers = 4
    config = config.finalize()
    configs.append((config, M.PretrainModel))

print(len(configs))


# %%
def run(config: M.PretrainConfig, model_cls: type[M.PretrainModel]):
    model = model_cls(config)

    trainer = ll.Trainer(config, **model.fsdp_trainer_kwargs())
    trainer.fit(model)


def nsys_python_command_prefix(
    configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]],
) -> str:
    assert len(configs) == 1, "Only one config is supported"
    config, _ = configs[0]
    profiler_directory = config.directory.resolve_subdirectory(config.id, "profiler")

    return rf"nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o {str(profiler_directory.absolute())}/nsight_report.${{SLURM_PROCID}}.${{SLURM_JOB_ID}} -f true -x true "


# %%

runner = ll.Runner(run)
runner.fast_dev_run(configs)

# %%
runner = ll.Runner(run)
runner.fast_dev_run_session(
    configs,
    n_batches=128,
    setup_commands=setup_commands,
    env=env,
    python_command_prefix=nsys_python_command_prefix(configs),
)


# %%
from datetime import timedelta


def submit_frontier(
    runner: ll.Runner[M.PretrainConfig, None, type[M.PretrainModel]],
    configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]],
    debug: bool = False,
    *,
    nodes: int,
    tasks_per_node: int = 8,
    name: str | None = None,
    print_command: bool = False,
    setup_commands: list[str] = [],
    env: dict[str, str] = {},
    walltime: timedelta | None = None,
    python_command_prefix: str | None = None,
):
    def compute_cpus_per_task(
        configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]],
    ):
        # Max `num_workers` + 1 for the main process
        max_num_workers = max(config.num_workers for config, _ in configs)
        return max_num_workers + 1

    def nodes_to_max_walltime(nodes: int) -> timedelta:
        if 1 <= nodes <= 91:
            return timedelta(hours=2.0)
        elif 92 <= nodes <= 183:
            return timedelta(hours=6.0)
        else:
            return timedelta(hours=12.0)

    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if debug:
        kwargs["qos"] = "debug"

    if walltime is None:
        walltime = nodes_to_max_walltime(nodes)
    return runner.submit_slurm(
        configs,
        snapshot=True,
        account="mat265",
        partition="batch",
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=compute_cpus_per_task(configs_copy),
        walltime=walltime,
        setup_commands=setup_commands,
        env=env,
        print_command=print_command,
        python_command_prefix=python_command_prefix,
        **kwargs,
    )


def submit_perlmutter(
    runner: ll.Runner[M.PretrainConfig, None, type[M.PretrainModel]],
    configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]],
    debug: bool = False,
    *,
    nodes: int,
    tasks_per_node: int = 4,
    name: str | None = None,
    print_command: bool = False,
    setup_commands: list[str] = [],
    env: dict[str, str] = {},
    walltime: timedelta | None = None,
    python_command_prefix: str | None = None,
):
    def compute_cpus_per_task(
        configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]],
    ):
        # Max `num_workers` + 1 for the main process
        max_num_workers = max(config.num_workers for config, _ in configs)
        return max_num_workers + 1

    kwargs = {}
    if name is not None:
        kwargs["name"] = name

    if walltime is None:
        walltime = timedelta(hours=24.0) if not debug else timedelta(hours=0.5)

    return runner.submit_slurm(
        configs,
        snapshot=True,
        account="m3641_g",
        qos="preempt" if not debug else "debug_preempt",
        constraint="gpu",
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=compute_cpus_per_task(configs_copy),
        gpus_per_task=1,  # Perlmutter has 4 GPUs per node
        walltime=walltime,
        setup_commands=setup_commands,
        env=env,
        print_command=print_command,
        python_command_prefix=python_command_prefix,
        **kwargs,
    )


submit = submit_perlmutter if perlmutter else submit_frontier

# nodes_list = [1, 8, 64, 128]
# nodes_list = [128]
# nodes_list = [1]
gpus_list = [4, 8, 64]

assert all(gpus % 4 == 0 for gpus in gpus_list)
nodes_list = [gpus // 4 for gpus in gpus_list]

commands: list[str] = []
for nodes in nodes_list:
    configs_copy: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []
    for config, model_cls in configs:
        config_copy = config.clone()
        config_copy.name_parts.append(f"nodes_{nodes}")
        configs_copy.append((config_copy, model_cls))

    runner = ll.Runner(run)
    commands.append(
        submit(
            runner,
            configs_copy,
            nodes=nodes,
            setup_commands=setup_commands,
            env=env,
            print_command=False,
            name=f"jmp_n{nodes}",
            # debug=True,
            walltime=timedelta(hours=2.0),
            python_command_prefix=nsys_python_command_prefix(configs_copy),
        ).command
    )

print("; ".join(commands))
