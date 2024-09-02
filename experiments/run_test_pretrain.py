"""
A reusable script to run pretraining experiments on JMP-PEFT
"""

import nshtrainer.ll as ll
from jmppeft.configs.pretrain.tasks import tasks_config_frontier_,tasks_config_oc20_4ktest_
from jmppeft.models.gemnet.config import BackboneConfig
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.pretrain import module as M
from pathlib import Path


"""
Set some Global Parameters for Pretraining
For most of the cases, we just need to change these Global Parameters
Change to parser parameters in the future, perhaps
"""
MODEL_TYPE = "gemnet_base" ## Choose from ["gemnet_base", "gemnet_large", "gemnet_xl","torchmd","graphormer"]
USE_FSDP = False ## Use Fully Sharded Data Parallelism
BATCH_SIZE = 24 ## Batch Size per gpu
NUM_WORKERS = 4 ## Number of Workers for DataLoader
GRAD_CLIP = 2.0 ## Gradient Clipping Value, 1.0 for torchmd, 2.0 for gemnet
# TODO: data path?


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
        value=GRAD_CLIP,
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
    config.batch_size = BATCH_SIZE
    config.num_workers = NUM_WORKERS

    # Set up the JMP MT dataset config and tasks
    config.mt_dataset = M.MTDatasetConfig(
        sample_type="temperature",
        sample_temperature=2.0,
    )
    

def backbone_config_(
    config: M.PretrainConfig,
):
    config.name_parts.append(MODEL_TYPE) ## Component of the exp name
    if MODEL_TYPE == "torchmd":
        config.dropout = None ## TODO:These two necessary? Why?
        config.edge_dropout = None ## TODO:These two necessary? Why?
        config.backbone = M.TorchMDNetBackboneConfig()
        # config.backbone.apply_1_5M_param_config_() ## Adjust model size
        config.exclude_keys.remove("edge_index") ## Torchmd needs edge_index
    elif MODEL_TYPE == "gemnet_base":
        config.backbone = M.GOCBackboneConfig.base()
        config.backbone.scale_basis = False # TODO: What is this for? List all possible settings here
    elif MODEL_TYPE == "gemnet_large":
        config.backbone = M.GOCBackboneConfig.large()
        config.backbone.scale_basis = False
    elif MODEL_TYPE == "gemnet_xl":
        config.backbone = M.GOCBackboneConfig.xl()
        config.backbone.scale_basis = False
    elif MODEL_TYPE == "graphormer":
        pass
    else:
        raise ValueError(f"Invalid Model Type: {MODEL_TYPE}")
    

# ## TODO:
# def gradient_checkpointing_config_(config: M.PretrainConfig):
#     ## Use gradient checkpointing for memory efficiency
#     config.name_parts.append("gc")
#     config.gradient_checkpointing = True
#     ## Don't use gradient checkpointing, don't use it
    

def fsdp_config_(config: M.PretrainConfig):
    ## Use fsdp
    config.fsdp = M.FSDPConfig(
        gradient_checkpointing=True,
    )
    ## Don't use fsdp, comment out
    

## TODO:
def profiling_config_(config: M.PretrainConfig):
    config.trainer.callbacks.append(ll.callbacks.EpochTimerConfig())
    config.trainer.callbacks.append(
        ll.callbacks.ThroughputMonitorConfig(batch_size=config.batch_size)
    )
    config.perf_metrics = True
    

configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []
config = M.PretrainConfig.draft()
base_config_(config)
# tasks_config_frontier_(config) ##TODO: We load data here
tasks_config_oc20_4ktest_(config, sample_seed=37, base_dir=Path("/nethome/lkong88/workspace/fairchem/src/fairchem/data/"))
backbone_config_(config)
# gradient_checkpointing_config_(config)
fsdp_config_(config)
# profiling_config_(config)
config = config.finalize()
configs.append((config, M.PretrainModel)) ## TODO:Match model type in M.PretrainModel


def run(config: M.PretrainConfig, model_cls: type[M.PretrainModel]):
    model = model_cls(config)
    trainer = ll.Trainer(config, **model.fsdp_trainer_kwargs())
    trainer.fit(model)


# setup_commands = [
#     "source /lustre/orion/mat265/world-shared/nimashoghi/repositories/jmp-peft/rocm53.sh"
# ] ## TODO:What is this for?


runner = ll.Runner(run)
runner.fast_dev_run_session(configs, snapshot=False, n_batches=128, env={
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NSHUTILS_DISABLE_TYPECHECKING": "0", ## for debug, 0
    },) ## snapshot=True, then we can change code without affecting the running jobs
    
# runner = ll.Runner(run)
# runner.session(configs, snapshot=False, env={
#         "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,6,7",
#         "NSHUTILS_DISABLE_TYPECHECKING": "1", ## for debug, 0
#     },) ## snapshot=True, then we can change code without affecting the running jobs


## TODO: Check if the following code can be used to run the pretrain on Frontier
# from datetime import timedelta

# def frontier_nodes_to_max_walltime(nodes: int) -> timedelta:
#     if 1 <= nodes <= 91:
#         return timedelta(hours=2.0)
#     elif 92 <= nodes <= 183:
#         return timedelta(hours=6.0)
#     else:
#         return timedelta(hours=12.0)


# def compute_cpus_per_task(
#     configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]],
# ):
#     # Max `num_workers` + 1 for the main process
#     max_num_workers = max(config.num_workers for config, _ in configs)
#     return max_num_workers + 1


# nodes_list = [1, 8, 64, 128]
# nodes_list = [1]

# commands: list[str] = []
# for nodes in nodes_list:
#     configs_copy: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []
#     for config, model_cls in configs:
#         config_copy = config.clone()
#         config_copy.name_parts.append(f"nodes_{nodes}")
#         configs_copy.append((config_copy, model_cls))

#     runner = ll.Runner(run)
#     commands.append(
#         runner.submit_slurm(
#             configs_copy,
#             snapshot=True,
#             account="mat265",
#             partition="batch",
#             nodes=nodes,
#             tasks_per_node=8,  # frontier has 8 GPUs per node
#             cpus_per_task=compute_cpus_per_task(configs_copy),
#             gpus_per_task=1,
#             walltime=frontier_nodes_to_max_walltime(nodes),
#             setup_commands=setup_commands,
#             name=f"goc_n{nodes}",
#             print_command=False,
#             qos="debug",
#         ).command
#     )

# print("; ".join(commands))