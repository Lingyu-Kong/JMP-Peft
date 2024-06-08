# %%
from pathlib import Path

import ll
from jmppeft.configs.finetune.jmp_s import jmp_s_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_matbench_discovery_config_
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune import base
from jmppeft.tasks.finetune.base import (
    BatchDumpConfig,
    FinetuneConfigBase,
    FinetuneModelBase,
    RLPConfig,
    WarmupCosRLPConfig,
)
from jmppeft.tasks.finetune.matbench_discovery import (
    MatbenchDiscoveryConfig,
    MatbenchDiscoveryModel,
)
from jmppeft.utils.param_specific_util import make_parameter_specific_optimizer_config

project_root = Path("/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft/")

ckpt_path = Path("/lustre/orion/mat265/world-shared/nimashoghi/checkpoints/jmp-s.pt")
dataset_base_path = Path(
    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/mptraj/"
)


def create_config():
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery-grad"
    jmp_s_ft_config_(config)
    jmp_matbench_discovery_config_(
        config,
        dataset_base_path,
        use_megnet_133k=True,
        use_linref=True,
    )
    config.conditional_max_neighbors = False
    config.energy_forces_config_(
        gradient=False,
        energy_coefficient=0.01,
        force_coefficient=1.0,
        force_loss="mae",
    )
    config.trainer.optimizer.gradient_clipping = ll.GradientClippingConfig(
        value=2.0,
        algorithm="norm",
    )

    config.optimizer = AdamWConfig(
        lr=2.0e-5,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=5,
        warmup_start_lr_factor=1.0e-1,
        should_restart=False,
        max_epochs=32,
        min_lr_factor=0.5,
        rlp=RLPConfig(patience=3, factor=0.8),
    )

    config.tags.append("grad_forces")
    config.name += "_grad_forces"
    config.trainer.precision = "16-mixed-auto"

    # Set data config
    config.batch_size = 6
    config.num_workers = 2

    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False

    config.backbone.regress_forces = True
    config.backbone.direct_forces = True
    config.backbone.regress_energy = True

    # config.meta["ft_ckpt_path"] = ckpt_path
    # config.meta["ckpt_path"] = ckpt_path
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path, ema=True
    )

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

    config.with_project_root_(project_root)

    config.name += "_fmae"

    if (wandb_config := config.trainer.logging.wandb) is not None:
        wandb_config.disable_()

    return config.finalize(), MatbenchDiscoveryModel


def debug_high_loss_(config: FinetuneConfigBase):
    # config.trainer.actsave = ll.model.ActSaveConfig()
    config.batch_dump = BatchDumpConfig(dump_if_loss_gt=2.5)
    # config.trainer.devices = (0,)
    # if config.trainer.logging.wandb:
    #     config.trainer.logging.wandb.disable_()

    config.name += "_debug_high_loss"


def ln_(config: FinetuneConfigBase):
    config.backbone.ln_per_layer = True
    config.backbone.scale_factor_to_ln = True


def debug_nans_(config: FinetuneConfigBase):
    config.trainer.detect_anomaly = True
    config.trainer.python_logging.lovely_tensors = True


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
config, model_cls = create_config()
debug_high_loss_(config)
ln_(config)
# debug_nans_(config)
configs.append((config, model_cls))


# %%
def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    model = model_cls.construct_and_load_checkpoint(config)

    trainer = ll.Trainer(config)
    trainer.fit(model)


# %%
runner = ll.Runner(run)
runner.session(
    configs,
    snapshot=False,
    env={"CUDA_VISIBLE_DEVICES": "0"},
    setup_commands=[
        "source /lustre/orion/mat265/world-shared/nimashoghi/repositories/jmp-peft/rocm60.sh"
    ],
)

# %%
from datetime import timedelta

runner = ll.Runner(run)
runner.submit_slurm(
    configs,
    snapshot=True,
    account="mat265",
    partition="batch",
    # qos="debug",
    nodes=2,
    tasks_per_node=8,  # frontier has 8 GPUs per node
    walltime=timedelta(hours=1),
    setup_commands=["source /ccs/home/nimashoghi/repositories/jmp-peft/rocm53.sh"],
)
