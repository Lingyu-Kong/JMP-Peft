# %%
from pathlib import Path

import ll
from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_matbench_discovery_config_
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune.base import (
    BatchDumpConfig,
    FinetuneConfigBase,
    FinetuneModelBase,
)
from jmppeft.tasks.finetune.matbench_discovery import (
    MatbenchDiscoveryConfig,
    MatbenchDiscoveryModel,
)
from jmppeft.utils.param_specific_util import (
    make_parameter_specific_optimizer_config,
)

ckpt_path = Path("/mnt/shared/checkpoints/fm_gnoc_large_2_epoch.ckpt")
base_path = Path("/mnt/datasets/matbench-discovery-traj/megnet-133k-npz/")


def create_config():
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery"
    jmp_l_ft_config_(config, ckpt_path, ema_backbone=True, use_bf16=True)
    jmp_matbench_discovery_config_(
        config,
        base_path,
        use_megnet_133k=True,
        use_linref=False,
    )
    config.energy_forces_config_(gradient=False)

    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    config.name += "_fc1"

    config.tags.append("direct_forces")
    config.name += "_direct_forces"

    config.batch_size = 2
    config.gradient_checkpointing = None

    config.backbone.regress_forces = True
    config.backbone.direct_forces = True
    config.backbone.regress_energy = False

    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
        config,
        config.backbone.num_blocks,
        {
            "embedding": 0.3,
            "blocks_0": 0.55,
            "blocks_1": 0.40,
            "blocks_2": 0.30,
            "blocks_3": 0.40,
            "blocks_4": 0.55,
            "blocks_5": 0.625,
        },
    )

    return config.finalize(), MatbenchDiscoveryModel


def ddp_(
    config: FinetuneConfigBase,
    *,
    use_balanced_batch_sampler: bool = True,
    batch_size: int | None = None,
):
    config.trainer.strategy = "ddp_find_unused_parameters_true"
    config.use_balanced_batch_sampler = use_balanced_batch_sampler
    if use_balanced_batch_sampler:
        config.trainer.use_distributed_sampler = False
    if batch_size is not None:
        config.batch_size = batch_size


def debug_high_loss_(config: FinetuneConfigBase):
    config.trainer.actsave = ll.model.ActSaveConfig()
    config.batch_dump = BatchDumpConfig(dump_if_loss_gt=2.5)
    config.trainer.devices = (0,)
    if config.trainer.logging.wandb:
        config.trainer.logging.wandb.disable_()

    config.name += "_debug_high_loss"


def ln_(config: FinetuneConfigBase):
    config.backbone.ln_per_layer = True
    config.backbone.scale_factor_to_ln = True


def mace_(config: MatbenchDiscoveryConfig):
    config.sanity_check_mace = True
    config.batch_size = 1
    config.trainer.inference_mode = False
    config.normalization = {}
    config.shuffle_val = True
    config.trainer.precision = "32-true"

    config.name = "mace-sanity-check"


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
# config, model_cls = create_config(gradient_forces=True)
config, model_cls = create_config()
# ln_(config)
mace_(config)
# config.meta["resume_ckpt_path"] = Path("/mnt/checkpoints/mpd.ckpt")
# ddp_(config, use_balanced_batch_sampler=False, batch_size=1)
# debug_high_loss_(config)
# ^ act path: /workspaces/repositories/jmp-peft/config/finetune/jmp_l/lltrainer/mcezpn6d/activation


configs.append((config, model_cls))

# %%
from ll import Runner, Trainer


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    model = model_cls(config)
    trainer = Trainer(config)
    trainer.validate(model)


# %%
runner = Runner(run)
runner.fast_dev_run(configs, n_batches=1024)


# %%
runner = Runner(run)
runner.local(configs)
