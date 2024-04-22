# %%
from pathlib import Path

import ll
from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_l_matbench_discovery_config_
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
from jmppeft.utils.gradient_checkpointing import GradientCheckpointingConfig
from jmppeft.utils.param_specific_util import (
    make_parameter_specific_optimizer_config,
)

ckpt_path = Path("/mnt/shared/checkpoints/fm_gnoc_large_2_epoch.ckpt")
base_path = Path("/mnt/datasets/matbench-discovery-traj/megnet-133k-npz/")


def create_config(gradient_forces: bool):
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery"
    jmp_l_ft_config_(config, ckpt_path, ema_backbone=True, use_bf16=True)
    jmp_l_matbench_discovery_config_(
        config,
        base_path,
        use_megnet_133k=True,
        use_linref=True,
    )
    config.forces_config_(gradient=gradient_forces, coefficient=1.0)

    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    config.name += "_fc1"

    if gradient_forces:
        config.trainer.precision = "32-true"
        config.tags.append("gradient_forces")
        config.name += "_gradient_forces"

        config.backbone.regress_forces = False
        config.backbone.direct_forces = False
        config.backbone.regress_energy = True

        config.batch_size = 1
        config.gradient_checkpointing = GradientCheckpointingConfig(
            checkpoint_early_stop=False,
        )
    else:
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


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
# config, model_cls = create_config(gradient_forces=True)
config, model_cls = create_config(gradient_forces=False)
config.meta["resume_ckpt_path"] = next(
    Path(
        "/workspaces/repositories/jmp-peft/lightning_logs/dc5rlskx/jmp_peft_nersc/dc5rlskx/checkpoints/"
    ).glob("*.ckpt")
)
ddp_(config, use_balanced_batch_sampler=True, batch_size=2)
# debug_high_loss_(config)
# ^ act path: /workspaces/repositories/jmp-peft/config/finetune/jmp_l/lltrainer/mcezpn6d/activation


configs.append((config, model_cls))

# %%
from jmppeft.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
)
from ll import Runner, Trainer


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    if (resume_ckpt_path := config.meta.get("resume_ckpt_path")) is not None:
        model = model_cls.load_from_checkpoint(
            resume_ckpt_path,
            strict=True,
            hparams=config,
        )
    elif (ckpt_path := config.meta.get("ckpt_path")) is not None:
        model = model_cls(config)

        # Load the checkpoint
        state_dict = retreive_state_dict_for_finetuning(
            ckpt_path, load_emas=config.meta.get("ema_backbone", False)
        )
        embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
        backbone = filter_state_dict(state_dict, "backbone.")
        model.load_backbone_state_dict(
            backbone=backbone,
            embedding=embedding,
            strict=True,
        )
    else:
        model = model_cls(config)

    trainer = Trainer(config)
    trainer.fit(model, ckpt_path=resume_ckpt_path)


# %%
runner = Runner(run)
runner.local(configs)

# %%
runner = Runner(run)
runner.fast_dev_run(configs)


# %%
runner = Runner(run)
runner.local_session_per_gpu(
    configs,
    snapshot=True,
    gpus=[(0, 1)],
    # prologue=["module load conda/Mambaforge-23.1.0-1"],
    env={"LL_DISABLE_TYPECHECKING": "1"},
)
