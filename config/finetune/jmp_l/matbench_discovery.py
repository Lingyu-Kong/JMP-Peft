# %%
from pathlib import Path

from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_l_matbench_discovery_config_
from jmppeft.tasks.finetune.base import (
    FinetuneConfigBase,
    FinetuneModelBase,
    WarmupCosRLPConfig,
)
from jmppeft.tasks.finetune.matbench_discovery import (
    MatbenchDiscoveryConfig,
    MatbenchDiscoveryModel,
)
from jmppeft.utils.gradient_checkpointing import GradientCheckpointingConfig
from jmppeft.utils.param_specific_util import (
    make_parameter_specific_optimizer_config,
)

ckpt_path = Path(
    "/global/cfs/cdirs/m3641/Nima/jmp/checkpoints/fm_gnoc_large_2_epoch.ckpt"
)
base_path = Path(
    "/global/cfs/cdirs/m3641/Nima/jmp/datasets/matbench-discovery-megnet-133k/"
)


def create_config():
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery"
    jmp_l_ft_config_(config, ckpt_path, ema_backbone=True, use_bf16=True)
    jmp_l_matbench_discovery_config_(
        config,
        base_path,
        use_megnet_133k=True,
        use_linref=True,
        gradient_forces=True,
        force_coefficient=10.0,
    )
    config.backbone.regress_forces = False
    config.backbone.direct_forces = False

    assert isinstance(config.lr_scheduler, WarmupCosRLPConfig)
    config.lr_scheduler.warmup_epochs = 0.1
    config.lr_scheduler.max_epochs = 1

    config.batch_size = 1
    config.gradient_checkpointing = GradientCheckpointingConfig(
        checkpoint_early_stop=False,
    )
    config.trainer.precision = "32-true"

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

    # config.trainer.strategy = "ddp_find_unused_parameters_true"

    return config.finalize(), MatbenchDiscoveryModel


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
config, model_cls = create_config()
# # config.meta["resume_ckpt_path"] = ()
# del config.meta["ckpt_path"]
# config.trainer.ckpt_path = "/workspaces/repositories/jmp-peft/lightning_logs/m742ekcy/on_exception_m742ekcy.ckpt"

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
    trainer.fit(model)


# %%
runner = Runner(run)
runner.fast_dev_run(configs)

# %%
runner = Runner(run)
runner.local_session_per_gpu(
    configs,
    snapshot=True,
    gpus=[1],
    # prologue=["module load conda/Mambaforge-23.1.0-1"],
    env={"LL_DISABLE_TYPECHECKING": "1"},
)
