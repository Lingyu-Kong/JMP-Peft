# %%
from pathlib import Path
from typing import Any

import nshtrainer.ll as ll
from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_matbench_discovery_config_
from jmppeft.modules import dist_lora as dlora
from jmppeft.modules.lora import LoraRootConfig
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


def _flatten(config: dict[str, dict[str, Any]]):
    return {
        f"{out_key}.{inner_key}": inner_value
        for out_key, outer_value in config.items()
        for inner_key, inner_value in outer_value.items()
    }


def lora_config_(
    config: FinetuneConfigBase,
    *,
    r: int,
    alpha: int,
    filter_children: bool,
):
    int_block_configs = {
        f"int_blocks_{idx}": {
            "dense_ca": {"enabled": True},
            **_flatten(
                {
                    "trip_interaction": {
                        "dense_ba": {"enabled": True},
                        "down_projection": {"enabled": True},
                        "up_projection_ca": {"enabled": True},
                        "up_projection_ac": {"enabled": True},
                    },
                    "quad_interaction": {
                        "dense_db": {"enabled": True},
                        "down_projection": {"enabled": True},
                        "up_projection_ca": {"enabled": True},
                        "up_projection_ac": {"enabled": True},
                    },
                    "atom_edge_interaction": {
                        "dense_ba": {"enabled": True},
                        "down_projection": {"enabled": True},
                        "up_projection_ca": {"enabled": True},
                        "up_projection_ac": {"enabled": True},
                    },
                    "edge_atom_interaction": {
                        "dense_ba": {"enabled": True},
                        "down_projection": {"enabled": True},
                        "up_projection_ca": {"enabled": True},
                    },
                    "atom_interaction": {
                        "bilinear": {"enabled": True},
                        "down_projection": {"enabled": True},
                        "up_projection": {"enabled": True},
                    },
                }
            ),
        }
        for idx in range(config.backbone.num_blocks + 1)
    }
    out_block_configs = {
        f"out_blocks_{idx}": _flatten(
            {
                "atom_update_block": {
                    "layers": {"enabled": True},
                },
            }
        )
        for idx in range(config.backbone.num_blocks + 1)
    }
    children = (
        {
            "bases": {"enabled": True},
            "out_mlp_E": {"enabled": True},
            # Flatten the out_blocks (e.g., out_blocks_0.layers, out_blocks_1.layers, ...)
            # and int_blocks (e.g., int_blocks_0.trip_interaction.dense_ba, ...)
            **_flatten(out_block_configs),
            **_flatten(int_block_configs),
        }
        if filter_children
        else {}
    )

    config.lora = LoraRootConfig(
        enabled_by_default=not filter_children,
        r=r,
        children=children,
        alpha=alpha,
        bias="all",
        use_rslora=True,
        add_bias_to_lora_linear=True,
    )


ckpt_path = Path("/mnt/shared/checkpoints/fm_gnoc_large_2_epoch.ckpt")
base_path = Path("/mnt/datasets/matbench-discovery-traj/")

# ckpt_path = Path("/ccs/home/nimashoghi/proj-shared/nimashoghi/checkpoints/jmp-l.ckpt")
# base_path = Path(
#     "/ccs/home/nimashoghi/proj-shared/nimashoghi/datasets/matbench-trajectory"
# )


def create_config():
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery"
    jmp_l_ft_config_(config, ckpt_path, ema_backbone=True, use_bf16=True)
    jmp_matbench_discovery_config_(
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
    # config.trainer.precision = "32-true"

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
from nshtrainer import Runner, Trainer


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
