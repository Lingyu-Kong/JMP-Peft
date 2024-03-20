# %%
from pathlib import Path
from typing import Any

from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.qm9 import jmp_l_qm9_config_
from jmppeft.modules.lora import LoraRootConfig
from jmppeft.tasks.finetune.base import (
    FinetuneConfigBase,
    FinetuneModelBase,
)
from jmppeft.tasks.finetune.qm9 import QM9Config, QM9Model, QM9Target
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
        enabled_by_default=True,
        r=r,
        children=children,
        alpha=alpha,
        bias="all",
        use_rslora=True,
        add_bias_to_lora_linear=True,
    )


# ckpt_path = Path(
#     "/global/cfs/cdirs/m3641/Nima/jmp/checkpoints/fm_gnoc_large_2_epoch.ckpt"
# )
# base_path = Path("/global/cfs/cdirs/m3641/Nima/jmp/datasets/qm9/")

ckpt_path = Path("/mnt/shared/checkpoints/fm_gnoc_large_2_epoch.ckpt")
base_path = Path("/mnt/shared/datasets/qm9/")


def create_config(
    target: QM9Target,
    lora: bool,
    lora_lr: float = 2.0e-4,
):
    config = QM9Config.draft()
    config.project = "jmp_peft_nersc"
    config.name = f"qm9-{target}"
    jmp_l_ft_config_(config, ckpt_path, ema_backbone=True, use_bf16=True)
    jmp_l_qm9_config_(config, target, base_path)

    config.batch_size = 32

    lora_lr_scale = None
    if lora:
        lora_config_(config, r=8, alpha=16, filter_children=False)
        config.name += "-lora"
        lora_lr_scale = lora_lr / config.optimizer.lr
    else:
        config.lora = None
        config.name += "-nolora"

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
        lora_lr_scale=lora_lr_scale,
    )

    return config.finalize(), QM9Model


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
configs.append(create_config("eps_HOMO", True))
# configs.append(create_config("eps_HOMO", False))

# %%
from jmppeft.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
)
from ll import Runner, Trainer


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    if (ckpt_path := config.meta.get("ckpt_path")) is None:
        raise ValueError("No checkpoint path provided")

    model = model_cls(config)

    # Load the checkpoint
    state_dict = retreive_state_dict_for_finetuning(
        ckpt_path, load_emas=config.meta.get("ema_backbone", False)
    )
    embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
    backbone = filter_state_dict(state_dict, "backbone.")
    model.load_backbone_state_dict(backbone=backbone, embedding=embedding, strict=True)

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
    num_jobs_per_gpu=1,
    # prologue=["module load conda/Mambaforge-23.1.0-1"],
    env={"LL_DISABLE_TYPECHECKING": "1"},
)

# %%
