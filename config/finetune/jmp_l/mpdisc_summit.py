# %%
from datetime import timedelta
from pathlib import Path

from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_l_matbench_discovery_config_
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmppeft.tasks.finetune.matbench_discovery import (
    MatbenchDiscoveryConfig,
    MatbenchDiscoveryModel,
)
from jmppeft.utils.param_specific_util import make_parameter_specific_optimizer_config

ckpt_path = Path("/ccs/home/nimashoghi/proj-shared/nimashoghi/checkpoints/mpd.ckpt")
base_path = Path(
    "/ccs/home/nimashoghi/proj-shared/nimashoghi/datasets/matbench-trajectory-m3gnet/"
)


def create_config():
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery-nograd"
    jmp_l_ft_config_(config, ckpt_path, ema_backbone=True, use_bf16=True)
    jmp_l_matbench_discovery_config_(
        config,
        base_path,
        use_megnet_133k=True,
        use_linref=True,
    )
    config.forces_config_(gradient=False, coefficient=1.0)

    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    config.tags.append("direct_forces")
    config.name += "_direct_forces"
    config.trainer.precision = "fp16-mixed"

    config.batch_size = 4

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

    # config.trainer.strategy = "ddp_find_unused_parameters_true"

    config.with_project_root_("/gpfs/alpine2/proj-shared/mat273/nimashoghi/jmp-peft/")

    if (wandb_config := config.trainer.logging.wandb) is not None:
        wandb_config.disable_()

    return config.finalize(), MatbenchDiscoveryModel


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
config, model_cls = create_config()
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
    trainer.validate(model)


# %%
runner = Runner(run)
runner.submit(
    configs,
    nodes=1,
    project="MAT273",
    queue="batch-hm",
    env={"LL_DISABLE_TYPECHECKING": "1"},
    walltime=timedelta(hours=24),
    # lsf_kwargs={"command_prefix": "jsrun -n1 -c42 -g6"},
)


# %%
# runner = Runner(run)
# runner.fast_dev_run(configs)

"""
# %%
runner = Runner(run)
runner.local_session_per_gpu(
    configs,
    snapshot=True,
    gpus=[1],
    # prologue=["module load conda/Mambaforge-23.1.0-1"],
    env={"LL_DISABLE_TYPECHECKING": "1"},
)
"""
