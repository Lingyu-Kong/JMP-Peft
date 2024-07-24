# %%
from pathlib import Path

import nshtrainer.ll as ll
from jmppeft.configs.finetune.jmp_ import jmp_l_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_matbench_discovery_config_
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmppeft.tasks.finetune.matbench_discovery import (
    MatbenchDiscoveryConfig,
    MatbenchDiscoveryModel,
)
from jmppeft.utils.param_specific_util import make_parameter_specific_optimizer_config

project_root = Path("/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft/")

ckpt_path = Path("/lustre/orion/mat265/world-shared/nimashoghi/checkpoints/jmp-l.pt")
dataset_base_path = Path(
    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/mptraj/"
)


def create_config():
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery-nograd"
    jmp_l_ft_config_(config)
    jmp_matbench_discovery_config_(
        config,
        dataset_base_path,
        use_megnet_133k=True,
        use_linref=True,
    )
    config.energy_forces_config_(
        gradient=False,
        energy_coefficient=0.05,
        force_coefficient=1.0,
    )

    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    config.tags.append("direct_forces")
    config.name += "_direct_forces"
    config.trainer.precision = "16-mixed-auto"

    # Set data config
    config.batch_size = 10
    config.num_workers = 8
    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False

    config.backbone.regress_forces = True
    config.backbone.direct_forces = True
    config.backbone.regress_energy = True

    config.meta["ft_ckpt_path"] = ckpt_path

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

    config.with_project_root_(project_root)

    if (wandb_config := config.trainer.logging.wandb) is not None:
        wandb_config.disable_()

    return config.finalize(), MatbenchDiscoveryModel


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
config, model_cls = create_config()
configs.append((config, model_cls))


# %%
def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    if (ft_ckpt_path := config.meta.get("ft_ckpt_path")) is not None:
        model = model_cls.load_from_checkpoint(
            ft_ckpt_path, strict=False, hparams=config
        )
    else:
        model = model_cls(config)

    trainer = ll.Trainer(config)
    trainer.fit(model)


# %%
# runner = ll.Runner(run)
# runner.fast_dev_run(configs)

# %%
runner = ll.Runner(run)
runner.local(configs)


# # %%
# runner = ll.Runner(run)
# runner.local_sessions(configs, sessions=1, env={"CUDA_VISIBLE_DEVICES": "0"})
