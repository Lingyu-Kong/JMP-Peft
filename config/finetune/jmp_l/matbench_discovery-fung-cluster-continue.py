# %%
from pathlib import Path

import ll
from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_l_matbench_discovery_config_
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmppeft.tasks.finetune.matbench_discovery import (
    MatbenchDiscoveryConfig,
    MatbenchDiscoveryModel,
)
from jmppeft.utils.param_specific_util import make_parameter_specific_optimizer_config

project_root = Path("/nimahome/experiment-data/")

ckpt_path = Path(
    "/net/csefiles/coc-fung-cluster/nima/experiment-data/lltrainer/5xrq84ta/checkpoint/latest_epoch16_step119663.ckpt"
)
dataset_base_path = Path("/nimahome/datasets/matbench_discovery/")


def create_config():
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery-nograd"
    jmp_l_ft_config_(config)
    jmp_l_matbench_discovery_config_(
        config,
        dataset_base_path,
        use_megnet_133k=True,
        use_linref=True,
    )
    config.energy_forces_config_(
        gradient=False,
        energy_coefficient=0.01,
        force_coefficient=1.0,
    )

    config.optimizer = AdamWConfig(
        lr=2.0e-5,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    config.tags.append("direct_forces")
    config.name += "_direct_forces"
    config.trainer.precision = "16-mixed-auto"

    # Set data config
    config.batch_size = 3
    config.num_workers = 2
    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False

    config.backbone.regress_forces = True
    config.backbone.direct_forces = True
    config.backbone.regress_energy = True

    config.meta["continue_ckpt_path"] = ckpt_path
    config.name += "_continue"

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

    return config.finalize(), MatbenchDiscoveryModel


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
config, model_cls = create_config()
configs.append((config, model_cls))


# %%
def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    if (ckpt_path := config.meta.get("continue_ckpt_path")) is None:
        raise ValueError("ckpt_path is not set in config")

    model = model_cls.load_from_checkpoint(ckpt_path, strict=True, hparams=config)
    trainer = ll.Trainer(config)
    trainer.fit(model, ckpt_path=ckpt_path)


# %%
runner = ll.Runner(run)
runner.fast_dev_run(configs)

# %%
runner = ll.Runner(run)
runner.local(configs, env={"CUDA_VISIBLE_DEVICES": "0"})


# %%
runner = ll.Runner(run)
runner.local_session_per_gpu(
    configs,
    snapshot=True,
    gpus=[(1, 2, 3, 4, 5, 6)],
    env={
        "LL_DISABLE_TYPECHECKING": "1",
        # "NCCL_DEBUG": "TRACE",
        # "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        # "TORCH_CPP_LOG_LEVEL": "INFO",
    },
)
