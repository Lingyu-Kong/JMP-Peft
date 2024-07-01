# %%
from pathlib import Path

import ll
from jmppeft.modules import loss
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune import base
from jmppeft.tasks.finetune import matbench_discovery as M
from jmppeft.tasks.finetune.base import (
    FinetuneConfigBase,
    FinetuneModelBase,
    RLPConfig,
    WarmupCosRLPConfig,
)
from jmppeft.utils.param_specific_util import make_parameter_specific_optimizer_config

project_root = Path("/net/csefiles/coc-fung-cluster/nima/shared/experiment-data/")
ckpt_path = Path("/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/")


def jmp_s_(config: FinetuneConfigBase):
    from jmppeft.configs.finetune.jmp_s import jmp_s_ft_config_

    jmp_s_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path / "jmp-s.pt", ema=True
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
    config.name_parts.append("jmp_s")


def jmp_l_(config: FinetuneConfigBase):
    from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_

    jmp_l_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path / "jmp-l.pt", ema=True
    )

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
    config.name_parts.append("jmp_l")


def create_config():
    config = M.MatbenchDiscoveryConfig.draft()
    config.project = "jmp_mptrj"
    config.name = "matbench_discovery-nograd"
    jmp_s_(config)

    config.train_dataset = base.FinetuneMPTrjHuggingfaceDatasetConfig(split="train")
    config.val_dataset = base.FinetuneMPTrjHuggingfaceDatasetConfig(split="val")
    config.test_dataset = base.FinetuneMPTrjHuggingfaceDatasetConfig(split="test")

    config.primary_metric = ll.PrimaryMetricConfig(
        name="matbench_discovery/force_mae", mode="min"
    )

    config.energy_forces_config_(
        gradient=False,
        energy_coefficient=1.0,
        energy_pooling="mean",
        force_coefficient=1.0,
        force_loss=loss.MACEHuberLossConfig(delta=0.01),
        energy_loss=loss.HuberLossConfig(delta=0.01),
    )
    config.trainer.optimizer.gradient_clipping = ll.GradientClippingConfig(
        value=5.0,
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

    config.with_project_root_(project_root)

    config.name += "_mace"

    return config.finalize(), M.MatbenchDiscoveryModel


def ln_(config: FinetuneConfigBase):
    config.backbone.ln_per_layer = True
    config.backbone.scale_factor_to_ln = False


def debug_(config: FinetuneConfigBase):
    config.num_workers = 0


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
config, model_cls = create_config()
ln_(config)
# debug_(config)
configs.append((config, model_cls))


# %%
def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    model = model_cls.construct_and_load_checkpoint(config)
    trainer = ll.Trainer(config)
    trainer.fit(model)


# %%
runner = ll.Runner(run)
runner.fast_dev_run(configs)

# %%
runner = ll.Runner(run)
runner.local(configs, env={"CUDA_VISIBLE_DEVICES": "0"})


# %%
runner = ll.Runner(run)
runner.session(
    configs,
    snapshot=True,
    # gpus=[(0, 2, 3, 4, 5, 6)],
    env={
        "CUDA_VISIBLE_DEVICES": "1,2,3,5",
        "LL_DISABLE_TYPECHECKING": "1",
        # "NCCL_DEBUG": "TRACE",
        # "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        # "TORCH_CPP_LOG_LEVEL": "INFO",
    },
)
