# %%
from pathlib import Path
from typing import Literal

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
    config.batch_size = 32
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
    config.batch_size = 3
    config.name_parts.append("jmp_l")


def create_config(*, grad: bool):
    config = M.MatbenchDiscoveryConfig.draft()
    config.project = "jmp_mptrj"
    config.name = "mptrj"
    jmp_s_(config)

    def dataset_fn(split: Literal["train", "val", "test"]):
        return base.FinetuneMPTrjHuggingfaceDatasetConfig(
            split=split, debug_repeat_largest_systems_for_testing=True
        )

    config.train_dataset = dataset_fn("train")
    config.val_dataset = dataset_fn("val")
    config.test_dataset = dataset_fn("test")

    config.primary_metric = ll.PrimaryMetricConfig(
        name="matbench_discovery/force_mae", mode="min"
    )

    if grad:
        config.energy_forces_config_(
            gradient=True,
            energy_coefficient=1.0,
            energy_pooling="mean",
            force_coefficient=1.0,
            force_loss=loss.MACEHuberLossConfig(delta=0.01),
            energy_loss=loss.HuberLossConfig(delta=0.01),
        )

        config.backbone.regress_forces = False
        config.backbone.direct_forces = False
        config.backbone.regress_energy = True

        config.tags.append("grad")
        config.name_parts.append("grad")
        config.trainer.precision = "16-mixed-auto"

        config.batch_size = 12

    else:
        config.energy_forces_config_(
            gradient=False,
            energy_coefficient=1.0,
            energy_pooling="mean",
            force_coefficient=1.0,
            force_loss=loss.MACEHuberLossConfig(delta=0.01),
            energy_loss=loss.HuberLossConfig(delta=0.01),
        )

        config.backbone.regress_forces = True
        config.backbone.direct_forces = True
        config.backbone.regress_energy = True

        config.tags.append("nograd")
        config.name_parts.append("nograd")
        config.trainer.precision = "16-mixed-auto"

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

    # Set data config
    config.num_workers = 4
    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False

    config.with_project_root_(project_root)

    config.name_parts.append("maceconf")

    return config.finalize(), M.MatbenchDiscoveryModel


def ln_(config: FinetuneConfigBase):
    config.backbone.ln_per_layer = True
    config.backbone.scale_factor_to_ln = False


def debug_(config: FinetuneConfigBase):
    config.num_workers = 0


def make_configs(*, grad: bool):
    configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
    config, model_cls = create_config(grad=grad)
    ln_(config)
    # debug_(config)
    configs.append((config, model_cls))
    return configs


configs_nograd = make_configs(grad=False)
configs_grad = make_configs(grad=True)


# %%
def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    model = model_cls.construct_and_load_checkpoint(config)
    trainer = ll.Trainer(config)
    trainer.fit(model)


# # %%
# runner = ll.Runner(run)
# runner.fast_dev_run(configs, n_batches=256)

# # %%
# runner = ll.Runner(run)
# runner.local(configs, env={"CUDA_VISIBLE_DEVICES": "0"})


# %%
runner = ll.Runner(run)
runner.session(
    configs_grad,
    snapshot=True,
    env={
        "CUDA_VISIBLE_DEVICES": "0,1",
        "LL_DISABLE_TYPECHECKING": "1",
    },
)

# %%
runner = ll.Runner(run)
runner.session(
    configs_nograd,
    snapshot=True,
    env={
        "CUDA_VISIBLE_DEVICES": "2,3",
        "LL_DISABLE_TYPECHECKING": "1",
    },
)

# %%
