# %%
import os
from pathlib import Path
from typing import Literal

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import ll
import rich
from jmppeft.configs.finetune.jmp_s import jmp_s_ft_config_
from jmppeft.modules import loss
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune import base, output_head
from jmppeft.tasks.finetune import matbench_discovery as M
from jmppeft.utils.param_specific_util import make_parameter_specific_optimizer_config

project_root = Path("/net/csefiles/coc-fung-cluster/nima/shared/experiment-data/")
ckpt_path = Path("/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/")


def jmp_s_(config: base.FinetuneConfigBase):
    jmp_s_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path / "jmp-s.pt", ema=True
    )

    config.name_parts.append("jmp_s")


def parameter_specific_optimizers_(config: base.FinetuneConfigBase):
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


def direct_(config: base.FinetuneConfigBase):
    config.backbone.regress_forces = True
    config.backbone.direct_forces = True
    config.backbone.regress_energy = True
    config.name_parts.append("direct")


def grad_(config: base.FinetuneConfigBase):
    config.backbone.regress_forces = False
    config.backbone.direct_forces = False
    config.backbone.regress_energy = True

    config.trainer.inference_mode = False

    config.name_parts.append("grad")


def ln_(config: base.FinetuneConfigBase):
    config.backbone.ln_per_layer = True
    config.backbone.scale_factor_to_ln = False


def create_config():
    config = M.MatbenchDiscoveryConfig.draft()
    config.project = "jmp_mptrj"
    config.name = "mptrj"
    jmp_s_(config)
    ln_(config)
    config.backbone.qint_tags = [0, 1, 2]

    def dataset_fn(split: Literal["train", "val", "test"]):
        return base.FinetuneMPTrjHuggingfaceDatasetConfig(
            split=split,
            debug_repeat_largest_systems_for_testing=False,
        )

    config.train_dataset = dataset_fn("train")
    config.val_dataset = dataset_fn("val")
    config.test_dataset = dataset_fn("test")

    config.primary_metric = ll.PrimaryMetricConfig(
        name="matbench_discovery/force_mae", mode="min"
    )
    config.trainer.optimizer.gradient_clipping = ll.GradientClippingConfig(
        value=2.0,
        algorithm="value",
    )

    config.optimizer = AdamWConfig(
        lr=2.0e-5,
        amsgrad=False,
        betas=(0.9, 0.95),
    )
    config.lr_scheduler = base.WarmupCosRLPConfig(
        warmup_epochs=5,
        warmup_start_lr_factor=1.0e-1,
        should_restart=False,
        max_epochs=32,
        min_lr_factor=0.5,
        rlp=base.RLPConfig(patience=3, factor=0.8),
    )

    # Set data config
    config.num_workers = 4

    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False

    config.trainer.precision = "16-mixed-auto"
    config.trainer.set_float32_matmul_precision = "medium"

    config.with_project_root_(project_root)
    return config


configs: list[tuple[M.MatbenchDiscoveryConfig, type[M.MatbenchDiscoveryModel]]] = []

# region direct, energy+force+stress
config = create_config()
ln_(config)
direct_(config)
# Energy head
config.graph_targets.append(
    output_head.ReferencedScalarTargetConfig(
        name="y",
        loss_coefficient=1.0,
        loss=loss.HuberLossConfig(delta=0.01),
        reduction="sum",
        max_atomic_number=config.backbone.num_elements,
        initialization=output_head.MPElementalReferenceInitializationConfig(),
        trainable_references=True,
    )
)
# Stress head
config.graph_targets.append(
    output_head.DirectStressTargetConfig(
        name="stress",
        loss_coefficient=100.0,
        loss=loss.HuberLossConfig(delta=0.01),
        reduction="mean",
    )
)
# Force head
config.node_targets.append(
    output_head.NodeVectorTargetConfig(
        name="force",
        loss_coefficient=10.0,
        loss=loss.MACEHuberLossConfig(delta=0.01),
        reduction="sum",
    )
)
config.batch_size = 64
config.name_parts.append("bsz64")
config.lr_scheduler.max_epochs = 64

parameter_specific_optimizers_(config)
config = config.finalize()
configs.append((config, M.MatbenchDiscoveryModel))
# endregion

# region grad, energy+force+stress
config = create_config()
ln_(config)
grad_(config)
# Energy head
config.graph_targets.append(
    output_head.ReferencedScalarTargetConfig(
        name="y",
        loss_coefficient=1.0,
        loss=loss.HuberLossConfig(delta=0.01),
        reduction="sum",
        max_atomic_number=config.backbone.num_elements,
        initialization=output_head.MPElementalReferenceInitializationConfig(),
        trainable_references=True,
    )
)
# Stress head
config.graph_targets.append(
    output_head.GradientStressTargetConfig(
        name="stress",
        energy_name="y",
        loss_coefficient=100.0,
        loss=loss.HuberLossConfig(delta=0.01),
        reduction="mean",
        forces=True,  # Computes forces and stress using 1 single "torch.autograd.grad"
    )
)
# Force head
config.node_targets.append(
    output_head.GradientForcesTargetConfig(
        name="force",
        energy_name="y",
        loss_coefficient=10.0,
        loss=loss.MACEHuberLossConfig(delta=0.01),
        use_stress_forces=True,  # Uses the force computed by the stress head
    )
)
config.batch_size = 8
config.trainer.precision = "32-true"
config.trainer.set_float32_matmul_precision = "high"
config.name_parts.append("bsz8")

parameter_specific_optimizers_(config)
config = config.finalize()
configs.append((config, M.MatbenchDiscoveryModel))
# endregion

rich.print(configs)


# %%
def run(
    config: M.MatbenchDiscoveryConfig, model_cls: type[M.MatbenchDiscoveryModel]
) -> None:
    model = model_cls.construct_and_load_checkpoint(config)
    trainer = ll.Trainer(config)
    trainer.fit(model)


# %%
runner = ll.Runner(run)
runner.fast_dev_run(configs)

# %%

for i, env in enumerate(
    [
        {"CUDA_VISIBLE_DEVICES": "0,1"},
        {"CUDA_VISIBLE_DEVICES": "2,3"},
    ]
):
    runner = ll.Runner(run)
    _ = runner.session(
        [configs[i]],
        snapshot=True,
        env={
            **env,
            "LL_DISABLE_TYPECHECKING": "1",
        },
    )
