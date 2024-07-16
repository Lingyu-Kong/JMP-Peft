# %%
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import ll
import rich
from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.jmp_s import jmp_s_ft_config_
from jmppeft.modules import loss
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune import base, output_head
from jmppeft.tasks.finetune import matbench_discovery as M
from jmppeft.utils.param_specific_util import (
    make_parameter_specific_optimizer_config,
    parameter_specific_optimizer_config,
)

base_dir = Path("/gpfs/alpine2/proj-shared/mat273/nimashoghi/")
assert base_dir.exists() and base_dir.is_dir(), f"Base directory not found: {base_dir}"

project_root = base_dir / "experiment-data"
project_root.mkdir(exist_ok=True, parents=True)

ckpt_dir = base_dir / "checkpoints"
assert (
    ckpt_dir.exists() and ckpt_dir.is_dir()
), f"Checkpoints directory not found: {ckpt_dir}"


def jmp_s_(config: base.FinetuneConfigBase):
    ckpt_path = ckpt_dir / "jmp-s.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    jmp_s_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path, ema=True
    )

    config.meta["jmp_kind"] = "s"
    config.name_parts.append("jmps")


def jmp_l_(config: base.FinetuneConfigBase):
    ckpt_path = ckpt_dir / "jmp-l.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    jmp_l_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path, ema=True
    )

    config.meta["jmp_kind"] = "l"
    config.name_parts.append("jmpl")


def parameter_specific_optimizers_(config: base.FinetuneConfigBase):
    match config.meta["jmp_kind"]:
        case "l":
            config.parameter_specific_optimizers = (
                make_parameter_specific_optimizer_config(
                    config,
                    config.backbone.num_blocks,
                    {
                        "embedding": 0.5,
                        "blocks_0": 0.50,
                        "blocks_1": 0.60,
                        "blocks_2": 0.75,
                        "blocks_3": 0.825,
                        "blocks_4": 0.875,
                        "blocks_5": 0.90,
                    },
                )
            )
        case "s":
            config.parameter_specific_optimizers = (
                make_parameter_specific_optimizer_config(
                    config,
                    config.backbone.num_blocks,
                    {
                        "embedding": 0.65,
                        "blocks_0": 0.75,
                        "blocks_1": 0.825,
                        "blocks_2": 0.875,
                        "blocks_3": 0.90,
                    },
                )
            )
        case _:
            raise ValueError(f"Invalid jmp_kind: {config.meta['jmp_kind']}")


def parameter_specific_optimizers_energy_references_(
    config: base.FinetuneConfigBase,
    lr_multiplier: float = 0.1,
):
    if not config.parameter_specific_optimizers:
        config.parameter_specific_optimizers = []

    if (
        energy_ref_head := next(
            (
                t
                for t in config.graph_targets
                if isinstance(t, output_head.ReferencedScalarTargetConfig)
            ),
            None,
        )
    ) is None:
        raise ValueError("Referenced energy head not found")

    config.parameter_specific_optimizers.extend(
        parameter_specific_optimizer_config(
            config,
            [
                {
                    "name": f"{energy_ref_head.name}.ref",
                    "lr_multiplier": lr_multiplier,
                    "parameter_patterns": [
                        f"graph_outputs._module_dict.ft_mlp_{energy_ref_head.name}.references.*"
                    ],
                }
            ],
        )
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


def pos_aug_(config: base.FinetuneConfigBase):
    config.pos_noise_augmentation = base.PositionNoiseAugmentationConfig(
        system_corrupt_prob=0.75,
        atom_corrupt_prob=0.5,
        noise_std=0.01,
    )
    config.name_parts.append("posaug")


def create_config(config_fn: Callable[[M.MatbenchDiscoveryConfig], None]):
    config = M.MatbenchDiscoveryConfig.draft()
    config.project = "jmp_mptrj"
    config.name = "mptrj"
    config_fn(config)
    ln_(config)
    config.backbone.qint_tags = [0, 1, 2]

    def dataset_fn(split: Literal["train", "val", "test"]):
        return base.FinetuneMPTrjHuggingfaceDatasetConfig(
            split=split, energy_column="corrected_total_energy"
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
    config.num_workers = 7

    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False

    config.trainer.precision = "16-mixed-auto"
    config.trainer.set_float32_matmul_precision = "medium"

    config.with_project_root_(project_root)
    return config


configs: list[tuple[M.MatbenchDiscoveryConfig, type[M.MatbenchDiscoveryModel]]] = []

# region direct, energy+force+stress
config = create_config(jmp_s_)
ln_(config)
direct_(config)
# Energy head
config.graph_targets.append(
    output_head.ReferencedScalarTargetConfig(
        name="y",
        loss_coefficient=1.0,
        loss=loss.MACEHuberEnergyLossConfig(delta=0.01),
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
config.batch_size = 32
config.name_parts.append(f"bsz{config.batch_size}")
config.lr_scheduler.warmup_epochs = 1
config.lr_scheduler.max_epochs = 128

parameter_specific_optimizers_(config)
parameter_specific_optimizers_energy_references_(config, lr_multiplier=0.1)

config.runner.submit.auto_requeue_signals = ["SIGUSR1"]
if (wandb_config := config.trainer.logging.wandb) is not None:
    wandb_config.offline = True

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
# runner = ll.Runner(run)
# runner.fast_dev_run(configs, n_batches=128)

# # %%
# for i, env in enumerate(
#     [
#         {"CUDA_VISIBLE_DEVICES": "0,1"},
#         {"CUDA_VISIBLE_DEVICES": "2,3"},
#     ]
# ):
#     runner = ll.Runner(run)
#     _ = runner.session(
#         [configs[i]],
#         snapshot=True,
#         env={
#             **env,
#             "LL_DISABLE_TYPECHECKING": "1",
#         },
#     )

# # %%
# runner = ll.Runner(run)
# _ = runner.session(
#     configs,
#     snapshot=True,
#     env={
#         "CUDA_VISIBLE_DEVICES": "0,1,2,3",
#         "LL_DISABLE_TYPECHECKING": "1",
#     },
# )

# %%
import datetime

runner = ll.Runner(run)
_ = runner.submit_lsf(
    configs,
    snapshot=True,
    nodes=2,
    tasks_per_node=6,
    project="MAT273",
    queue="debug",
    walltime=datetime.timedelta(hours=2),
    lsf_options={
        "summit": True,
    },
    env={
        "LL_DISABLE_TYPECHECKING": "1",
    },
)
