# %%
import itertools
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

jmp_s_ckpt_path = Path("/mnt/shared/checkpoints/jmp-s.pt")
jmp_l_ckpt_path = Path("/mnt/shared/checkpoints/jmp-l.pt")

# Set this to None if you want the run logs to be saved in the current directory
project_root: Path | None = Path("/mnt/datasets/experiment-data/jmp-peft/")
project_root.mkdir(exist_ok=True, parents=True)


def jmp_s_(config: base.FinetuneConfigBase):
    ckpt_path = jmp_s_ckpt_path
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    jmp_s_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path, ema=True
    )

    config.meta["jmp_kind"] = "s"
    config.name_parts.append("jmps")


def jmp_l_(config: base.FinetuneConfigBase):
    ckpt_path = jmp_l_ckpt_path
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    jmp_l_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path, ema=True
    )

    config.meta["jmp_kind"] = "l"
    config.name_parts.append("jmpl")


def parameter_specific_optimizers_(config: base.FinetuneConfigBase):
    if config.parameter_specific_optimizers is None:
        config.parameter_specific_optimizers = []

    match config.meta["jmp_kind"]:
        case "l":
            config.parameter_specific_optimizers.extend(
                make_parameter_specific_optimizer_config(
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
            )
        case "s":
            config.parameter_specific_optimizers.extend(
                make_parameter_specific_optimizer_config(
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
            )
        case _:
            raise ValueError(f"Invalid jmp_kind: {config.meta['jmp_kind']}")


def parameter_specific_optimizers_energy_references_(
    config: base.FinetuneConfigBase,
    lr_multiplier: float = 0.1,
):
    if not config.parameter_specific_optimizers:
        config.parameter_specific_optimizers = []

    if energy_ref_heads := [
        t
        for t in config.graph_targets
        if isinstance(t, output_head.ReferencedScalarTargetConfig)
    ]:
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
                    for energy_ref_head in energy_ref_heads
                ],
            )
        )

    elif allegro_heads := [
        t
        for t in config.graph_targets
        if isinstance(t, output_head.AllegroScalarTargetConfig)
    ]:
        config.parameter_specific_optimizers.extend(
            parameter_specific_optimizer_config(
                config,
                [
                    {
                        "name": f"{h.name}.scales",
                        "lr_multiplier": lr_multiplier,
                        "parameter_patterns": [
                            f"graph_outputs._module_dict.ft_mlp_{h.name}.per_atom_scales.*",
                            f"graph_outputs._module_dict.ft_mlp_{h.name}.per_atom_shifts.*",
                            *(
                                [
                                    f"graph_outputs._module_dict.ft_mlp_{h.name}.pairwise_scales.*"
                                ]
                                if h.edge_level_energies
                                else []
                            ),
                        ],
                    }
                    for h in allegro_heads
                ],
            )
        )
    else:
        raise ValueError("No energy reference or allegro heads found")


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


def ln_(
    config: base.FinetuneConfigBase,
    *,
    lr_multiplier: float | None,
):
    config.backbone.ln_per_layer = True
    config.backbone.scale_factor_to_ln = True

    if lr_multiplier is not None:
        if config.parameter_specific_optimizers is None:
            config.parameter_specific_optimizers = []

        config.parameter_specific_optimizers = [
            *parameter_specific_optimizer_config(
                config,
                [
                    {
                        "name": "ln",
                        "lr_multiplier": lr_multiplier,
                        "parameter_patterns": [
                            "backbone.h_lns.*",
                            "backbone.m_lns.*",
                            "backbone.*.scale*.ln.*",
                        ],
                    }
                ],
            ),
            *config.parameter_specific_optimizers,
        ]

    config.name_parts.append("ln")


def pos_aug_(config: base.FinetuneConfigBase, *, std: float):
    config.pos_noise_augmentation = base.PositionNoiseAugmentationConfig(
        system_corrupt_prob=0.75,
        atom_corrupt_prob=0.5,
        noise_std=std,
    )
    config.name_parts.append(f"posaug_std{std}")


def data_config_(
    config: M.MatbenchDiscoveryConfig,
    *,
    batch_size: int,
    reference: bool,
):
    config.batch_size = batch_size
    config.name_parts.append(f"bsz{batch_size}")

    def dataset_fn(split: Literal["train", "val", "test"]):
        return base.FinetuneMPTrjHuggingfaceDatasetConfig(
            split=split,
            energy_column_mapping={
                "y": "corrected_total_energy_referenced",
                "y_relaxed": "corrected_total_energy_relaxed_referenced",
            }
            if reference
            else {
                "y": "corrected_total_energy",
                "y_relaxed": "corrected_total_energy_relaxed",
            },
        )

    config.train_dataset = dataset_fn("train")
    config.val_dataset = dataset_fn("val")
    config.test_dataset = dataset_fn("test")

    if reference:
        config.name_parts.append("linrefenergy")
    else:
        config.name_parts.append("totalenergy")

    # Set data config
    config.num_workers = 7

    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False


def output_heads_config_(
    config: M.MatbenchDiscoveryConfig,
    *,
    relaxed_energy: bool,
    mace_energy_loss: bool,
    mace_force_loss: bool,
    energy_coefficient: float,
    force_coefficient: float,
    stress_coefficient: float,
):
    energy_loss = loss.HuberLossConfig(delta=0.01)
    if mace_energy_loss:
        energy_loss = loss.MACEHuberEnergyLossConfig(delta=0.01)
        config.name_parts.append("maceenergy")

    force_loss = loss.HuberLossConfig(delta=0.01)
    if mace_force_loss:
        force_loss = loss.MACEHuberLossConfig(delta=0.01)
        config.name_parts.append("maceforce")

    # Energy head
    config.graph_targets.append(
        output_head.AllegroScalarTargetConfig(
            name="y",
            loss_coefficient=1.0,
            loss=energy_loss.model_copy(),
            reduction="sum",
            max_atomic_number=config.backbone.num_elements,
            edge_level_energies=True,
        )
    )
    if relaxed_energy:
        # Relaxed Energy head
        config.graph_targets.append(
            output_head.AllegroScalarTargetConfig(
                name="y_relaxed",
                loss_coefficient=1.0,
                loss=energy_loss.model_copy(),
                reduction="sum",
                max_atomic_number=config.backbone.num_elements,
                edge_level_energies=True,
            )
        )

        config.name_parts.append("rele")
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
            loss=force_loss,
            reduction="sum",
        )
    )

    config.name_parts.append(f"ec{energy_coefficient}")
    config.name_parts.append(f"fc{force_coefficient}")
    config.name_parts.append(f"sc{stress_coefficient}")


def optimization_config_(
    config: M.MatbenchDiscoveryConfig,
    *,
    lr: float,
):
    config.optimizer = AdamWConfig(
        lr=lr,
        amsgrad=False,
        betas=(0.9, 0.95),
    )
    config.lr_scheduler = base.WarmupCosRLPConfig(
        warmup_epochs=1,
        warmup_start_lr_factor=1.0e-1,
        should_restart=False,
        max_epochs=128,
        min_lr_factor=0.5,
        rlp=base.RLPConfig(patience=5, factor=0.8),
    )
    config.trainer.optimizer.gradient_clipping = ll.GradientClippingConfig(
        value=2.0,
        algorithm="value",
    )

    config.name_parts.append(f"lr{lr}")


def create_config(config_fn: Callable[[M.MatbenchDiscoveryConfig], None]):
    config = M.MatbenchDiscoveryConfig.draft()

    config.trainer.precision = "16-mixed-auto"
    config.trainer.set_float32_matmul_precision = "medium"

    config.project = "jmp_mptrj"
    config.name = "mptrj"
    config_fn(config)
    config.backbone.qint_tags = [0, 1, 2]

    config.primary_metric = ll.PrimaryMetricConfig(
        name="matbench_discovery/force_mae", mode="min"
    )

    if project_root:
        config.with_project_root_(project_root)
    return config


configs: list[tuple[M.MatbenchDiscoveryConfig, type[M.MatbenchDiscoveryModel]]] = []

batch_size = 32
lr = 8.0e-5
linref = True

for linref, lr, pos_aug in itertools.product(
    (True, False),
    (5.0e-6, 1.0e-5, 8.0e-5),
    (False,),
):
    config = create_config(jmp_s_)
    config.parameter_specific_optimizers = []
    data_config_(config, reference=linref, batch_size=batch_size)
    optimization_config_(config, lr=lr)
    ln_(config, lr_multiplier=1.5)
    direct_(config=config)
    output_heads_config_(
        config,
        relaxed_energy=True,
        mace_energy_loss=True,
        mace_force_loss=True,
        energy_coefficient=1.0,
        force_coefficient=10.0,
        stress_coefficient=100.0,
    )
    parameter_specific_optimizers_(config)
    parameter_specific_optimizers_energy_references_(config, lr_multiplier=0.1)
    if pos_aug:
        pos_aug_(config, std=pos_aug)
    config.per_graph_radius_graph = True

    config = config.finalize()
    configs.append((config, M.MatbenchDiscoveryModel))

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
runner.fast_dev_run(configs, n_batches=128)

# %%
runner = ll.Runner(run)
_ = runner.submit_slurm(
    configs,
    snapshot=True,
    env={
        "LL_DISABLE_TYPECHECKING": "1",
    },
    partition="learnaccel",
    nodes=4,
    tasks_per_node=8,  # Change this to limit # of GPUs
    gpus_per_task=1,
    cpus_per_task=configs[0][0].num_workers + 1,
)
