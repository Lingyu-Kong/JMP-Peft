# %%
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import ll
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


def jmp_s_(config: base.FinetuneConfigBase):
    ckpt_path = jmp_s_ckpt_path
    # assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    jmp_s_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path, ema=True
    )

    config.meta["jmp_kind"] = "s"
    config.tags.append("jmps")
    config.name_parts.append("jmps")


def jmp_l_(config: base.FinetuneConfigBase):
    ckpt_path = jmp_l_ckpt_path
    # assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    jmp_l_ft_config_(config)
    config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
        path=ckpt_path, ema=True
    )

    config.meta["jmp_kind"] = "l"
    config.tags.append("jmpl")
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
    config.tags.append("direct")


def grad_(config: base.FinetuneConfigBase):
    config.backbone.regress_forces = False
    config.backbone.direct_forces = False
    config.backbone.regress_energy = True

    config.trainer.inference_mode = False

    config.tags.append("grad")


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

    config.tags.append("ln")


def pos_aug_(config: base.FinetuneConfigBase, *, std: float):
    config.pos_noise_augmentation = base.PositionNoiseAugmentationConfig(
        system_corrupt_prob=0.75,
        atom_corrupt_prob=0.5,
        noise_std=std,
    )
    config.tags.append(f"posaug_std{std}")


def data_config_(
    config: M.MatbenchDiscoveryConfig,
    *,
    batch_size: int,
    reference: bool,
):
    config.batch_size = batch_size
    config.tags.append(f"bsz{batch_size}")

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
        config.tags.append("linrefenergy")
    else:
        config.tags.append("totalenergy")

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
    energy_loss_ratio: float = 0.75,
):
    energy_loss = loss.HuberLossConfig(delta=0.01)
    if mace_energy_loss:
        energy_loss = loss.MACEHuberEnergyLossConfig(delta=0.01)
        config.tags.append("maceenergy")

    force_loss = loss.HuberLossConfig(delta=0.01)
    if mace_force_loss:
        force_loss = loss.MACEHuberLossConfig(delta=0.01)
        config.tags.append("maceforce")

    # Energy head
    energy_loss_coefficient = energy_coefficient
    relaxed_energy_loss_coefficient = energy_coefficient
    if relaxed_energy:
        energy_loss_coefficient *= energy_loss_ratio
        relaxed_energy_loss_coefficient *= 1 - energy_loss_ratio
    config.graph_targets.append(
        output_head.AllegroScalarTargetConfig(
            name="y",
            loss_coefficient=energy_coefficient,
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
                loss_coefficient=energy_coefficient / 2,
                loss=energy_loss.model_copy(),
                reduction="sum",
                max_atomic_number=config.backbone.num_elements,
                edge_level_energies=True,
            )
        )

        config.tags.append("rele")
    # Stress head
    config.graph_targets.append(
        output_head.DirectStressTargetConfig(
            name="stress",
            loss_coefficient=stress_coefficient,
            loss=loss.HuberLossConfig(delta=0.01),
            reduction="mean",
        )
    )
    # Force head
    config.node_targets.append(
        output_head.NodeVectorTargetConfig(
            name="force",
            loss_coefficient=force_coefficient,
            loss=force_loss,
            reduction="sum",
        )
    )

    config.tags.append(f"ec{energy_coefficient}")
    config.tags.append(f"fc{force_coefficient}")
    config.tags.append(f"sc{stress_coefficient}")


def optimization_config_(
    config: M.MatbenchDiscoveryConfig,
    *,
    lr: float,
    wd: float,
):
    config.optimizer = AdamWConfig(
        lr=lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=wd,
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

    config.tags.append(f"lr{lr}")
    config.tags.append(f"wd{wd}")


def base_config_(
    config: M.MatbenchDiscoveryConfig,
    config_fn: Callable[[M.MatbenchDiscoveryConfig], None],
):
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


def make_config(
    model_fn: Callable[[base.FinetuneConfigBase], None],
    *,
    linref: bool,
    batch_size: int,
    lr: float,
    coefficients: tuple[float, float, float],
    pos_aug: float | None,
    relaxed_energy: bool,
    wd: float = 0.01,
):
    config = M.MatbenchDiscoveryConfig.draft()

    base_config_(config, model_fn)
    config.parameter_specific_optimizers = []
    data_config_(config, reference=linref, batch_size=batch_size)
    optimization_config_(config, lr=lr, wd=wd)
    ln_(config, lr_multiplier=1.5)
    direct_(config=config)
    energy, force, stress = coefficients
    output_heads_config_(
        config,
        relaxed_energy=relaxed_energy,
        mace_energy_loss=True,
        mace_force_loss=True,
        energy_coefficient=energy,
        force_coefficient=force,
        stress_coefficient=stress,
    )
    parameter_specific_optimizers_(config)
    parameter_specific_optimizers_energy_references_(config, lr_multiplier=0.1)
    if pos_aug:
        pos_aug_(config, std=pos_aug)
    config.per_graph_radius_graph = True

    config = config.finalize()
    return config


configs: list[tuple[M.MatbenchDiscoveryConfig, type[M.MatbenchDiscoveryModel]]] = []

# Base settings
batch_size = 32
lr = 8.0e-5
wd = 0.01
linref = True
coefficients = (20.0, 20.0, 10.0)
pos_aug = None
relaxed_energy = True

for relaxed_energy_ in (False, True):
    config = make_config(
        jmp_s_,
        linref=linref,
        batch_size=batch_size,
        lr=lr,
        coefficients=coefficients,
        pos_aug=pos_aug,
        relaxed_energy=relaxed_energy_,
    )
    if relaxed_energy_:
        config.name_parts.append("s2ef_s2re")
    else:
        config.name_parts.append("s2efonly")
    configs.append((config, M.MatbenchDiscoveryModel))


for coefficients_ in (
    (20.0, 20.0, 10.0),  # Luis' original coefficients
    (2.0, 10.0, 100.0),  # Current best coefficients
    (100.0, 100.0, 1.0),  # SevenNet coefficients
    (1.0, 100.0, 1.0),  # Force preferred
    (100.0, 1.0, 1.0),  # Energy preferred
):
    config = make_config(
        jmp_s_,
        linref=linref,
        batch_size=batch_size,
        lr=lr,
        coefficients=coefficients_,
        pos_aug=pos_aug,
        relaxed_energy=relaxed_energy,
    )
    config.name_parts.append(
        f"ec{coefficients_[0]}_fc{coefficients_[1]}_sc{coefficients_[2]}"
    )
    configs.append((config, M.MatbenchDiscoveryModel))

for wd_ in (0.01, 0.1):
    config = make_config(
        jmp_s_,
        linref=linref,
        batch_size=batch_size,
        lr=lr,
        coefficients=coefficients,
        pos_aug=pos_aug,
        relaxed_energy=relaxed_energy,
        wd=wd_,
    )
    config.name_parts.append(f"wd{wd_}")
    configs.append((config, M.MatbenchDiscoveryModel))
print(f"{len(configs)} configs")


# Remove duplicate configs
def _remove_duplicate_configs(
    configs: list[tuple[M.MatbenchDiscoveryConfig, type[M.MatbenchDiscoveryModel]]],
):
    seen_configs: set[str] = set()
    unique_configs: list[
        tuple[M.MatbenchDiscoveryConfig, type[M.MatbenchDiscoveryModel]]
    ] = []

    for config, model_cls in configs:
        config_json = config.model_dump_json(
            exclude={"id", "name", "name_parts", "tags"}
        )
        if config_json in seen_configs:
            print(f"Duplicate config found: {config.run_name}")
            continue

        seen_configs.add(config_json)
        unique_configs.append((config, model_cls))

    return unique_configs


og_size = len(configs)
configs = _remove_duplicate_configs(configs)
print(f"{len(configs)} unique configs ({og_size - len(configs)} duplicates removed)")

print([c.run_name for c, _ in configs])

# %%
for c, _ in configs:
    print(f"- {c.run_name}")


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
