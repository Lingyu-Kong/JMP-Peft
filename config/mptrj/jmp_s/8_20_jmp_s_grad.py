# %%
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import nshtrainer as nt
import nshutils as nu
from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.jmp_s import jmp_s_ft_config_
from jmppeft.datasets import mptrj_hf
from jmppeft.modules import loss
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune import base, output_head
from jmppeft.tasks.finetune import matbench_discovery as M
from jmppeft.utils.param_specific_util import (
    make_parameter_specific_optimizer_config,
    parameter_specific_optimizer_config,
)

jmp_s_ckpt_path = Path(
    "/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/jmp-s.pt"
)
jmp_l_ckpt_path = Path(
    "/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/jmp-l.pt"
)

# Set this to None if you want the run logs to be saved in the current directory
project_root: Path | None = Path(
    "/net/csefiles/coc-fung-cluster/nima/shared/experiment-data/"
)
project_root.mkdir(exist_ok=True, parents=True)


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
    config.backbone.regress_forces = True
    config.backbone.direct_forces = True
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


def pos_noise_aug_(config: base.FinetuneConfigBase, *, std: float):
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
    reference: mptrj_hf.ReferenceConfig | None,
):
    config.batch_size = batch_size
    config.tags.append(f"bsz{batch_size}")

    def dataset_fn(split: Literal["train", "val", "test"]):
        dataset_config = mptrj_hf.FinetuneMPTrjHuggingfaceDatasetConfig(
            split=split,
            energy_column_mapping={
                "y": "corrected_total_energy",
                "y_relaxed": "corrected_total_energy_relaxed",
            },
        )
        if reference:
            dataset_config.references["y"] = reference
        return dataset_config

    config.train_dataset = dataset_fn("train")
    config.val_dataset = dataset_fn("val")
    config.test_dataset = dataset_fn("test")

    # Set data config
    config.num_workers = 7

    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False


def output_heads_config_direct_(
    config: M.MatbenchDiscoveryConfig,
    *,
    mace_energy_loss: bool,
    mace_force_loss: bool,
    energy_coefficient: float,
    force_coefficient: float,
    stress_coefficient: float,
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
    config.graph_targets.append(
        output_head.AllegroScalarTargetConfig(
            name="y",
            loss_coefficient=energy_coefficient,
            loss=energy_loss,
            reduction="sum",
            max_atomic_number=config.backbone.num_elements,
            edge_level_energies=True,
        )
    )
    # Stress head
    config.graph_targets.append(
        output_head.DirectStressTargetConfig(
            name="stress",
            loss_coefficient=stress_coefficient,
            loss=loss.HuberLossConfig(
                y_mult_coeff=1602.1766208,  # eV/A^3 to kbar
                delta=0.01,
            ),
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


def output_heads_config_grad_(
    config: M.MatbenchDiscoveryConfig,
    *,
    mace_energy_loss: bool,
    mace_force_loss: bool,
    energy_coefficient: float,
    force_coefficient: float,
    stress_coefficient: float,
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
    config.graph_targets.append(
        output_head.AllegroScalarTargetConfig(
            name="y",
            loss_coefficient=energy_coefficient,
            loss=energy_loss,
            reduction="sum",
            max_atomic_number=config.backbone.num_elements,
            edge_level_energies=True,
        )
    )
    # Stress head
    config.graph_targets.append(
        output_head.GradientStressTargetConfig(
            name="stress",
            energy_name="y",
            forces=True,
            loss_coefficient=stress_coefficient,
            loss=loss.HuberLossConfig(delta=0.01),
            reduction="mean",
        )
    )
    # Force head
    config.node_targets.append(
        output_head.GradientForcesTargetConfig(
            name="force",
            energy_name="y",
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
    config.trainer.optimizer.gradient_clipping = nt.model.GradientClippingConfig(
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

    config.primary_metric = nt.MetricConfig(
        name="matbench_discovery/force_mae", mode="min"
    )

    if project_root:
        config.with_project_root_(project_root)


def make_config(
    model_fn: Callable[[base.FinetuneConfigBase], None],
    *,
    reference: mptrj_hf.ReferenceConfig | None,
    batch_size: int,
    lr: float,
    coefficients: tuple[float, float, float],
    pos_aug: float | None,
    wd: float = 0.01,
    mace_energy_loss: bool = False,
    mace_force_loss: bool = False,
    grad: bool = False,
):
    config = M.MatbenchDiscoveryConfig.draft()

    base_config_(config, model_fn)
    config.parameter_specific_optimizers = []
    data_config_(config, reference=reference, batch_size=batch_size)
    optimization_config_(config, lr=lr, wd=wd)
    ln_(config, lr_multiplier=1.5)
    if grad:
        grad_(config=config)
    else:
        direct_(config=config)

    energy, force, stress = coefficients
    output_heads_config_ = (
        output_heads_config_grad_ if grad else output_heads_config_direct_
    )
    output_heads_config_(
        config,
        mace_energy_loss=mace_energy_loss,
        mace_force_loss=mace_force_loss,
        energy_coefficient=energy,
        force_coefficient=force,
        stress_coefficient=stress,
    )
    parameter_specific_optimizers_(config)
    parameter_specific_optimizers_energy_references_(config, lr_multiplier=0.1)
    if pos_aug:
        pos_noise_aug_(config, std=pos_aug)
    config.per_graph_radius_graph = True

    config = config.finalize()
    return config


configs: list[tuple[M.MatbenchDiscoveryConfig, type[M.MatbenchDiscoveryModel]]] = []

# Base settings
batch_size = 50
lr = 8.0e-5
wd = 0.01
coefficients = (1.0, 1.0, 0.01)
pos_aug = None
mace_energy_loss = False
mace_force_loss = False
reference = mptrj_hf.RidgeReferenceConfig(alpha=0.1)
config = make_config(
    jmp_s_,
    reference=reference,
    batch_size=batch_size,
    lr=lr,
    coefficients=coefficients,
    pos_aug=pos_aug,
    wd=wd,
    mace_energy_loss=mace_energy_loss,
    mace_force_loss=mace_force_loss,
    grad=False,
)
config.cutoffs = M.Cutoffs.from_constant(20.0)
config.backbone.absolute_rbf_cutoff = 5.0
configs.append((config, M.MatbenchDiscoveryModel))


nu.display(config)


# %%
def run(
    config: M.MatbenchDiscoveryConfig, model_cls: type[M.MatbenchDiscoveryModel]
) -> None:
    model = model_cls.construct_and_load_checkpoint(config)
    trainer = nt.Trainer(config)
    trainer.fit(model)


# %%
runner = nt.Runner(run)
runner.fast_dev_run(configs, n_batches=128)

# %%
runner = nt.Runner(run)
_ = runner.session(
    configs,
    snapshot=True,
    env={
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NSHUTILS_DISABLE_TYPECHECKING": "1",
    },
)
