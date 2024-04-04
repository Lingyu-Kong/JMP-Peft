from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import MatbenchDiscoveryConfig
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig


def jmp_l_matbench_discovery_config_(
    config: MatbenchDiscoveryConfig,
    base_path: Path,
    use_megnet_133k: bool = True,
    use_atoms_metadata: bool = True,
    use_linref: bool = False,
    gradient_forces: bool = True,
    force_coefficient: float = 100.0,
):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Set up dataset
    config.train_dataset = DC.matbench_discovery_config(
        base_path,
        "train",
        use_megnet_133k=use_megnet_133k,
        use_atoms_metadata=use_atoms_metadata,
        use_linref=use_linref,
    )
    config.val_dataset = DC.matbench_discovery_config(
        base_path,
        "val",
        use_megnet_133k=use_megnet_133k,
        use_atoms_metadata=use_atoms_metadata,
        use_linref=use_linref,
    )
    config.test_dataset = DC.matbench_discovery_config(
        base_path,
        "test",
        use_megnet_133k=use_megnet_133k,
        use_atoms_metadata=use_atoms_metadata,
        use_linref=use_linref,
    )

    # MatbenchDiscovery specific settings
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

    # Gradient forces
    config.forces_config_(gradient=gradient_forces, coefficient=force_coefficient)
    if gradient_forces:
        config.trainer.inference_mode = False

    # Set up normalization
    if use_megnet_133k:
        config.normalization["force"] = NC(mean=0.0, std=0.5662145031694755)
        if use_linref:
            config.normalization["y"] = NC(
                mean=0.5590934925198368, std=31.5895592795005
            )
        else:
            config.normalization["y"] = NC(
                mean=-184.37418267781965, std=188.89161113304596
            )
    else:
        raise NotImplementedError("Only megnet json is supported for now")
