from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import MatbenchDiscoveryConfig
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig


def jmp_l_matbench_discovery_config_(
    config: MatbenchDiscoveryConfig,
    base_path: Path,
    use_megnet_json: bool = True,
    total_energy: bool = True,
    use_atoms_metadata: bool = True,
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
        use_megnet_json=use_megnet_json,
        total_energy=total_energy,
        use_atoms_metadata=use_atoms_metadata,
    )
    config.val_dataset = DC.matbench_discovery_config(
        base_path,
        "val",
        use_megnet_json=use_megnet_json,
        total_energy=total_energy,
        use_atoms_metadata=use_atoms_metadata,
    )
    config.test_dataset = DC.matbench_discovery_config(
        base_path,
        "test",
        use_megnet_json=use_megnet_json,
        total_energy=total_energy,
        use_atoms_metadata=use_atoms_metadata,
    )

    # MatbenchDiscovery specific settings
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

    # Gradient forces
    config.forces_config_(gradient=True)
    config.trainer.inference_mode = False

    # Set up normalization
    if total_energy:
        """
        -188.96425528278039 191.08882335828582
        -6.361073855810166e-12 0.8080419658412661
        0.6529318185606179

        """
        config.normalization = {
            "y": NC(mean=-188.96425528278039, std=191.08882335828582),
            "force": NC(mean=0.0, std=0.6529318185606179),
        }
    else:
        raise NotImplementedError("Only total energy is supported for now")
