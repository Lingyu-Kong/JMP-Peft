from pathlib import Path

from nshtrainer.ll.model import PrimaryMetricConfig

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.finetune import MatbenchDiscoveryConfig
from ...tasks.finetune import dataset_config as DC


def jmp_matbench_discovery_config_(
    config: MatbenchDiscoveryConfig,
    base_path: Path,
    use_megnet_133k: bool = True,
    use_atoms_metadata: bool = True,
    use_linref: bool = False,
):
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
    config.primary_metric = PrimaryMetricConfig(
        name="matbench_discovery/force_mae", mode="min"
    )

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
