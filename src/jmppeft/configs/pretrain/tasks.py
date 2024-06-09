from pathlib import Path

from ...modules.dataset.common import DatasetFirstNConfig, DatasetSampleRatioConfig
from ...tasks.pretrain import module as M


def tasks_config_frontier_old_(
    config: M.PretrainConfig,
    sample_seed: int = 0,
):
    oc20_ratio: float = 2_000_000 / 100_000_000
    sample_ratio = DatasetSampleRatioConfig(sample_ratio=oc20_ratio, seed=sample_seed)

    config.tasks = [
        # OC20 doesn't get a sample ratio because it's already pre-sampled (we only downloaded 2M / 100M samples)
        M.TaskConfig(
            name="oc20",
            train_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc20/s2ef/2M/train/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc20/s2ef/2M/train/metadata.npz"
                ),
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc20/s2ef/all/val_id/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc20/s2ef/all/val_id/metadata.npz"
                ),
                first_n=DatasetFirstNConfig(first_n=20_000),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=73.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=24.901469505465872),
                "force": M.NormalizationConfig(mean=0.0, std=0.5111534595489502),
            },
        ),
        M.TaskConfig(
            name="oc22",
            train_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train/metadata.npz"
                ),
                sample_ratio=sample_ratio,
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/val_id/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/val_id/metadata.npz"
                ),
                first_n=DatasetFirstNConfig(first_n=10_000),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=80.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=25.229595396538468),
                "force": M.NormalizationConfig(mean=0.0, std=0.25678861141204834),
            },
        ),
        M.TaskConfig(
            name="ani1x",
            train_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/ani1x/lmdbs/train/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/ani1x/lmdbs/train/ani1x_metadata.npz"
                ),
                sample_ratio=sample_ratio,
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/ani1x/lmdbs/val/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/ani1x/lmdbs/val/ani1x_val_metadata.npz"
                ),
                first_n=DatasetFirstNConfig(first_n=5_000),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=15.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=2.8700712783472118),
                "force": M.NormalizationConfig(mean=0.0, std=2.131422996520996),
            },
        ),
        M.TaskConfig(
            name="transition1x",
            train_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/transition1x/lmdbs/train/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/transition1x/lmdbs/train/metadata.npz"
                ),
                sample_ratio=sample_ratio,
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/transition1x/lmdbs/val/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/transition1x/lmdbs/train/metadata.npz"
                ),
                first_n=DatasetFirstNConfig(first_n=10_000),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=14.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=1.787466168382901),
                "force": M.NormalizationConfig(mean=0.0, std=0.3591422140598297),
            },
        ),
    ]


def tasks_config_generic_(
    config: M.PretrainConfig,
    *,
    sample_seed: int = 0,
    base_dir: Path,
    metadatas_dir: Path,
):
    oc20_ratio: float = 2_000_000 / 100_000_000
    sample_ratio = DatasetSampleRatioConfig(sample_ratio=oc20_ratio, seed=sample_seed)

    config.tasks = [
        # OC20 doesn't get a sample ratio because it's already pre-sampled (we only downloaded 2M / 100M samples)
        M.TaskConfig(
            name="oc20",
            train_dataset=M.PretrainDatasetConfig(
                src=base_dir / "oc20/s2ef/2M/train/",
                metadata_path=metadatas_dir / "oc20-2M-train.npz",
                lin_ref=base_dir / "oc20/lin_ref_coeffs.npz",
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=base_dir / "oc20/s2ef/all/val_id/",
                metadata_path=metadatas_dir / "oc20-val_id.npz",
                first_n=DatasetFirstNConfig(first_n=20_000),
                lin_ref=base_dir / "oc20/lin_ref_coeffs.npz",
            ),
            energy_loss_scale=1.0,
            force_loss_scale=73.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=24.901469505465872),
                "force": M.NormalizationConfig(mean=0.0, std=0.5111534595489502),
            },
        ),
        M.TaskConfig(
            name="oc22",
            train_dataset=M.PretrainDatasetConfig(
                src=base_dir
                / "oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train/",
                metadata_path=metadatas_dir / "oc22-train.npz",
                sample_ratio=sample_ratio,
                lin_ref=base_dir / "oc22/oc22_linfit_coeffs.npz",
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=base_dir
                / "oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/val_id/",
                metadata_path=metadatas_dir / "oc22-val_id.npz",
                first_n=DatasetFirstNConfig(first_n=10_000),
                lin_ref=base_dir / "oc22/oc22_linfit_coeffs.npz",
            ),
            energy_loss_scale=1.0,
            force_loss_scale=80.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=25.229595396538468),
                "force": M.NormalizationConfig(mean=0.0, std=0.25678861141204834),
            },
        ),
        M.TaskConfig(
            name="ani1x",
            train_dataset=M.PretrainDatasetConfig(
                src=base_dir / "ani1x/lmdbs/train/",
                metadata_path=metadatas_dir / "ani1x-train.npz",
                sample_ratio=sample_ratio,
                lin_ref=base_dir / "ani1x/lmdbs/train/lin_ref_coeffs.npz",
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=base_dir / "ani1x/lmdbs/val/",
                metadata_path=metadatas_dir / "ani1x-val.npz",
                first_n=DatasetFirstNConfig(first_n=5_000),
                lin_ref=base_dir / "ani1x/lmdbs/train/lin_ref_coeffs.npz",
            ),
            energy_loss_scale=1.0,
            force_loss_scale=15.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=2.8700712783472118),
                "force": M.NormalizationConfig(mean=0.0, std=2.131422996520996),
            },
        ),
        M.TaskConfig(
            name="transition1x",
            train_dataset=M.PretrainDatasetConfig(
                src=base_dir / "transition1x/lmdbs/train/",
                metadata_path=metadatas_dir / "transition1x-train.npz",
                sample_ratio=sample_ratio,
                lin_ref=base_dir / "transition1x/lmdbs/train/lin_ref_coeffs.npz",
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=base_dir / "transition1x/lmdbs/val/",
                metadata_path=metadatas_dir / "transition1x-val.npz",
                first_n=DatasetFirstNConfig(first_n=10_000),
                lin_ref=base_dir / "transition1x/lmdbs/train/lin_ref_coeffs.npz",
            ),
            energy_loss_scale=1.0,
            force_loss_scale=14.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=1.787466168382901),
                "force": M.NormalizationConfig(mean=0.0, std=0.3591422140598297),
            },
        ),
    ]


def tasks_config_perlmutter_(
    config: M.PretrainConfig,
    sample_seed: int = 0,
    base_dir: Path = Path("/global/cfs/cdirs/m3641/Nima/datasets/"),
    metadatas_dir: Path = Path("/global/cfs/cdirs/m3641/Nima/metadatas/"),
):
    tasks_config_generic_(
        config=config,
        sample_seed=sample_seed,
        base_dir=base_dir,
        metadatas_dir=metadatas_dir,
    )


def tasks_config_frontier_(
    config: M.PretrainConfig,
    sample_seed: int = 0,
    base_dir: Path = Path("/lustre/orion/mat265/world-shared/nimashoghi/datasets/"),
    metadatas_dir: Path = Path(
        "/lustre/orion/mat265/world-shared/nimashoghi/datasets/metadatas/"
    ),
):
    tasks_config_generic_(
        config=config,
        sample_seed=sample_seed,
        base_dir=base_dir,
        metadatas_dir=metadatas_dir,
    )
