from pathlib import Path

from jmppeft.tasks.pretrain import module as M


def tasks_config_frontier_(config: M.PretrainConfig):
    config.tasks = [
        M.TaskConfig(
            name="oc20",
            train_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc20/s2ef/s2ef/2M/train/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc20/s2ef/s2ef/2M/train/metadata.npz"
                ),
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc20/s2ef/s2ef/all/val_id/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc20/s2ef/s2ef/all/val_id/metadata.npz"
                ),
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
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/val_id/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/val_id/metadata.npz"
                ),
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
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/ani1x/lmdbs/val/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/ani1x/lmdbs/train/ani1x_val_metadata.npz"
                ),
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
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/transition1x/lmdbs/val/"
                ),
                metadata_path=Path(
                    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/transition1x/lmdbs/train/metadata.npz"
                ),
            ),
            energy_loss_scale=1.0,
            force_loss_scale=14.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=1.787466168382901),
                "force": M.NormalizationConfig(mean=0.0, std=0.3591422140598297),
            },
        ),
    ]
