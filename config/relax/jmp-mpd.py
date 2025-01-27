# %%
from pathlib import Path

import nshtrainer.ll as ll
import torch
from jmppeft.modules.dataset.common import DatasetSampleNConfig
from jmppeft.tasks.finetune import base
from jmppeft.tasks.finetune import energy_forces_base as ef
from jmppeft.tasks.finetune import matbench_discovery as mpd
from typing_extensions import TypeVar

TConfig = TypeVar("TConfig", bound=ll.BaseConfig, infer_variance=True)


def config_from_ckpt(config_cls: type[TConfig], ckpt_path: Path) -> TConfig:
    ckpt = torch.load(ckpt_path)
    config = config_cls.model_validate(ckpt["hyper_parameters"])
    return config.reset_()


def update_for_relaxation_(config: ef.EnergyForcesConfigBase):
    config.predict_dataset = base.FinetuneMatBenchDiscoveryIS2REDatasetConfig(
        # energy_linref_path=Path(
        #     "/mnt/datasets/matbench-discovery-traj/megnet-133k-npz/linrefs.npy"
        # ),
        sample_n=DatasetSampleNConfig(sample_n=1024, seed=42),
    )
    config.train_dataset = None
    config.val_dataset = None

    config.batch_size = 1
    config.eval_batch_size = 1
    config.relaxation = ef.RelaxationConfig(
        validation=None,
        predict=ef.RelaxerConfig(fmax=0.05),
        # relaxed_energy_linref_path=Path(
        #     "/mnt/datasets/matbench-discovery-traj/megnet-133k-npz/linrefs.npy"
        # ),
        use_chgnet_for_relaxed_energy=True,
    )

    # config.normalization["y_relaxed"] = config.normalization["y"].model_copy()


configs: list[tuple[base.FinetuneConfigBase, type[base.FinetuneModelBase]]] = []

ckpt_path = Path("/mnt/shared/checkpoints/jmp-mptrj/latest_epoch85_step807110.ckpt")
config = config_from_ckpt(mpd.MatbenchDiscoveryConfig, ckpt_path)
update_for_relaxation_(config)
config.meta["ckpt_path"] = ckpt_path

configs.append((config, mpd.MatbenchDiscoveryModel))


# %%
def run(config: base.FinetuneConfigBase, model_cls: type[base.FinetuneModelBase]):
    model = model_cls(config)
    model.load_state_dict(torch.load(config.meta["ckpt_path"])["state_dict"])

    trainer = ll.Trainer(config)
    trainer.predict(model, return_predictions=False)


# %%
runner = ll.Runner(run)
runner.fast_dev_run(configs, n_batches=32)

# %%
runner = ll.Runner(run)
runner.local_session_per_gpu(configs, gpus=(1,))
