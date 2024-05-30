# %%
from pathlib import Path

import ll
import rich
import torch
from jmppeft.tasks.finetune import base
from jmppeft.tasks.finetune import dataset_config as DC
from jmppeft.tasks.finetune import energy_forces_base as ef
from jmppeft.tasks.finetune import matbench_discovery as mpd
from typing_extensions import TypeVar

TConfig = TypeVar("TConfig", bound=ll.BaseConfig, infer_variance=True)
base_path = Path("/mnt/datasets/matbench-discovery-traj/megnet-133k-npz/")


def config_from_ckpt(config_cls: type[TConfig], ckpt_path: Path) -> TConfig:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = config_cls.model_validate(ckpt["hyper_parameters"])
    return config.reset_()


def update_config_(config: ef.EnergyForcesConfigBase):
    config.train_dataset = DC.matbench_discovery_config(
        base_path,
        "train",
        use_megnet_133k=True,
        use_atoms_metadata=True,
        use_linref=True,
    )
    config.val_dataset = DC.matbench_discovery_config(
        base_path,
        "val",
        use_megnet_133k=True,
        use_atoms_metadata=True,
        use_linref=True,
    )
    config.test_dataset = DC.matbench_discovery_config(
        base_path,
        "test",
        use_megnet_133k=True,
        use_atoms_metadata=True,
        use_linref=True,
    )

    config.batch_size = 1
    config.eval_batch_size = 1

    config.trainer.precision = "32-true"
    config.trainer.accelerator = "cpu"

    config.trainer.actsave = ll.ActSaveConfig()


configs: list[tuple[base.FinetuneConfigBase, type[base.FinetuneModelBase]]] = []

ckpt_path = Path("/mnt/shared/checkpoints/jmp-mptrj/latest_epoch85_step807110.ckpt")
config = config_from_ckpt(mpd.MatbenchDiscoveryConfig, ckpt_path)
update_config_(config)
config.meta["ckpt_path"] = ckpt_path

configs.append((config, mpd.MatbenchDiscoveryModel))


# %%
def run(config: base.FinetuneConfigBase, model_cls: type[base.FinetuneModelBase]):
    model = model_cls(config)
    ckpt = torch.load(config.meta["ckpt_path"], map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    trainer = ll.Trainer(config)
    trainer.validate(model)

    import rich

    rich.print(model.config.id)


# %%
runner = ll.Runner(run)
runner.fast_dev_run(configs, n_batches=1024, reset_memory_caches=False)

# %%
acts = ll.ActLoad.from_latest_version(
    "/workspaces/repositories/jmp-peft/config/relax/lltrainer/nz809rlq/activation"
)
rich.print(acts)

# %%
# prefix = "validation.m_"
# prefix = "validation.h_"
prefix = "validation.x_E_"
acts_list = [act for act in acts if act.name.startswith(prefix)]
rich.print(acts_list)

# %%
import lovely_numpy as ln
import numpy as np

ln.set_config(repr=ln.lovely)


def process(act: np.ndarray):
    # Take norm of the final axis
    act = np.linalg.norm(act, axis=-1)
    return act


stacked = {
    act.name[len("validation.") :]: process(np.concatenate(act.all_activations()))
    for act in acts_list
}
rich.print(stacked)

# %%
# Plot a histogram of the activations, colored by the name of the activation

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

fig, ax = plt.subplots()
for name, m in stacked.items():
    sns.histplot(m, kde=True, label=name, ax=ax)

ax.legend()
plt.show()
