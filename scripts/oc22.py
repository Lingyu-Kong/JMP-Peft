# %%
import nshutils

nshutils.pretty()
# %%
from pathlib import Path

import rich
from jmppeft.tasks.pretrain import module as M

base_path = Path(
    "/mnt/datasets/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total"
)
configs = [
    M.PretrainDatasetConfig(src=subdir)
    for subdir in base_path.iterdir()
    if subdir.is_dir()
]
rich.print(configs)

# %%
import datasets
import numpy as np
import torch

datasets.disable_caching()


def generator(config: M.PretrainDatasetConfig):
    dataset = M.PretrainLmdbDataset(config, use_referenced_energies=False)
    for i in range(len(dataset)):
        d = dataset[i].to_dict()
        d["atomic_numbers"] = d["atomic_numbers"].long()
        d["tags"] = d["tags"].long()
        d["fixed"] = d["fixed"].bool()
        d = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        d["composition"] = np.bincount(d["atomic_numbers"], minlength=120)
        yield d


ddict = {
    config.src.name: datasets.Dataset.from_generator(
        generator,
        gen_kwargs={"config": config},
        # num_proc=32,
    )
    for config in configs
}
ddict


# %%
ddict_new = {**ddict}
# Update ddict['test_id'] and ddict['test_ood'] to set the targets ("y" and "force") to float('nan')
for key in ["test_id", "test_ood"]:
    ddict_new[key] = ddict_new[key].map(
        lambda natoms: {"y": float("nan"), "force": np.full((natoms, 3), float("nan"))},
        input_columns=["natoms"],
    )

# %%
for key in ["test_id", "test_ood"]:
    ddict_new[key] = ddict_new[key].cast_column(
        "force",
        datasets.Sequence(
            datasets.Sequence(datasets.Value(dtype="float32"), length=-1),
            length=-1,
        ),
    )
# %%
ddict_datasets = datasets.DatasetDict(ddict_new)
ddict_datasets.save_to_disk("/mnt/datasets/oc22/hf_datasets")

# %%
import datasets

ddict_datasets = datasets.load_from_disk("/mnt/datasets/oc22/hf_datasets")
ddict_datasets

# %%
ddict_datasets.push_to_hub("nimashoghi/oc22", private=True)

# %%
ddict_datasets["train"].features["force"]
ddict_datasets["test_id"].features["force"]
