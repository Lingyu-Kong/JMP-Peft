import datasets

ddict_datasets = datasets.load_from_disk("/mnt/datasets/oc22/hf_datasets")
ddict_datasets

ddict_datasets.push_to_hub("nimashoghi/oc22", private=True)
