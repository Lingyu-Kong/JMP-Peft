# %%
import ll

ll.pretty()
# %%
import datasets

datasets.disable_caching()

dataset = datasets.load_dataset("nimashoghi/mptrj")
dataset.set_format("numpy")
dataset


# %%
import json

import numpy as np

with open("data/mptrj_ids.json", "r") as f:
    ids = frozenset[str](json.load(f))


def filter(mp_id: str, *, all_ids: frozenset[str]):
    return mp_id in all_ids


dataset = dataset.filter(filter, input_columns=["mp_id"], fn_kwargs={"all_ids": ids})
dataset

# %%
dataset.push_to_hub("nimashoghi/mptrj_filtered")
