# %%
import ll

ll.pretty()
# %%
import datasets
import numpy as np

datasets.disable_caching()

dataset = datasets.load_dataset("nimashoghi/mptrj")
dataset.set_format("numpy")
dataset


# %%
def transform(numbers: np.ndarray):
    return {"composition": np.bincount(numbers, minlength=120).tolist()}


dataset = dataset.map(
    transform,
    input_columns=["numbers"],
    batched=False,
)
dataset

# %%
dataset.push_to_hub("nimashoghi/mptrj")
