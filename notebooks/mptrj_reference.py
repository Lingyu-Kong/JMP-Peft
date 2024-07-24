# %%
import nshtrainer.ll as ll

ll.pretty()

# %%
# First, compute the references

import datasets
import numpy as np

datasets.disable_caching()

dataset = datasets.load_dataset("nimashoghi/mptrj", split="train")
dataset.set_format("numpy")
dataset


# %%
def _compositions(numbers: np.ndarray):
    return {"composition": np.bincount(numbers, minlength=120).tolist()}


dataset_with_compositions = dataset.map(
    _compositions,
    input_columns=["numbers"],
)
dataset_with_compositions

# %%
X = dataset_with_compositions["composition"]
y = dataset_with_compositions["energy"]
print(X.shape, y.shape)

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression(fit_intercept=False)
lr.fit(X, y)

print(lr.coef_)

# %%
np.save("/mnt/datasets/matbench-discovery-traj/mptrj_linref.npy", lr.coef_)

# %%
import datasets
import numpy as np

datasets.disable_caching()

ddict = datasets.load_dataset("nimashoghi/mptrj")
ddict.set_format("numpy")
ddict

# %%
energy_columns = [
    "energy",
    "corrected_total_energy",
    "corrected_total_energy_relaxed",
]
linref = np.load("/mnt/datasets/matbench-discovery-traj/mptrj_linref.npy")


def reference(
    numbers: np.ndarray,
    energy: float,
    reference: np.ndarray,
):
    return energy - float(reference[numbers].sum())


def transform(
    numbers: np.ndarray,
    *energies: float,
    linref: np.ndarray,
):
    return {
        f"{col}_referenced": reference(numbers, energy, linref)
        for col, energy in zip(energy_columns, energies)
    }


ddict = ddict.map(
    transform,
    input_columns=["numbers", *energy_columns],
    fn_kwargs={"linref": linref},
    batched=False,
)
ddict

# %%
ddict.push_to_hub("nimashoghi/mptrj")
