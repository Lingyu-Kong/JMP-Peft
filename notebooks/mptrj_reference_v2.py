# %%
import nshtrainer.ll as ll

ll.pretty()
# %%
import datasets
import numpy as np

datasets.disable_caching()

dataset = datasets.load_dataset("nimashoghi/mptrj")
dataset.set_format("numpy")
dataset

# %%
energy_columns = [
    "energy",
    "corrected_total_energy",
    "corrected_total_energy_relaxed",
]
linref = np.load("")


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


dataset = dataset.map(
    transform,
    input_columns=["numbers", *energy_columns],
    fn_kwargs={"linref": linref},
    batched=False,
)
dataset

# %%
dataset.push_to_hub("nimashoghi/mptrj")
