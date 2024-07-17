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

with open("data/mptrj_ids.json", "r") as f:
    ids = frozenset[str](json.load(f))


def filter(mp_id: str, *, all_ids: frozenset[str]):
    return mp_id in all_ids


dataset = dataset.filter(filter, input_columns=["mp_id"], fn_kwargs={"all_ids": ids})


# %%
import numpy as np

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
