"""
Count the structures with specific number of elements
"""

from typing import Literal
from jmppeft.tasks.finetune import base
from ase.data import chemical_symbols
from tqdm import tqdm
import json

split: Literal["train", "val", "test"] = "train"
dataset_config = base.FinetuneMPTrjHuggingfaceDatasetConfig(
    split=split,
    energy_column_mapping={
        "y": "corrected_total_energy",
        "y_relaxed": "corrected_total_energy_relaxed",
    },
)

dataset = dataset_config.create_dataset()
chemical_symbols_count_k = {}
k = 3

pbar = tqdm(total=len(dataset))
for i in range(len(dataset)):
    data = dataset[i]
    atomic_numbers = data["atomic_numbers"].numpy()
    chemical_symbols_ = [chemical_symbols[number] for number in atomic_numbers]
    if len(set(chemical_symbols_)) == k:
        symbol_pair = tuple(set(sorted(chemical_symbols_)))
        if symbol_pair in chemical_symbols_count_k:
            chemical_symbols_count_k[symbol_pair] += 1
        else:
            chemical_symbols_count_k[symbol_pair] = 1
    pbar.update(1)
pbar.close()

## Sort the dictionary by value
chemical_symbols_count_k = dict(sorted(chemical_symbols_count_k.items(), key=lambda item: item[1], reverse=True))
## Turn keys from tuple to string
chemical_symbols_count_k = {str(key): value for key, value in chemical_symbols_count_k.items()}

print(chemical_symbols_count_k)

## export chemical_symbols_count_2 to json
with open(f"chemical_symbols_count_{k}.json", "w") as f:
    json.dump(chemical_symbols_count_k, f, indent=4)
    
