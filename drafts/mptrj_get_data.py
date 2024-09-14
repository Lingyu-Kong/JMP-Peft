from typing import Literal
from jmppeft.tasks.finetune import base

split: Literal["train", "val", "test"] = "train"
dataset_config = base.FinetuneMPTrjHuggingfaceDatasetConfig(
    split=split,
    energy_column_mapping={
        "y": "corrected_total_energy",
        "y_relaxed": "corrected_total_energy_relaxed",
    },
)

dataset = dataset_config.create_dataset()
print(dataset[0]["natoms"], type(dataset[0]["natoms"]))

