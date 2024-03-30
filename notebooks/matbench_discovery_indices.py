# %%
import pickle
from pathlib import Path

base_path = Path("/mnt/datasets/matbench-discovery-traj")
id_to_traj_len = pickle.load(open(base_path / "id_to_traj_lens.pkl", "rb"))
print(len(id_to_traj_len))

# %%
from tqdm.auto import tqdm

flattened_list: list[tuple[str, int]] = []

for id_, traj_len in tqdm(id_to_traj_len.items()):
    for traj_idx in range(traj_len):
        flattened_list.append((id_, traj_idx))

print(f"{len(flattened_list):,}")

# %%
train_ratio, val_ratio, test_ratio = 0.975, 0.0125, 0.0125

train_len = int(len(flattened_list) * train_ratio)
val_len = int(len(flattened_list) * val_ratio)
test_len = int(len(flattened_list) * test_ratio)
print(f"{train_len:,}, {val_len:,}, {test_len:,}")

# %%
import random

random.seed(42)
random.shuffle(flattened_list)

train_list = flattened_list[:train_len]
val_list = flattened_list[train_len : train_len + val_len]
test_list = flattened_list[train_len + val_len :]

print(f"{len(train_list):,}, {len(val_list):,}, {len(test_list):,}")
# %%
import pandas as pd


def to_df(data_list: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(data_list, columns=["id", "traj_idx"])


train_df = to_df(train_list)
val_df = to_df(val_list)
test_df = to_df(test_list)

from IPython.display import display

display(train_df.head())
display(val_df.head())
display(test_df.head())

# %%
# Save
save_path = base_path / "splits"
save_path.mkdir(exist_ok=True, parents=True)

train_df.to_csv(save_path / "train.csv", index=False)
val_df.to_csv(save_path / "val.csv", index=False)
test_df.to_csv(save_path / "test.csv", index=False)
