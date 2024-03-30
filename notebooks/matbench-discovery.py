# %%
import matbench_discovery.data as mpd

mpd.DATA_FILES

# %%
loaded_datasets = {}
for k in ["all_mp_tasks"]:
    print(k)
    print(loaded := mpd.load(k))
    print()

    loaded_datasets[k] = loaded
    break

# %%
# Import pymatgen Structure

from monty.json import MontyDecoder

structure_dict = loaded_datasets["mp_computed_structure_entries"].iloc[0].entry
structure = MontyDecoder().process_decoded(structure_dict)

# %%
import rich

rich.print(structure_dict)

# %%
from pathlib import Path

base_path = Path("/root/.cache/matbench-discovery/1.0.0/mp/mp-tasks")
base_path.exists()

# %%
json_files = list(base_path.glob("*.json.gz"))
json_files


# %%
def load_json_gz(file_path):
    import gzip
    import json

    with gzip.open(file_path, "rt") as f:
        return json.load(f)


loaded_json = load_json_gz(json_files[0])
loaded_json

# %%
type(loaded_json)

loaded_json.keys()

# %%
mpd.DATA_FILES["mp_trj_extxyz_by_yuan"]


mpd.load("mp_trj_extxyz_by_yuan")
