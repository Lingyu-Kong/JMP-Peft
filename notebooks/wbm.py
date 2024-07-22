# %%
import pandas as pd
from matbench_discovery.data import DATA_FILES, Key

df_initial_structures = pd.read_json(DATA_FILES.wbm_initial_structures).set_index(
    Key.mat_id
)
df_summary = pd.read_csv(DATA_FILES.wbm_summary).set_index(Key.mat_id)

# %%
# Merge the two dataframes
df = df_summary.merge(df_initial_structures, left_index=True, right_index=True)
# display(df.head())
print(df.head())

# %%
import datasets
import numpy as np
from pymatgen.core import Structure

datasets.disable_caching()


def generator():
    # Iterate through the dataframe rows
    for i, row in df.iterrows():
        # yield {**row.to_dict(), "id": i, "material_id": i}
        d = row.to_dict()
        d["id"] = i
        d["material_id"] = i

        structure = Structure.from_dict(d[Key.init_struct])
        d["frac_pos"] = structure.frac_coords
        d["cart_pos"] = structure.cart_coords
        d["pos"] = structure.cart_coords

        d["cell"] = structure.lattice.matrix.reshape(3, 3)
        d["num_atoms"] = len(structure)

        atomic_numbers = structure.atomic_numbers
        d["atomic_numbers"] = atomic_numbers
        d["composition"] = np.bincount(atomic_numbers, minlength=120).tolist()

        for site in d["initial_structure"].get("sites", []):
            if "properties" not in site:
                continue

            value = site.pop("properties", {})
            assert not value, f"Unexpected properties: {value}"

        yield d


dataset = datasets.Dataset.from_generator(generator)
dataset

# %%
dataset.push_to_hub("nimashoghi/wbm")
