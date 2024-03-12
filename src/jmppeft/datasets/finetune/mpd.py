# %%
import matbench_discovery.data as mpd
import pandas as pd
from matbench_discovery import Key
from matbench_discovery.data import DATA_FILES
from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from tqdm import tqdm

target_col = Key.form_energy
input_col = "atoms"

# %%
mpd.load(mpd.DATA_FILES.mp_trj_extxyz_by_yuan)

# %%
df_cse = pd.read_json(DATA_FILES.mp_computed_structure_entries).set_index(Key.mat_id)
df_cse[Key.struct] = [
    Structure.from_dict(cse[Key.struct])
    for cse in tqdm(df_cse.entry, desc="Structures from dict")
]

# load energies
df_in = pd.read_csv(DATA_FILES.mp_energies).set_index(Key.mat_id)
df_in[Key.struct] = df_cse[Key.struct]
assert target_col in df_in

df_in[input_col] = df_in[Key.struct]
df_in[input_col] = [
    JarvisAtomsAdaptor.get_atoms(struct)
    for struct in tqdm(df_in[Key.struct], desc="Converting to JARVIS atoms")
]
