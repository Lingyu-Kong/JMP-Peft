from functools import cache
from logging import getLogger
from typing import Any

import numpy as np
import pandas as pd
import torch
from matbench_discovery.data import DATA_FILES
from pymatgen.core import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import override

log = getLogger(__name__)


class Key:
    """Keys used to access dataframes columns."""

    arity = "arity", "Arity"
    bandgap_pbe = "bandgap_pbe", "PBE Band Gap"
    chem_sys = "chemical_system", "Chemical System"
    composition = "composition", "Composition"
    cse = "computed_structure_entry", "Computed Structure Entry"
    daf = "DAF", "Discovery Acceleration Factor"
    dft_energy = "uncorrected_energy", "DFT Energy"
    e_form = "e_form_per_atom_mp2020_corrected", "DFT E_form"
    e_form_pred = "e_form_per_atom_pred", "Predicted E_form"
    e_form_raw = "e_form_per_atom_uncorrected", "DFT E_form raw"
    e_form_wbm = "e_form_per_atom_wbm", "WBM E_form"
    each = "energy_above_hull", "E<sub>hull dist</sub>"
    each_pred = "e_above_hull_pred", "Predicted E<sub>hull dist</sub>"
    each_true = "e_above_hull_mp2020_corrected_ppd_mp", "E<sub>MP hull dist</sub>"
    each_wbm = "e_above_hull_wbm", "E<sub>WBM hull dist</sub>"
    final_struct = "relaxed_structure", "Relaxed Structure"
    forces = "forces", "Forces"
    form_energy = "formation_energy_per_atom", "Formation Energy (eV/atom)"
    formula = "formula", "Formula"
    init_struct = "initial_structure", "Initial Structure"
    magmoms = "magmoms", "Magnetic Moments"
    mat_id = "material_id", "Material ID"
    each_mean_models = "each_mean_models", "E<sub>hull dist</sub> mean of models"
    each_err_models = "each_err_models", "E<sub>hull dist</sub> mean error of models"
    model_std_each = "each_std_models", "Std. dev. over models"
    n_sites = "n_sites", "Number of Sites"
    site_nums = "site_nums", "Site Numbers", "Atomic numbers for each crystal site"
    spacegroup = "spacegroup", "Spacegroup"
    stress = "stress", "Stress"
    stress_trace = "stress_trace", "Stress Trace"
    struct = "structure", "Structure"
    task_id = "task_id", "Task ID"
    task_type = "task_type", "Task Type"
    train_task = "train_task", "Training Task"
    test_task = "test_task", "Test Task"
    targets = "targets", "Targets"
    # lowest WBM structures for a given prototype that isn't already in MP
    uniq_proto = "unique_prototype", "Unique Prototype"
    volume = "volume", "Volume (Å³)"
    wyckoff = "wyckoff_spglib", "Aflow-Wyckoff Label"  # relaxed structure Aflow label
    init_wyckoff = (
        "wyckoff_spglib_initial_structure",
        "Aflow-Wyckoff Label Initial Structure",
    )
    # number of structures in a model's training set
    train_set = "train_set", "Training Set"
    model_params = "model_params", "Model Params"  # model's parameter count
    model_type = "model_type", "Model Type"  # number of parameters in the model
    openness = "openness", "Openness"


class MatBenchDiscoveryIS2REDataset(Dataset[Data]):
    def __init__(self):
        super().__init__()

        self.df = pd.read_json(DATA_FILES.wbm_initial_structures).set_index(Key.mat_id)
        self.df_summary = pd.read_csv(DATA_FILES.wbm_summary).set_index(Key.mat_id)

    def __len__(self):
        return len(self.df)

    @override
    def __getitem__(self, idx: Any):
        row = self.df.iloc[idx]
        # Get the ID (`Key.mat_id`) and the initial structure (`Key.init_struct`)
        id_ = row.name

        summary_row = self.df_summary.loc[id_]
        structure = Structure.from_dict(row[Key.init_struct])

        data = Data(
            id=id_,
            pos=torch.tensor(structure.cart_coords, dtype=torch.float),
            atomic_numbers=torch.tensor(structure.atomic_numbers, dtype=torch.long),
            cell=torch.tensor(structure.lattice.matrix, dtype=torch.float).view(
                1, 3, 3
            ),
            y_formation=torch.tensor(summary_row[Key.e_form], dtype=torch.float),
            y_above_hull=torch.tensor(summary_row[Key.each_true], dtype=torch.float),
            natoms=torch.tensor(len(structure), dtype=torch.long),
        )

        return data

    @property
    @cache
    def atoms_metadata(self):
        log.critical("Computing atoms metadata...")
        atoms_metadata = (
            self.df[Key.init_struct].apply(lambda x: len(Structure.from_dict(x))).values
        )
        log.critical("Done computing atoms metadata.")
        return atoms_metadata

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]
