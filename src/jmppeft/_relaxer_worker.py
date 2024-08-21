import copy
import logging
from pathlib import Path
from typing import Any, Literal

import nshconfig as C
from nshconfig_extra import HFPath


class Config(C.Config):
    ckpt: Path | HFPath
    dest: Path
    idx_subset: Path | None = None
    num_items: int
    fmax: float = 0.05
    energy_key: Literal["s2e_energy", "s2re_energy"] = "s2e_energy"
    linref: bool = True
    ignore_if_exists: bool = True
    device_id: int | None = None
    save_traj: Path | None = None
    stress_weight: float = 0.1


def run(config: Config):
    if config.ignore_if_exists and config.dest.exists():
        logging.warning(f"Skipping {config.dest} as it already exists")
        return

    try:
        import os

        if config.device_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)

        import numpy as np
        import torch

        from jmppeft.utils import wbm_relax

        with wbm_relax.eval_context():
            setup = wbm_relax.RelaxSetup()
            # setup.device = torch.device(f"cuda:{config.device_id}")
            setup.dtype = torch.float32
            setup.relax_config.compute_stress = True
            setup.relax_config.stress_weight = config.stress_weight
            setup.relax_config.optimizer = "FIRE"
            setup.relax_config.fmax = config.fmax
            setup.relax_config.ase_filter = "exp"
            setup.energy_key = config.energy_key
            setup.idx_subset = config.idx_subset
            setup.save_traj = config.save_traj
            setup.linref = (
                np.load("/workspaces/repositories/jmp-peft/notebooks/mptrj_linref.npy")
                if config.linref
                else None
            )

            def update_hparams(hparams: dict[str, Any]):
                hparams = copy.deepcopy(hparams)
                hparams.pop("environment", None)
                hparams.pop("trainer", None)
                hparams.pop("runner", None)
                hparams.pop("directory", None)
                hparams.pop("ckpt_load", None)

                hparams.pop("pos_noise_augmentation", None)
                hparams.pop("dropout", None)
                hparams.pop("edge_dropout", None)

                return hparams

            if isinstance(ckpt := config.ckpt, HFPath):
                ckpt = ckpt.download()

            model = wbm_relax.load_ckpt(ckpt, setup, update_hparams)

            dl = wbm_relax.setup_dataset_and_loader(
                num_items=config.num_items, model=model, setup=setup
            )

            preds_targets = wbm_relax.relax_loop(dl, setup=setup, model=model)
            import dill

            with open(config.dest, "wb") as f:
                dill.dump(preds_targets, f)

    except Exception:
        logging.exception("An error occurred. Continuing...")
