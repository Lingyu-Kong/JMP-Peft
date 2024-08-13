import copy
import logging
from pathlib import Path
from typing import Any, Literal

import nshconfig as C


class Config(C.Config):
    ckpt: Path
    dest: Path
    num_items: int
    fmax: float = 0.05
    energy_key: Literal["s2e_energy", "s2re_energy"] = "s2e_energy"
    device_id: int | None = None


def run(config: Config):
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
            setup.relax_config.stress_weight = 0.1
            setup.relax_config.optimizer = "FIRE"
            setup.relax_config.fmax = config.fmax
            setup.relax_config.ase_filter = "exp"
            setup.energy_key = config.energy_key
            setup.linref = np.load(
                "/workspaces/repositories/jmp-peft/notebooks/mptrj_linref.npy"
            )

            def update_hparams(hparams: dict[str, Any]):
                hparams = copy.deepcopy(hparams)
                hparams.pop("environment", None)
                hparams.pop("trainer", None)
                hparams.pop("runner", None)
                hparams.pop("directory", None)
                hparams.pop("ckpt_load", None)
                return hparams

            model = wbm_relax.load_ckpt(
                ckpt_path=config.ckpt, setup=setup, update_hparams=update_hparams
            )

            dl = wbm_relax.setup_dataset_and_loader(
                num_items=config.num_items, model=model, setup=setup
            )

            preds_targets = wbm_relax.relax_loop(dl, setup=setup, model=model)
            import dill

            with open(config.dest, "wb") as f:
                dill.dump(preds_targets, f)

    except Exception:
        logging.exception("An error occurred. Continuing...")
