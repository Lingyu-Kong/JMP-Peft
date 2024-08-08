import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import dill
import nshutils
import numpy as np
import torch


def worker(args: argparse.Namespace):
    from jmppeft.utils import wbm_relax

    with wbm_relax.eval_context():
        setup = wbm_relax.RelaxSetup()
        setup.device = torch.device(f"cuda:{args.device_id}")
        setup.dtype = torch.float32
        setup.relax_config.compute_stress = True
        setup.relax_config.stress_weight = 0.1
        setup.relax_config.optimizer = "FIRE"
        setup.relax_config.fmax = 0.05
        setup.relax_config.ase_filter = "exp"
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
            ckpt_path=args.ckpt, setup=setup, update_hparams=update_hparams
        )

        dl = wbm_relax.setup_dataset_and_loader(
            num_items=args.num_items, model=model, setup=setup
        )

        preds_targets = wbm_relax.relax_loop(dl, setup=setup, model=model)
        with open(args.dest, "wb") as f:
            dill.dump(preds_targets, f)


def main():
    nshutils.pretty(rich_log_handler=False)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--dest", type=Path, required=True, help="Path to save results")
    parser.add_argument(
        "--num-items", type=int, required=True, help="Number of items in dataset"
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device ID")
    args = parser.parse_args()

    if not args.ckpt.exists():
        parser.error(f"Checkpoint {args.ckpt} does not exist")

    if args.dest.exists():
        parser.error(f"Destination {args.dest} already exists")

    worker(args)


if __name__ == "__main__":
    main()
