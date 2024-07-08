import argparse
import functools
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jmppeft.modules.dataset.dataset_transform as DT
import numpy as np
import torch
import torch.utils._pytree as tree
from jmppeft.modules.relaxer import ModelOutput, Relaxer, RelaxerConfig
from jmppeft.tasks.finetune import matbench_discovery as M
from jmppeft.tasks.finetune.base import FinetuneMatBenchDiscoveryIS2REDatasetConfig
from lightning.fabric.utilities.apply_func import move_data_to_device
from matbench_discovery.energy import get_e_form_per_atom
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric.data import Batch, Data
from tqdm import tqdm


def _create_model_and_dataset(
    *,
    ckpt_path: Path,
    dataset_config: FinetuneMatBenchDiscoveryIS2REDatasetConfig,
    default_dtype: torch.dtype = torch.float32,
):
    model = M.MatbenchDiscoveryModel.load_from_checkpoint(
        ckpt_path,
        map_location="cuda",
    )
    model = model.to(default_dtype).eval()

    def data_transform(data: Data):
        data = model.data_transform(data)
        data = Data.from_dict(
            tree.tree_map(
                lambda x: x.type(default_dtype)
                if torch.is_tensor(x) and torch.is_floating_point(x)
                else x,
                data.to_dict(),
            )
        )
        return data

    dataset = DT.transform(dataset_config.create_dataset(), data_transform)

    return model, dataset


def _composition(data: Batch):
    return dict(Counter(data.atomic_numbers.tolist()))


@torch.no_grad()
@torch.inference_mode()
def _model_fn(
    data,
    initial_data,
    *,
    model: M.MatbenchDiscoveryModel,
    add_correction: bool = False,
) -> ModelOutput:
    model_out = model.forward_denormalized(data)

    energy = model_out["y"]
    forces = model_out["force"]
    stress = model_out["stress"]

    # JMP-S v2 energy is corrected_energy, i.e., DFT total energy
    # This energy is now DFT total energy, we need to convert it to formation energy per atom
    energy = get_e_form_per_atom(
        {
            "composition": _composition(data),
            "energy": energy,
        }
    )
    assert isinstance(energy, torch.Tensor)

    # Add the correction factor
    if add_correction:
        energy += initial_data.y_formation_correction.item()

    energy = energy.view(1)
    forces = forces.view(-1, 3)
    stress = stress.view(1, 3, 3) if stress.numel() == 9 else stress.view(1, 6)

    return {
        "energy": energy,
        "forces": forces,
        "stress": stress,
    }


def _relax(
    model: M.MatbenchDiscoveryModel,
    dl: DataLoader,
    *,
    relaxer_config: RelaxerConfig,
):
    relaxer = Relaxer(
        config=relaxer_config,
        model=functools.partial(_model_fn, model=model),
        collate_fn=model.collate_fn,
        device=model.device,
    )

    preds_targets = defaultdict[str, list[tuple[float, float]]](lambda: [])
    corrections = []
    for data in tqdm(dl, total=len(dl)):
        data = move_data_to_device(data, model.device)
        relax_out = relaxer.relax(data, verbose=False)

        e_form_true = data.y_formation.item()
        e_form_pred = relax_out.atoms.get_total_energy()
        preds_targets["e_form"].append((e_form_pred, e_form_true))

        e_above_hull_true = data.y_above_hull.item()
        e_above_hull_pred = e_above_hull_true + (e_form_pred - e_form_true)
        preds_targets["e_above_hull"].append((e_above_hull_pred, e_above_hull_true))

        corrections.append(data.y_formation_correction.item())

    corrections = np.array(corrections)

    e_form_true, e_form_pred = zip(*preds_targets["e_form"])
    e_form_true = np.array(e_form_true)
    e_form_pred = np.array(e_form_pred)

    e_above_hull_true, e_above_hull_pred = zip(*preds_targets["e_above_hull"])
    e_above_hull_true = np.array(e_above_hull_true)
    e_above_hull_pred = np.array(e_above_hull_pred)

    return (
        (e_form_true, e_form_pred),
        (e_above_hull_true, e_above_hull_pred),
        corrections,
    )


def relax_main(args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    model, dataset = _create_model_and_dataset(
        ckpt_path=args.ckpt_path,
        dataset_config=FinetuneMatBenchDiscoveryIS2REDatasetConfig(),
        default_dtype=torch.float64 if args.use_fp64 else torch.float32,
    )

    dl = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=model.collate_fn,
        sampler=DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        ),
        num_workers=1,
        pin_memory=True,
    )

    (
        (e_form_true, e_form_pred),
        (e_above_hull_true, e_above_hull_pred),
        corrections,
    ) = _relax(
        model,
        dl,
        relaxer_config=RelaxerConfig(
            compute_stress=True,
            stress_weight=0.1,
            optimizer="FIRE",
            fmax=0.05,
            ase_filter="frechet",
        ),
    )

    base_save_dir: Path = args.save_dir
    base_save_dir.mkdir(parents=True, exist_ok=True)

    npz_path = base_save_dir / "result.npz"
    np.savez(
        npz_path,
        e_form_true=e_form_true,
        e_form_pred=e_form_pred,
        e_above_hull_true=e_above_hull_true,
        e_above_hull_pred=e_above_hull_pred,
        corrections=corrections,
    )


def generate_main(args: argparse.Namespace):
    # This command just generates a list of commands to run for each worker.
    # The user can then run these commands in parallel on multiple nodes.
    # The commands will be wrapped with `screen` to run in the background.

    def _python_command_str(rank: int):
        # Partition the CUDA_VISIBLE_DEVICES environment variable to each worker
        device_idx = args.available_gpu_indices[rank % len(args.available_gpu_indices)]

        use_fp64 = "--use-fp64" if args.use_fp64 else "--no-use-fp64"

        save_dir = args.save_dir / f"rank-{rank}"
        return f"conda run --prefix {sys.prefix} --live-stream python {__file__} relax --ckpt-path {args.ckpt_path} --save-dir {save_dir} --world-size {args.world_size} --rank {rank} {use_fp64} --gpu-index {device_idx}"

    for rank in range(args.world_size):
        print(f"screen -dmS relaxer-{rank} {_python_command_str(rank)}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="valid subcommands",
        help="additional help",
        required=True,
    )

    # `relax` subcommand
    relax_parser = subparsers.add_parser("relax")
    relax_parser.set_defaults(fn=relax_main)
    relax_parser.add_argument(
        "--ckpt-path", type=Path, required=True, help="Path to the checkpoint"
    )
    relax_parser.add_argument(
        "--save-dir", type=Path, required=True, help="Path to the save directory"
    )
    relax_parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    relax_parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    relax_parser.add_argument(
        "--use-fp64",
        action=argparse.BooleanOptionalAction,
        help="Use 64-bit floating point precision",
        default=True,
    )
    relax_parser.add_argument(
        "--gpu-index",
        type=int,
        help="GPU index for single GPU training",
        required=True,
    )

    # `generate` subcommand
    generate_parser = subparsers.add_parser("generate")
    generate_parser.set_defaults(fn=generate_main)
    generate_parser.add_argument(
        "--ckpt-path", type=Path, required=True, help="Path to the checkpoint"
    )
    generate_parser.add_argument(
        "--save-dir", type=Path, required=True, help="Path to the save directory"
    )
    generate_parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    generate_parser.add_argument(
        "--use-fp64",
        action=argparse.BooleanOptionalAction,
        help="Use 64-bit floating point precision",
        default=True,
    )
    generate_parser.add_argument(
        "--available-gpu-indices",
        nargs="+",
        type=int,
        help="List of available GPU indices",
        required=True,
    )
    args = parser.parse_args()

    args.fn(args)


if __name__ == "__main__":
    main()
