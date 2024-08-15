import contextlib
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sized
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import IO, Any, Literal, cast

import numpy as np
import torch
import torch.utils._pytree as tree
from jmppeft.modules.relaxer import ModelOutput, Relaxer, RelaxerConfig
from lightning.fabric.utilities.apply_func import move_data_to_device
from matbench_discovery.energy import get_e_form_per_atom
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm

from ..modules.dataset import dataset_transform as DT
from ..tasks.finetune import matbench_discovery as M
from ..tasks.finetune.base import (
    FinetuneMatBenchDiscoveryIS2REDatasetConfig,
    MatBenchDiscoveryIS2REDataset,
)

log = logging.getLogger(__name__)


@contextlib.contextmanager
def eval_context():
    with torch.no_grad(), torch.inference_mode():
        yield


@dataclass
class RelaxSetup:
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    dtype: torch.dtype = torch.float32
    relax_config: RelaxerConfig = field(default_factory=lambda: RelaxerConfig())
    linref: np.ndarray | None = None
    energy_key: Literal["s2e_energy", "s2re_energy"] = "s2e_energy"


def load_ckpt(
    ckpt: str | Path | IO,
    setup: RelaxSetup,
    update_hparams: Callable[[dict[str, Any]], dict[str, Any]] | None,
):
    hparams = None
    if update_hparams:
        hparams = torch.load(ckpt, map_location="cpu")["hyper_parameters"]
        hparams = update_hparams(hparams)

    model = M.MatbenchDiscoveryModel.load_checkpoint(
        ckpt,
        map_location=setup.device,
        hparams=hparams,
    )
    model = model.to(setup.device, dtype=setup.dtype).eval()
    return model


def data_transform(data: Data, *, model: M.MatbenchDiscoveryModel, setup: RelaxSetup):
    data = model.data_transform(data)
    data = Data.from_dict(
        tree.tree_map(
            lambda x: x.type(setup.dtype)
            if torch.is_tensor(x) and torch.is_floating_point(x)
            else x,
            data.to_dict(),
        )
    )
    return data


def setup_dataset(
    num_items: int,
    *,
    model: M.MatbenchDiscoveryModel,
    setup: RelaxSetup,
):
    dataset_config = FinetuneMatBenchDiscoveryIS2REDatasetConfig()
    dataset = dataset_config.create_dataset()

    dataset = DT.transform(dataset, partial(data_transform, model=model, setup=setup))
    dataset = DT.sample_n_transform(dataset, n=num_items, seed=42)

    return dataset


def setup_dataloader(
    dataset: MatBenchDiscoveryIS2REDataset,
    *,
    model: M.MatbenchDiscoveryModel,
):
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=model.collate_fn,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def setup_dataset_and_loader(
    num_items: int,
    *,
    model: M.MatbenchDiscoveryModel,
    setup: RelaxSetup,
):
    dataset = setup_dataset(num_items, model=model, setup=setup)
    loader = setup_dataloader(dataset, model=model)
    return loader


def _composition(data: Batch):
    return dict(Counter(data.atomic_numbers.tolist()))


def model_fn(
    data,
    initial_data,
    *,
    model: M.MatbenchDiscoveryModel,
    setup: RelaxSetup,
):
    model_out = model.forward_denormalized(data)

    # energy = model_out["y_relaxed"] if use_y_relaxed else model_out["y"]
    energy = model_out["y"]
    relaxed_energy = model_out["y_relaxed"]
    forces = model_out["force"]
    stress = model_out["stress"]

    # Undo the linref
    if setup.linref is not None:
        energy = energy + setup.linref[data.atomic_numbers.cpu().numpy()].sum()
        relaxed_energy = (
            relaxed_energy + setup.linref[data.atomic_numbers.cpu().numpy()].sum()
        )

    # JMP-S v2 energy is corrected_energy, i.e., DFT total energy
    # This energy is now DFT total energy, we need to convert it to formation energy per atom
    energy = get_e_form_per_atom(
        {
            "composition": _composition(data),
            "energy": energy,
        }
    )
    relaxed_energy = get_e_form_per_atom(
        {
            "composition": _composition(data),
            "energy": relaxed_energy,
        }
    )
    assert isinstance(energy, torch.Tensor)
    assert isinstance(relaxed_energy, torch.Tensor)

    # energy, relaxed_energy = tree.tree_map(
    #     lambda energy: energy.view(1), (energy, relaxed_energy)
    # )
    energy = energy.view(1)
    relaxed_energy = relaxed_energy.view(1)
    forces = forces.view(-1, 3)
    stress = stress.view(1, 3, 3) if stress.numel() == 9 else stress.view(1, 6)

    energy_dict = {"s2e_energy": energy, "s2re_energy": relaxed_energy}
    return cast(
        ModelOutput,
        {
            **energy_dict,
            "energy": energy_dict[setup.energy_key],
            "forces": forces,
            "stress": stress,
        },
    )


def relax_loop(
    dl: Iterable[Batch],
    *,
    setup: RelaxSetup,
    model: M.MatbenchDiscoveryModel,
):
    relaxer = Relaxer(
        config=setup.relax_config,
        model=partial(model_fn, model=model, setup=setup),
        collate_fn=model.collate_fn,
        device=model.device,
    )

    preds_targets = defaultdict[str, list[tuple[float, float]]](lambda: [])
    mae_error = 0.0
    mae_count = 0

    for data in tqdm(dl, total=len(dl) if isinstance(dl, Sized) else None):
        data = cast(Batch, data)
        data = move_data_to_device(data, model.device)
        data.y_prediction = data.y_formation
        relaxed_data, relax_out = relaxer.relax_and_return_structure(
            data,
            device=setup.device,
            verbose=False,
        )

        e_form_true = data.y_formation.item()
        e_form_pred = relax_out.atoms.get_total_energy()
        preds_targets["e_form"].append((e_form_pred, e_form_true))

        e_above_hull_true = data.y_above_hull.item()
        e_above_hull_pred = e_above_hull_true + (e_form_pred - e_form_true)
        preds_targets["e_above_hull"].append((e_above_hull_pred, e_above_hull_true))

        error = abs(e_form_pred - e_form_true)
        mae_error += error
        mae_count += 1
        mae_running = mae_error / mae_count

        nsteps = len(relax_out.trajectory.frames)

        log.info(
            f"# Steps: {nsteps}; e_form: P={e_form_pred:.4f}, GT={e_form_true:.4f}, Δ={error:.4f}, MAE={mae_running:.4f}"
        )

    return preds_targets
