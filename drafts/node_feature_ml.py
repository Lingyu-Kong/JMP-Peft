from collections.abc import Callable
from pathlib import Path
from typing import Literal, Any, IO
import copy
import torch
import nshtrainer as nt
from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.jmp_s import jmp_s_ft_config_
from jmppeft.tasks.finetune import base, output_head
from jmppeft.tasks.finetune import matbench_discovery as M
from jmppeft.datasets.mptrj_hf import MPTrjDatasetFromXYZ, MPTrjDatasetFromXYZConfig
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData
from ase import Atoms
from ase.io import write
import numpy as np
from tqdm import tqdm
import argparse


def LoadModel(args_dict:dict):
    ckpt_path = Path(args_dict['ckpt_path'])
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    device = torch.device(args_dict['device'])
    if args_dict["ckpt_path"].endswith(".pt") and "jmp" in args_dict["ckpt_path"]:
        def jmp_(config: base.FinetuneConfigBase):
            if "-s" in args_dict["ckpt_path"]:
                jmp_s_ft_config_(config)
                config.meta["jmp_kind"] = "s"
            elif "-l" in args_dict["ckpt_path"]:
                jmp_l_ft_config_(config)
                config.meta["jmp_kind"] = "l"
            else:
                raise ValueError(f"Invalid Model Type: {args_dict['ckpt_path']}")
            config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
                path=ckpt_path, ema=True
            )
            
        def direct_(config: base.FinetuneConfigBase):
            config.backbone.regress_forces = True
            config.backbone.direct_forces = True
            config.backbone.regress_energy = True
            
        def data_config_(
            config: M.MatbenchDiscoveryConfig,
            *,
            batch_size: int,
            reference: bool,
        ):
            config.batch_size = batch_size
            # config.name_parts.append(f"bsz{batch_size}")

            def dataset_fn(split: Literal["train", "val", "test"]):
                return base.FinetuneMPTrjHuggingfaceDatasetConfig(
                    split=split,
                    energy_column_mapping={
                        "y": "corrected_total_energy",
                        "y_relaxed": "corrected_total_energy_relaxed",
                    },
                )

            config.train_dataset = dataset_fn("train")
            config.val_dataset = dataset_fn("val")
            config.test_dataset = dataset_fn("test")

            # Set data config
            config.num_workers = 7

            # Balanced batch sampler
            config.use_balanced_batch_sampler = True
            config.trainer.use_distributed_sampler = False

        def output_heads_config_(
            config: M.MatbenchDiscoveryConfig,
        ):
            config.node_targets.append(output_head.DirectNodeFeatureTargetConfig(name="node_vector"))

        def create_config(config_fn: Callable[[M.MatbenchDiscoveryConfig], None]):
            config = M.MatbenchDiscoveryConfig.draft()

            config.trainer.precision = "16-mixed-auto"
            config.trainer.set_float32_matmul_precision = "medium"

            config.project = "jmp_mptrj"
            config.name = "mptrj"
            config_fn(config)
            config.backbone.qint_tags = [0, 1, 2]

            config.primary_metric = nt.MetricConfig(
                name="matbench_discovery/force_mae", mode="min"
            )
            return config
        
        config = create_config(jmp_)
        config.parameter_specific_optimizers = []
        config.max_neighbors = M.MaxNeighbors(main=25, aeaint=20, aint=1000, qint=8)
        config.cutoffs = M.Cutoffs.from_constant(12.0)
        data_config_(config, reference=False, batch_size=50)
        direct_(config=config)
        output_heads_config_(
            config,
        )
        config.per_graph_radius_graph = True
        config.ignore_graph_generation_errors = True
        config = config.finalize()
        model = M.MatbenchDiscoveryModel.construct_and_load_checkpoint(config)
        model.to(device)
    elif args_dict["ckpt_path"].endswith(".ckpt"):
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

        def load_ckpt(
            ckpt: str | Path | IO,
            update_hparams: Callable[[dict[str, Any]], dict[str, Any]] | None,
            device: torch.device | str | None = None,
        ):
            hparams = None
            if update_hparams:
                hparams = torch.load(ckpt, map_location="cpu")["hyper_parameters"]
                hparams = update_hparams(hparams)

            model = M.MatbenchDiscoveryModel.load_checkpoint(
                ckpt,
                map_location=device,
                hparams=hparams,
            )
            return model
        
        model = load_ckpt(ckpt_path, update_hparams, device=device)
    else:
        raise ValueError(f"Invalid Checkpoint Type: {args_dict['ckpt_path']}")
    
    model.eval()
    return model, device


def LoadData(args_dict:dict):
    def collate_fn(data_list: list[BaseData]):
        return Batch.from_data_list(data_list)

    dataset_config = MPTrjDatasetFromXYZConfig(
        file_path=args_dict['xyz_files'],
        split="all",
    )
    dataset = MPTrjDatasetFromXYZ(dataset_config)
    dataloader = DataLoader(dataset, batch_size=args_dict['batch_size'], shuffle=False, num_workers=args_dict['num_workers'], collate_fn=collate_fn)
    return dataloader
    

def main(args_dict:dict):
    model, device = LoadModel(args_dict)
    dataloader = LoadData(args_dict)
    
    max_num_blocks = model.backbone.num_blocks
    for num_blocks in range(args_dict["min_block"], max_num_blocks+1):
        atoms_list = []
        pbar = tqdm(total=len(dataloader), desc="Generating node features for {} blocks".format(num_blocks))
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                node_features_ = model.get_node_features(batch, num_blocks=num_blocks)
                node_features_ = node_features_.cpu().numpy()
                natoms_ = batch.natoms.cpu().numpy()
                atomic_numbers = batch.atomic_numbers.cpu().numpy()
                pos = batch.pos.cpu().numpy()
                cell = batch.cell.cpu().numpy()
                for i in range(len(batch.idx)):
                    atoms = Atoms(
                        numbers=atomic_numbers[np.sum(natoms_[:i]):np.sum(natoms_[:i+1])],
                        positions=pos[np.sum(natoms_[:i]):np.sum(natoms_[:i+1])],
                        cell=cell[i],
                        pbc=True,
                    )
                    atoms.info["node_features"] = node_features_[np.sum(natoms_[:i]):np.sum(natoms_[:i+1])]
                    atoms_list.append(atoms)
                pbar.update(1)
        pbar.close()
        file_name = args_dict["xyz_files"].split("/")[-1].split(".")[0]
        model_name = args_dict["ckpt_path"].split("/")[-1].split(".")[0]
        file_name_ = f"{model_name}-block{num_blocks}-{file_name}"
        write(args_dict["xyz_files"].replace(file_name, file_name_), atoms_list)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Load a pre-trained/fine-tuned model and generate node features")
    parser.add_argument("--ckpt_path", type=str, default="../checkpoints/jmp-l.pt", help="Path to the checkpoint file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    parser.add_argument("--xyz_files", type=str, default="./temp_data/MgSi-mptrj.xyz", help="Path to the xyz files")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the data loader")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the data loader")
    parser.add_argument("--min_block", type=int, default=1, help="Minimum number of blocks to generate node features")
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)