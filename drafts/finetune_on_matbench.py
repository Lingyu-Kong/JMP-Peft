import argparse
from collections.abc import Callable
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import nshtrainer as nt
import nshutils as nu
import wandb

from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.jmp_s import jmp_s_ft_config_
from jmppeft.datasets import mptrj_hf
from jmppeft.datasets.matbench import MatBenchDatasetConfig
from jmppeft.modules import loss
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune import base, output_head
from jmppeft.tasks.finetune import matbench as M
from jmppeft.utils.param_specific_util import (
    make_parameter_specific_optimizer_config,
    parameter_specific_optimizer_config,
)

logging.basicConfig(level=logging.WARNING)
nu.pretty()

def main(args_dict: dict):
    ## BackBone Model Config
    def jmp_(config: base.FinetuneConfigBase):
        if args_dict["model"] == "jmp_l":
            ckpt_path = Path(
                "/net/csefiles/coc-fung-cluster/lingyu/checkpoints/jmp-l.pt"
            )
            jmp_l_ft_config_(config)
            config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
                path=ckpt_path, ema=True
            )
            config.meta["jmp_kind"] = "l"
        elif args_dict["model"] == "jmp_s":
            ckpt_path = Path(
                "/net/csefiles/coc-fung-cluster/lingyu/checkpoints/jmp-s.pt"
            )
            jmp_s_ft_config_(config)
            config.ckpt_load.checkpoint = base.PretrainedCheckpointConfig(
                path=ckpt_path, ema=True
            )
            config.meta["jmp_kind"] = "s"
        else:
            raise ValueError(
                "Invalid Model Name, Please choose between jmp_l and jmp_s"
            )

    ## Predict Forces Directly
    def direct_(config: base.FinetuneConfigBase):
        config.backbone.regress_forces = True
        config.backbone.direct_forces = True
        config.backbone.regress_energy = True
        
        config.trainer.inference_mode = False

    # ## Predict Forces with Gradient Method
    # def grad_(config: base.FinetuneConfigBase):
    #     config.backbone.regress_forces = True
    #     config.backbone.direct_forces = True
    #     config.backbone.regress_energy = True

    #     config.trainer.inference_mode = False

    ## Data Config
    def data_config_(
        config: M.MatbenchConfig,
        *,
        batch_size: int,
    ):
        config.batch_size = batch_size
        config.name_parts.append(f"bsz{batch_size}")

        def dataset_fn(split: Literal["train", "val", "test"]):
            dataset_config = MatBenchDatasetConfig(
                task = args_dict["task"],
                fold_idx = args_dict["fold"],
                split = split,
                split_ratio=args_dict["split_ratio"],
            )
            return dataset_config

        config.train_dataset = dataset_fn("train")
        config.val_dataset = dataset_fn("val")
        config.test_dataset = dataset_fn("test")

        # Set data config
        # config.num_workers = 0

        # Balanced batch sampler
        config.use_balanced_batch_sampler = True
        config.trainer.use_distributed_sampler = False

    def output_heads_config_direct_(
        config: M.MatbenchConfig,
    ):
        target_loss = loss.MAELossConfig()
        # target head
        config.graph_targets.append(
            output_head.GraphScalarTargetConfig(
                name=args_dict["task"],
                loss=target_loss,
                reduction="sum",
            )
        )
        return config

    ## Optimizer Config
    def optimization_config_(
        config: M.MatbenchConfig,
        *,
        lr: float,
        wd: float,
    ):
        config.optimizer = AdamWConfig(
            lr=lr,
            amsgrad=False,
            betas=(0.9, 0.95),
            weight_decay=wd,
        )
        config.lr_scheduler = base.WarmupCosRLPConfig(
            warmup_epochs=1,
            warmup_start_lr_factor=1.0e-1,
            should_restart=False,
            max_epochs=128,
            min_lr_factor=0.5,
            rlp=base.RLPConfig(patience=5, factor=0.8),
        )
        config.trainer.optimizer.gradient_clipping = nt.model.GradientClippingConfig(
            value=2.0,
            algorithm="value",
        )

        config.name_parts.append(f"lr{lr}")
        config.name_parts.append(f"wd{wd}")

    ## Freeze the backbone
    def parameter_specific_optimizers_(config: base.FinetuneConfigBase):
        if config.parameter_specific_optimizers is None:
            config.parameter_specific_optimizers = []

        match config.meta["jmp_kind"]:
            case "l":
                config.parameter_specific_optimizers.extend(
                    make_parameter_specific_optimizer_config(
                        config,
                        config.backbone.num_blocks,
                        {
                            "embedding": 0.3,
                            "blocks_0": 0.55,
                            "blocks_1": 0.40,
                            "blocks_2": 0.30,
                            "blocks_3": 0.40,
                            "blocks_4": 0.55,
                            "blocks_5": 0.625,
                        },
                    )
                )
            case "s":
                config.parameter_specific_optimizers.extend(
                    make_parameter_specific_optimizer_config(
                        config,
                        config.backbone.num_blocks,
                        {
                            "embedding": 0.3,
                            "blocks_0": 0.30,
                            "blocks_1": 0.40,
                            "blocks_2": 0.55,
                            "blocks_3": 0.625,
                        },
                    )
                )
            case _:
                raise ValueError(f"Invalid jmp_kind: {config.meta['jmp_kind']}")

    ## Layer Norm
    def ln_(
        config: base.FinetuneConfigBase,
        *,
        lr_multiplier: float | None,
    ):
        config.backbone.ln_per_layer = True
        config.backbone.scale_factor_to_ln = True

        if lr_multiplier is not None:
            if config.parameter_specific_optimizers is None:
                config.parameter_specific_optimizers = []

            config.parameter_specific_optimizers = [
                *parameter_specific_optimizer_config(
                    config,
                    [
                        {
                            "name": "ln",
                            "lr_multiplier": lr_multiplier,
                            "parameter_patterns": [
                                "backbone.h_lns.*",
                                "backbone.m_lns.*",
                                "backbone.*.scale*.ln.*",
                            ],
                        }
                    ],
                ),
                *config.parameter_specific_optimizers,
            ]

        config.name_parts.append("ln")

    def create_config(config_fn: Callable[[M.MatbenchConfig], None]):
        config = M.MatbenchConfig.draft()
        config.trainer.precision = "16-mixed-auto"
        config.trainer.set_float32_matmul_precision = "medium"
        config.project = "jmp_matbench"
        config.name = args_dict["task"]
        config_fn(config)
        config.backbone.qint_tags = [0, 1, 2]
        config.primary_metric = nt.MetricConfig(
            name=f"matbench_discovery/{args_dict['task']}_mae", mode="min"
        )
        return config

    configs: list[tuple[M.MatbenchConfig, type[M.MatbenchModel]]] = []

    config = create_config(jmp_)
    if args_dict["freeze_backbone"]:
        config.freeze = base.FreezeConfig(
            backbone=True,
            embedding=True,
        )
    config.parameter_specific_optimizers = []
    # config.max_neighbors = M.MaxNeighbors(main=25, aeaint=20, aint=1000, qint=8)
    # config.cutoffs = M.Cutoffs.from_constant(args_dict["cutoff"])
    config.dataset = args_dict["task"].replace("matbench_", "")
    data_config_(config, batch_size=args_dict["batch_size"])
    optimization_config_(config, lr=args_dict["lr"], wd=args_dict["weight_decay"])
    direct_(config)
    output_heads_config_direct_(config,)

    if not args_dict["freeze_backbone"]:
        parameter_specific_optimizers_(config)
        ln_(config, lr_multiplier=1.5)
    config.per_graph_radius_graph = True
    # config.ignore_graph_generation_errors = True
    config.trainer.early_stopping = nt.model.EarlyStoppingConfig(
        patience=args_dict["earlystop_patience"], min_lr=1e-08
    )
    config.trainer.max_epochs = args_dict["max_epochs"]
    config = config.finalize()
    configs.append((config, M.MatbenchModel))

    def run(
        config: M.MatbenchConfig, model_cls: type[M.MatbenchModel]
    ) -> None:
        model = model_cls.construct_and_load_checkpoint(config)
        if args_dict["num_blocks"] is not None:
            model.backbone.num_blocks = args_dict["num_blocks"]
        if args_dict["padding_method"] == "zero":
            model.backbone.padding_method = "zero"
        elif args_dict["padding_method"] == "repeat":
            model.backbone.padding_method = "repeat"
        else:
            raise ValueError("Invalid Padding Method")
        trainer = nt.Trainer(config)
        trainer.fit(model)
        ## Save the model

    runner = nt.Runner(run)
    if args_dict["run_as_test"]:
        nu.display(configs)
        runner.fast_dev_run(
            configs,
            n_batches=128,
            env={
                "CUDA_VISIBLE_DEVICES": args_dict["gpu"],
                "NSHUTILS_DISABLE_TYPECHECKING": "1",  ## for debug, 0
            },
        )  ## snapshot=True, then we can change code without affecting the running jobs
    else:
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
        wandb.init(project="jmp_mptrj", name="mptrj_" + datetime_str, config=args_dict)
        nu.display(configs)
        _ = runner.local(
            configs,
            env={
                "CUDA_VISIBLE_DEVICES": args_dict["gpu"],
                "NSHUTILS_DISABLE_TYPECHECKING": "1",
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="jmp_s", help="Model Name")
    parser.add_argument("--freeze_backbone", type=bool, default=False, help="Freeze Backbone")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of Blocks used for Backbone")
    parser.add_argument("--padding_method", type=str, default="zero", help="Padding Method")
    parser.add_argument("--cutoff", type=float, default=12.0, help="Cutoff")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Split Ratio")
    parser.add_argument("--lr", type=float, default=8.0e-5, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight Decay")
    parser.add_argument("--task", type=str, default="matbench_mp_gap", help="Task Name")
    parser.add_argument("--fold", type=int, default=0, help="Fold")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--earlystop_patience", type=int, default=200, help="Early Stop Patience")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Max Epoch")
    parser.add_argument("--run_as_test", type=bool, default=True, help="Run as Test")
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
