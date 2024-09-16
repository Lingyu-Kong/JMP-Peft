"""
Script to pretrain m3gnet with JMP-S data
"""
from typing import Literal
import nshtrainer.ll as ll
import nshutils as nu
from jmppeft.configs.pretrain.tasks import tasks_config_perlmutter_,tasks_config_oc20_4ktest_
from jmppeft.models.gemnet.config import BackboneConfig
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.pretrain import module as M
import wandb
from datetime import datetime
from datetime import timedelta
import argparse

"""
Set some Global Parameters for Pretraining
For most of the cases, we just need to change these Global Parameters
Change to parser parameters in the future, perhaps
"""


def main(args_dict):

    MODEL_TYPE: Literal["m3gnet_base", "m3gnet_large"] = args_dict["model_type"]
    USE_FSDP = args_dict["use_fsdp"]
    BATCH_SIZE = args_dict["batch_size"]
    NUM_WORKERS = args_dict["num_workers"]
    GRAD_CLIP = args_dict["grad_clip"]
    INIT_LR = args_dict["init_lr"]

    def base_config_(config: M.PretrainConfig):
        # Set the model trainer settings for maximum performance
        config.trainer.precision = "16-mixed-auto"
        config.trainer.set_float32_matmul_precision = "medium"

        # Optimizer settings
        config.optimizer = AdamWConfig(
            lr=INIT_LR,
            amsgrad=False,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        config.trainer.optimizer.log_grad_norm = True
        config.trainer.optimizer.gradient_clipping = ll.GradientClippingConfig(
            value=GRAD_CLIP,
            algorithm="value",
        )
        # LR Scheduler settings
        config.lr_scheduler = M.LinearWarmupCosineAnnealingSchedulerConfig(
            warmup_steps=2000,
            warmup_start_lr_factor=0.2,
            min_lr_factor=0.1,
            max_epochs=2,
        )
        # Regularization settings
        config.edge_dropout = 0.1
        # EMA settings
        config.ema = M.EMAConfig(decay=0.99)

        # Set data config
        config.batch_size = BATCH_SIZE
        config.num_workers = NUM_WORKERS

        # Set up the JMP MT dataset config and tasks
        config.mt_dataset = M.MTDatasetConfig(
            sample_type="temperature",
            sample_temperature=2.0,
        )
        
    def backbone_config_(config: M.PretrainConfig):
        config.name_parts.append(MODEL_TYPE) ## Component of the exp name
        if MODEL_TYPE == "m3gnet_base":
            config.backbone = M.M3GNetBackboneConfig.base()
        elif MODEL_TYPE == "m3gnet_large":
            config.backbone = M.M3GNetBackboneConfig.large()
        else:
            raise ValueError(f"Unknown model type: {MODEL_TYPE}, choose from ['m3gnet_base', 'm3gnet_large']")

    def fsdp_config_(config: M.PretrainConfig):
        ## Use fsdp
        config.fsdp = M.FSDPConfig(
            gradient_checkpointing=True,
        )
    
    def gradient_checkpointing_config_(config: M.PretrainConfig):
        config.gradient_checkpointing = True
        config.name_parts.append("gc")
        
    def profiling_config_(config: M.PretrainConfig):
        config.trainer.callbacks.append(ll.callbacks.EpochTimerConfig())
        config.trainer.callbacks.append(
            ll.callbacks.ThroughputMonitorConfig(batch_size=config.batch_size)
        )
        config.perf_metrics = True

    configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []
    config = M.PretrainConfig.draft()
    base_config_(config)
    if args_dict["run_test"]:
        tasks_config_oc20_4ktest_(config)
    else:
        tasks_config_perlmutter_(config)
    backbone_config_(config)
    gradient_checkpointing_config_(config)
    profiling_config_(config)
    config.runner.python_logging.log_level = "INFO"
    config.batch_size = BATCH_SIZE
    config.num_workers = NUM_WORKERS
    config = config.finalize()
    configs.append((config, M.PretrainModel)) ## TODO:Match model type in M.PretrainModel


    def run(config: M.PretrainConfig, model_cls: type[M.PretrainModel]):
        model = model_cls(config)
        trainer = ll.Trainer(config, **model.fsdp_trainer_kwargs())
        trainer.fit(model)

    runner = ll.Runner(run)
    nu.display(configs)
    if args_dict["run_test"]:
        runner.fast_dev_run(configs, n_batches=64, env={
            "CUDA_VISIBLE_DEVICES": args_dict["cuda_visible_devices"],
            "NSHUTILS_DISABLE_TYPECHECKING": "0", ## for debug, 0
        },)
    else:
        if args_dict["cluster"] == "local":
            raise ValueError("Local cluster not supported for pretraining")
            datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
            wandb.init(project="m3gnet_pretrain", name=MODEL_TYPE+"-"+datetime_str, config=args_dict)
            _ = runner.local(
                configs,
                env={
                    "CUDA_VISIBLE_DEVICES": args_dict["cuda_visible_devices"],
                    "NSHUTILS_DISABLE_TYPECHECKING": "1",
                },
            )
            wandb.finish()
        elif args_dict["cluster"] == "nersc":
            runner.submit_slurm(
                configs,
                {
                    "account": "m3641_g",
                    "qos": "preempt",
                    "constraint": "gpu",
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "time": timedelta(hours=48.0),
                },
                snapshot=True,
                setup_commands = ["wandb login 37f3de06380e350727df28b49712f8b7fe5b14aa",],
                env={
                    "LL_DISABLE_TYPECHECKING": "1",
                    "CUDA_VISIBLE_DEVICES": args_dict["cuda_visible_devices"],
                    "NSHUTILS_DISABLE_TYPECHECKING": "1",
                }
            )
        else:
            raise ValueError(f"Unknown cluster: {args_dict['cluster']}, choose from ['local', 'nersc']")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain m3gnet with JMP-S data')
    parser.add_argument('--model_type', type=str, default="m3gnet_base", help='Model type to use: m3gnet_base or m3gnet_large')
    parser.add_argument('--use_fsdp', type=bool, default=False, help='Use fsdp for training')
    parser.add_argument('--batch_size', type=int, default=120, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='Gradient clipping value')
    parser.add_argument('--init_lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--run_test', type=bool, default=False, help='Run a test session')
    parser.add_argument('--cluster', type=str, default="nersc", help='Cluster to run the experiment')
    parser.add_argument('--cuda_visible_devices', type=str, default="0,1,2,3", help='CUDA_VISIBLE_DEVICES')
    args_dict = vars(parser.parse_args())
    main(args_dict)