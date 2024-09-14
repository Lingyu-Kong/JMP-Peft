"""
Train a JMP model from scratch
"""
import nshtrainer.ll as ll
from jmppeft.models.gemnet.config import BackboneConfig
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.pretrain import module as M
from pathlib import Path


"""
Set some Global Parameters for Pretraining
For most of the cases, we just need to change these Global Parameters
Change to parser parameters in the future, perhaps
"""
MODEL_TYPE = "gemnet_base" ## Choose from ["gemnet_base", "gemnet_large", "gemnet_xl","torchmd","graphormer"]
USE_FSDP = False ## Use Fully Sharded Data Parallelism
BATCH_SIZE = 24 ## Batch Size per gpu
NUM_WORKERS = 4 ## Number of Workers for DataLoader
GRAD_CLIP = 2.0 ## Gradient Clipping Value, 1.0 for torchmd, 2.0 for gemnet
# TODO: data path?

def base_config_(config: M.PretrainConfig):
    # Set the model trainer settings for maximum performance
    config.trainer.precision = "16-mixed-auto"
    config.trainer.set_float32_matmul_precision = "medium"

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=3.0e-4,
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
    

def backbone_config_(
    config: M.PretrainConfig,
):
    config.name_parts.append(MODEL_TYPE) ## Component of the exp name
    if MODEL_TYPE == "torchmd":
        config.dropout = None ## TODO:These two necessary? Why?
        config.edge_dropout = None ## TODO:These two necessary? Why?
        config.backbone = M.TorchMDNetBackboneConfig()
        # config.backbone.apply_1_5M_param_config_() ## Adjust model size
        config.exclude_keys.remove("edge_index") ## Torchmd needs edge_index
    elif MODEL_TYPE == "gemnet_base":
        config.backbone = M.GOCBackboneConfig.base()
        config.backbone.scale_basis = False # TODO: What is this for? List all possible settings here
    elif MODEL_TYPE == "gemnet_large":
        config.backbone = M.GOCBackboneConfig.large()
        config.backbone.scale_basis = False
    elif MODEL_TYPE == "gemnet_xl":
        config.backbone = M.GOCBackboneConfig.xl()
        config.backbone.scale_basis = False
    elif MODEL_TYPE == "graphormer":
        pass
    else:
        raise ValueError(f"Invalid Model Type: {MODEL_TYPE}")
    
def tasks_config_(config: M.PretrainConfig):
    config.tasks = [
        M.TaskConfig(
            name="MgSi-mptrj",
            train_dataset=M.PretrainDatasetConfig(
                src=base_dir / "oc20/s2ef/200k/train/",
                metadata_path=base_dir / "oc20/s2ef/200k/metadata.npz",
                lin_ref=base_dir / "oc20/lin_ref_coeffs.npz",
                sample_ratio=DatasetSampleRatioConfig(sample_ratio=0.01, seed=sample_seed),
            ),
            val_dataset=M.PretrainDatasetConfig(
                src=base_dir / "oc20/s2ef/all/val_id/",
                metadata_path=base_dir / "oc20/s2ef/all/metadata.npz",
                first_n=DatasetFirstNConfig(first_n=200),
                lin_ref=base_dir / "oc20/lin_ref_coeffs.npz",
            ),
            energy_loss_scale=1.0,
            force_loss_scale=73.0,
            normalization={
                "y": M.NormalizationConfig(mean=0.0, std=24.901469505465872),
                "force": M.NormalizationConfig(mean=0.0, std=0.5111534595489502),
            },
        )
    ]
    
    
def fsdp_config_(config: M.PretrainConfig):
    ## Use fsdp
    config.fsdp = M.FSDPConfig(
        gradient_checkpointing=True,
    )
    
configs: list[tuple[M.PretrainConfig, type[M.PretrainModel]]] = []
config = M.PretrainConfig.draft()
base_config_(config)
tasks_config_(config) ##TODO: We load data here
backbone_config_(config)
# gradient_checkpointing_config_(config)
fsdp_config_(config)
# profiling_config_(config)
config = config.finalize()
configs.append((config, M.PretrainModel)) ## TODO:Match model type in M.PretrainModel


def run(config: M.PretrainConfig, model_cls: type[M.PretrainModel]):
    model = model_cls(config)
    trainer = ll.Trainer(config, **model.fsdp_trainer_kwargs())
    trainer.fit(model)
