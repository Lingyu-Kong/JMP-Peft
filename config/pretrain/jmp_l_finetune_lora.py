# %%
from pathlib import Path

from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.rmd17 import jmp_l_rmd17_config_
from jmppeft.modules.lora import LoraConfig
from jmppeft.tasks.finetune.base import (
    FinetuneConfigBase,
    FinetuneModelBase,
    RLPConfig,
    RLPWarmupConfig,
)
from jmppeft.tasks.finetune.rmd17 import RMD17Config, RMD17Model

ckpt_path = Path("/mnt/shared/checkpoints/fm_gnoc_large_2_epoch.ckpt")
base_path = Path("/mnt/shared/datasets/rmd17/")

config = RMD17Config.draft()
jmp_l_ft_config_(config, ckpt_path, ema_backbone=True)
jmp_l_rmd17_config_(config, "aspirin", base_path)

config.parameter_specific_optimizers = None
config.optimizer.lr = 1.0e-4
config.lr_scheduler = RLPConfig(
    patience=25,
    factor=0.8,
    interval="epoch",
    warmup=RLPWarmupConfig(
        step_type="epoch",
        steps=5,
        start_lr_factor=1.0e-1,
    ),
)

config.lora = LoraConfig(r=4)
config.num_workers = 8

configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
configs.append((config.finalize(), RMD17Model))

# %%
from ll import Runner, Trainer

from jmppeft.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
)


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    if (ckpt_path := config.meta.get("ckpt_path")) is None:
        raise ValueError("No checkpoint path provided")

    model = model_cls(config)

    # Load the checkpoint
    state_dict = retreive_state_dict_for_finetuning(
        ckpt_path, load_emas=config.meta.get("ema_backbone", False)
    )
    embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
    backbone = filter_state_dict(state_dict, "backbone.")
    model.load_backbone_state_dict(backbone=backbone, embedding=embedding, strict=True)

    trainer = Trainer(config)
    trainer.fit(model)


runner = Runner(run)
runner.fast_dev_run(configs)

# %%
runner = Runner(run)
runner.local_session_per_gpu(configs, snapshot=True)
