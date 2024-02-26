# %%
from pathlib import Path

from jmppeft.configs.finetune.rmd17 import jmp_l_rmd17_config
from jmppeft.modules.lora import LoraConfig
from jmppeft.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase

ckpt_path = Path("/mnt/shared/checkpoints/fm_gnoc_large_2_epoch.ckpt")
base_path = Path("/mnt/shared/datasets/rmd17/")

config, model_cls = jmp_l_rmd17_config("aspirin", base_path, ckpt_path)
print(config)

config.trainer.python_logging.log_level = "INFO"
config.lora = LoraConfig(r=4)
config.num_workers = 0

configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
configs.append((config, model_cls))

# %%
from jmppeft.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
)
from ll import Runner, Trainer


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
LoraConfig.pprint_path_tree()

# %%
import rich

rich.print(LoraConfig._all_paths)
