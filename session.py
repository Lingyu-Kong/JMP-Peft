# %%
def monkey_patch_torch_scatter():
    import torch
    import torch_scatter

    def segment_coo(
        src: torch.Tensor,
        index: torch.Tensor,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
        reduce: str = "sum",
    ):
        # Dim should be the first (and only) non-broadcastable dimension in index.
        dims_to_squeeze: list[int] = []
        dim: int = -1
        for dim_idx in range(index.dim()):
            if index.size(dim_idx) == 1:
                dims_to_squeeze.append(dim)
                continue

            if dim != -1:
                raise ValueError(
                    "Found multiple non-broadcastable dimensions in index."
                )
            dim = dim_idx

        index = index.squeeze(dims_to_squeeze)
        return torch_scatter.scatter(src, index, dim, out, dim_size, reduce)

    torch_scatter.segment_coo = segment_coo

    print("Monkey-patched torch_scatter.segment_coo")


# %%
from pathlib import Path

import ll
from jmppeft.configs.finetune.jmp_l import jmp_l_ft_config_
from jmppeft.configs.finetune.matbench_discovery import jmp_matbench_discovery_config_
from jmppeft.tasks.config import AdamWConfig
from jmppeft.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmppeft.tasks.finetune.matbench_discovery import (
    MatbenchDiscoveryConfig,
    MatbenchDiscoveryModel,
)
from jmppeft.utils.param_specific_util import make_parameter_specific_optimizer_config

project_root = Path("/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft/")

ckpt_path = Path("/lustre/orion/mat265/world-shared/nimashoghi/checkpoints/jmp-l.pt")
dataset_base_path = Path(
    "/lustre/orion/mat265/world-shared/nimashoghi/datasets/mptraj/"
)


def create_config():
    config = MatbenchDiscoveryConfig.draft()
    config.project = "jmp_peft_nersc"
    config.name = "matbench_discovery-nograd"
    jmp_l_ft_config_(config)
    jmp_matbench_discovery_config_(
        config,
        dataset_base_path,
        use_megnet_133k=True,
        use_linref=True,
    )
    config.energy_forces_config_(
        gradient=False,
        energy_coefficient=0.05,
        force_coefficient=1.0,
    )

    config.optimizer = AdamWConfig(
        lr=5.0e-6,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    config.tags.append("direct_forces")
    config.name += "_direct_forces"
    config.trainer.precision = "16-mixed-auto"

    # Set data config
    config.batch_size = 10
    config.num_workers = 2
    # Balanced batch sampler
    config.use_balanced_batch_sampler = True
    config.trainer.use_distributed_sampler = False

    config.backbone.regress_forces = True
    config.backbone.direct_forces = True
    config.backbone.regress_energy = True

    # config.meta["ft_ckpt_path"] = ckpt_path

    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
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

    config.with_project_root_(project_root)

    if (wandb_config := config.trainer.logging.wandb) is not None:
        wandb_config.disable_()

    config.batch_size = 1
    config.eval_batch_size = 1
    config.trainer.fast_dev_run = 32
    config.num_workers = 0

    return config.finalize(), MatbenchDiscoveryModel


configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
config, model_cls = create_config()
configs.append((config, model_cls))
# %%
import rich

ll.pretty()

config, model_cls = configs[0]
model = model_cls(config)
dataset = model.train_dataset()
batch = model.collate_fn([dataset[0], dataset[1]])
print(batch)

# %%
import copy

rich.print(model(copy.deepcopy(batch)))

# %%
model = model.cuda()
batch = batch.to(model.device)
batch

rich.print(model(batch))
