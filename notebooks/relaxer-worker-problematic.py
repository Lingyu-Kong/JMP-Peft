# %%
from pathlib import Path

import nshutils as nu
from jmppeft._relaxer_worker import Config, run

configs: list[tuple[Config]] = []

ckpt_path = Path(
    "/mnt/datasets/jmp-mptrj-checkpoints/sm_checkpoints/mptrj-jmps-s2ef_s2re/checkpoint/epoch=125-step=356580-val_matbench_discovery_force_mae=0.01882246695458889.ckpt"
)
dest = Path(
    "/mnt/datasets/jmp-mptrj-checkpoints/relaxer-results-8k_problematic/base.dill"
)
dest.parent.mkdir(parents=True, exist_ok=True)
traj_dir = dest.parent / "trajs"
traj_dir.mkdir(exist_ok=True)


config = Config.draft()
config.num_items = 1024 * 8
config.fmax = 0.05
config.energy_key = "s2e_energy"
config.linref = True
config.idx_subset = Path(
    "/workspaces/repositories/jmp-peft/notebooks/top_5pct_indices.npy"
)
config.save_traj = traj_dir
config.ckpt = ckpt_path
config.dest = dest
config.device_id = 1
config = config.finalize()
configs.append((config,))

nu.display(configs)


# %%
import nshrunner as nr

runner = nr.Runner(run)
runner.session(configs, snapshot=False)

# %%
