# %%
from pathlib import Path

from nshconfig_extra import HFPath

from jmppeft._relaxer_worker import Config, run

dest_dir = Path("/mnt/datasets/jmp-mptrj-checkpoints/relaxer-results-1k-hf/")


configs: list[tuple[Config]] = []
config = Config.draft()
config.num_items = 1024
config.fmax = 0.05
# config.ckpt = HFPath(
#     repo="nimashoghi/jmp-mptrj-mptrj-total-lr8e-05-ln-maceenergy-maceforce-withrel-ec1-0-fc10-0-sc100-0-6lijquot",
#     path="checkpoints/last/epoch32-step136620.ckpt",
# )
config.ckpt = HFPath(
    repo="nimashoghi/jmp-mptrj-linref-lr8e-05-wd0-1-ln-emae-fl2mae-smae-ec1-0-fc10-0-sc100-0-posaug-std0-01-8x2q7vme",
    path="checkpoints/last/epoch12-step47086.ckpt",
)
id_ = config.ckpt.repo.rsplit("-", 1)[1]
config.dest = (
    dest_dir / id_ / config.ckpt.path.rsplit("/", 1)[1].replace(".ckpt", ".dill")
)
config.device_id = 1
config.energy_key = "s2e_energy"
config.linref = True
config.ignore_if_exists = False
config.dest = config.dest.with_stem(f"{config.dest.stem}_{config.energy_key}")
config.dest.parent.mkdir(parents=True, exist_ok=True)
config = config.finalize()
configs.append((config,))


# %%
import nshrunner as nr

runner = nr.Runner(run)
runner.session(configs, snapshot=False)
