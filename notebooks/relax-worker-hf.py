# %%
from pathlib import Path

from jmppeft._relaxer_worker import Config, HuggingfaceCkpt, run

dest_dir = Path("/mnt/datasets/jmp-mptrj-checkpoints/relaxer-results-1k-hf/")


configs: list[tuple[Config]] = []
config = Config.draft()
config.num_items = 1024
config.fmax = 0.05
# config.ckpt = HuggingfaceCkpt(
#     repo_id="nimashoghi/jmp-mptrj-mptrj-total-lr8e-05-ln-maceenergy-maceforce-withrel-ec1-0-fc10-0-sc100-0-6lijquot",
#     filename="checkpoints/last/epoch32-step136620.ckpt",
# )
config.ckpt = HuggingfaceCkpt(
    repo_id="nimashoghi/jmp-mptrj-linref-lr8e-05-wd0-1-ln-emae-fl2mae-smae-ec1-0-fc10-0-sc100-0-posaug-std0-01-onjos0cn",
    filename="checkpoints/last.ckpt",
)
config.dest = dest_dir / config.ckpt.filename.rsplit("/", 1)[1].replace(
    ".ckpt", ".dill"
)
config.device_id = 1
config.energy_key = "s2e_energy"
config.linref = True
config.dest = config.dest.with_stem(f"{config.dest.stem}_{config.energy_key}")
config = config.finalize()
configs.append((config,))


# %%
import nshrunner as nr

runner = nr.Runner(run)
runner.session(configs, snapshot=False)
