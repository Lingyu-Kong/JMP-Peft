# %%
from pathlib import Path

from jmppeft._relaxer_worker import Config, run


def _latest_ckpts(dir_: Path) -> dict[str, Path]:
    best_ckpts: dict[str, Path] = {}
    for run_ in dir_.iterdir():
        if not run_.is_dir():
            continue

        # Find all ckpt files and take the latest one
        ckpts = [
            ckpt
            for ckpt in run_.glob("checkpoint/*.ckpt")
            if "exception" not in ckpt.name
        ]
        if not ckpts:
            continue

        # Resolve symlinks and ignore duplicates
        ckpts_resolved = []
        for ckpt in ckpts:
            # Account for symlink loops
            try:
                ckpt_resolved = ckpt.resolve()
            except BaseException:
                continue
            ckpts_resolved.append(ckpt_resolved)
        ckpts = list(set(ckpts_resolved))

        # Find the latest checkpoint
        best_ckpt = max(ckpts, key=lambda x: x.stat().st_mtime)
        best_ckpts[run_.name] = best_ckpt

    return best_ckpts


def _find_ckpts(ckptdir: Path, destdir: Path):
    resolved_dirs = _latest_ckpts(ckptdir)
    if not resolved_dirs:
        raise ValueError(f"No checkpoints found in {ckptdir}")

    dests = [destdir / f"{rundir}.dill" for rundir in resolved_dirs.keys()]

    return [
        (name, ckpt, dest) for (name, ckpt), dest in zip(resolved_dirs.items(), dests)
    ]


def _make_configs(config_draft: Config, ckptdir: Path, destdir: Path):
    assert config_draft._is_draft_config, "Config must be a draft config"

    configs: list[Config] = []
    for _, ckpt, dest in _find_ckpts(ckptdir, destdir):
        config = config_draft.model_copy(deep=True)
        config.ckpt = ckpt
        config.dest = dest
        configs.append(config)

    return configs


# %%
import nshutils as nu


def _create_draft_config():
    draft_config = Config.draft()
    draft_config.num_items = 1024
    draft_config.fmax = 0.01
    draft_config.stress_weight = 1.0

    return draft_config


all_configs: list[Config] = []

ckpt_dir = Path("/mnt/datasets/jmp-mptrj-checkpoints/sm_checkpoints/")
dest_dir = Path("/mnt/datasets/jmp-mptrj-checkpoints/relaxer-results-1k/")

for energy_key in ("s2e_energy", "s2re_energy"):
    draft_config = _create_draft_config()
    draft_config.energy_key = energy_key
    for config in _make_configs(draft_config, ckpt_dir, dest_dir):
        config.dest = config.dest.with_stem(f"{config.dest.stem}_{config.energy_key}")
        config = config.finalize()

        if energy_key != "s2e_energy":
            continue
        if config.ckpt.parts[-3] != "mptrj-jmps-s2ef_s2re":
            continue
        all_configs.append(config)

        config = config.model_copy()
        config.fmax = 0.05
        all_configs.append(config)


nu.display(all_configs)


# %%
import more_itertools
import nshrunner as nr

num_workers = 2
worker_configs_list = [
    [(config.model_copy(update={"device_id": worker_idx}),) for config in config_list]
    for worker_idx, config_list in enumerate(
        more_itertools.distribute(num_workers, all_configs)
    )
]

for worker_configs in worker_configs_list:
    if not worker_configs:
        continue
    runner = nr.Runner(run)
    runner.session(worker_configs, snapshot=False)
