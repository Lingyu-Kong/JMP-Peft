# %%
from pathlib import Path

base_paths = [
    "/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft-realruns-6_9",
    "/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft-realruns-final",
]
base_paths = [Path(base_path) for base_path in base_paths]

all_runs: list[tuple[Path, Path]] = []
for base_path in base_paths:
    assert base_path.exists() and base_path.is_dir()

    for metrics_path in base_path.glob("lltrainer/*/log/csv/**/metrics.csv"):
        hparams_path = metrics_path.parent / "hparams.yaml"
        if not hparams_path.exists():
            continue

        all_runs.append((metrics_path, hparams_path))

print(len(all_runs))

# %%
# Write a command to zip all the files
import shutil
import tempfile

from tqdm.auto import tqdm

with tempfile.TemporaryDirectory() as tmpdirname:
    for i, (metrics_path, hparams_path) in enumerate(tqdm(all_runs)):
        base_dir = Path(tmpdirname) / f"run_{i}"
        base_dir.mkdir()

        shutil.copy(metrics_path, base_dir / "metrics.csv")
        shutil.copy(hparams_path, base_dir / "hparams.yaml")

    path = shutil.make_archive("all_runs", "zip", tmpdirname)

    # Move the zip to cwd/all_runs.zip
    shutil.move(path, Path.cwd() / "all_runs.zip")

# %%
import pandas as pd
import yaml

metrics_path, hparams_path = all_runs[0]
df = pd.read_csv(metrics_path)
df

# %%
hparams = yaml.safe_load(hparams_path.read_text())
