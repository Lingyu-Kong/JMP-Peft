# %%
import ll
import rich

ll.pretty()

# %%
from pathlib import Path

base_path = Path(
    "/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft-realruns-final"
)
base_path

# %%
paths = list(base_path.glob("lltrainer/*/log/csv/csv/*/*/"))
paths


# %%
import pandas as pd
import yaml


def process_path(path: Path):
    hparams_path = path / "hparams.yaml"
    csv_path = path / "metrics.csv"
    if not hparams_path.exists():
        return None

    with hparams_path.open() as f:
        hparams = yaml.unsafe_load(f)

    if not hparams:
        print("No hparams found for", hparams_path)
        return None

    if not csv_path.exists():
        print("No csv file found for", hparams["name_parts"])
        return None

    df = pd.read_csv(csv_path)
    return df, hparams


def fn(df: pd.DataFrame, hparams: dict):
    name = "-".join(hparams["name_parts"])
    print(name)

    print(
        df["computed_epoch"].dropna().iloc[-1],
        df["train/loss"].dropna().iloc[-1],
    )


for path in paths:
    if (out := process_path(path)) is None:
        continue

    df, hparams = out
    fn(df, hparams)
    # break

    # ckpt_dir = hparams["environment"]["checkpoint_dir"]
    # rich.print((hparams["name_parts"], ckpt_dir))
    # rich.print(list(Path(ckpt_dir).rglob("**/*.ckpt")))

# %%
