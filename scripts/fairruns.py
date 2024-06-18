# %%
from pathlib import Path

base_path = Path(
    "/mnt/datasets/jmppeft/jmp_logs_for_nima/logs_for_nima/ablation_dev_runs"
)

runs = list(base_path.rglob("lightning_logs/*/csv/**/metrics.csv"))
runs

import pandas as pd
from IPython.display import display

metrics = [
    "step",
    "epoch",
    "val/oc20/energy_mae",
    "val/oc20/forces_mae",
    "val/oc22/energy_mae",
    "val/oc22/forces_mae",
    "val/ani1x/energy_mae",
    "val/ani1x/forces_mae",
    "val/transition1x/energy_mae",
    "val/transition1x/forces_mae",
]


runs_of_interest = [
    "temp2.0-lsreduce-hinge-edgedropout0.1",
    "large-temp2.0-lsreduce-hinge-wd0.1",
]


def handle_run(run_path: Path):
    name = run_path.parent.parent.name
    if name not in runs_of_interest:
        return

    df = pd.read_csv(run_path)

    if not all(metric in df.columns for metric in metrics):
        return

    df = df[metrics].dropna()
    # print(name, str(run_path.parent))
    # display(df)

    return name, df.iloc[-1]


rundata_ablations = []

for run in runs:
    res = handle_run(run)
    if not res:
        continue
    rundata_ablations.append(res)

# %%

base_path = Path(
    "/mnt/datasets/jmppeft/jmp_logs_for_nima/logs_for_nima/pretrain_large_runs"
)

runs = list(base_path.rglob("lightning_logs/*/csv/**/metrics.csv"))
runs


metrics = [
    "step",
    "epoch",
    "val/oc20/energy_mae",
    "val/oc20/forces_mae",
    "val/oc22/energy_mae",
    "val/oc22/forces_mae",
    "val/ani1x/energy_mae",
    "val/ani1x/forces_mae",
    "val/transition1x/energy_mae",
    "val/transition1x/forces_mae",
]


runs_of_interest = [
    "hinge-base-all-physical-wd0.1-ema0.99-coslr",
    "large-wd0.1-ema0.99-coslr-resume_8_16",
]


def handle_run(run_path: Path):
    name = run_path.parent.parent.name
    if runs_of_interest and name not in runs_of_interest:
        return

    df = pd.read_csv(run_path)

    if not all(metric in df.columns for metric in metrics):
        return

    df = df[metrics].dropna()
    # print(name, str(run_path.parent))
    # display(df)

    return name, df.iloc[-1]


rundata_pretrain = []

for run in runs:
    res = handle_run(run)
    if not res:
        continue
    rundata_pretrain.append(res)
    name, df = res
    print(name)
    display(df)

# %%
records_list = []

name_mapping = {
    "temp2.0-lsreduce-hinge-edgedropout0.1": "JMP-S (2M Data)",
    "large-temp2.0-lsreduce-hinge-wd0.1": "JMP-L (2M Data)",
    "hinge-base-all-physical-wd0.1-ema0.99-coslr": "JMP-S (120M Data)",
    "large-wd0.1-ema0.99-coslr-resume_8_16": "JMP-L (120M Data)",
}

for name, df in [*rundata_ablations, *reversed(rundata_pretrain)]:
    records_list.append(
        {
            "name": name_mapping[name],
            "OC20": df["val/oc20/forces_mae"],
            "OC22": df["val/oc22/forces_mae"],
            "ANI1x": df["val/ani1x/forces_mae"],
            "Transition1X": df["val/transition1x/forces_mae"],
        }
    )

df = pd.DataFrame.from_records(records_list)
df

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

df_plot = df.melt("name", var_name="Dataset", value_name="MAE")

sns.barplot(data=df_plot, x="name", y="MAE", hue="Dataset", ax=ax)
plt.xticks(rotation=45)
plt.ylabel("Forces MAE")
plt.title("Forces MAE on different datasets")
plt.show()
