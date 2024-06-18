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
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Reshape dataframe for seaborn
df_plot = df.melt("name", var_name="Dataset", value_name="MAE")

# Compute average force MAE per model
df_avg = df.set_index("name").mean(axis=1).reset_index()
df_avg.columns = ["name", "Average MAE"]

# Plot average force MAE per model translucently
sns.barplot(data=df_avg, x="name", y="Average MAE", color="gray", alpha=0.3, ax=ax)

# Plot per-dataset MAEs with thinner bars
sns.barplot(
    data=df_plot,
    x="name",
    y="MAE",
    hue="Dataset",
    dodge=True,
    ax=ax,
    edgecolor=".2",
    linewidth=0.85,
    ci=None,
    # Make them translucent
    alpha=0.5,
)

plt.xticks(rotation=45)
plt.ylabel("Forces MAE (eV/Angstrom)")
# No x-axis label
plt.xlabel("")
plt.title("The effect of model and dataset size on JMP forces MAE")
plt.show()

# %%
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

df_plot = df.melt("name", var_name="Dataset", value_name="MAE")

# Compute the average force MAE per model
df_avg = df.set_index("name").mean(axis=1).reset_index()
df_avg.columns = ["name", "Average MAE"]

# Plot the average force MAE per model
sns.barplot(data=df_avg, x="name", y="Average MAE", color="gray", alpha=0.85, ax=ax)

# Plot the per-dataset MAEs with thinner bars
sns.barplot(
    data=df_plot,
    x="name",
    y="MAE",
    hue="Dataset",
    ax=ax,
    dodge=True,
    edgecolor="black",
    linewidth=1,
    ci=None,
    alpha=0.85,
)

plt.xticks(rotation=45)
plt.ylabel("Forces MAE")
plt.title("Forces MAE on different datasets")

plt.xlabel("")
# Adjust the legend and show the plot
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[1:], labels=labels[1:])
plt.show()

# %%

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

df_plot = df_plot.copy()
# Reorder
df_plot = df.iloc[[0, 2, 1, 3]]

df_plot = df_plot.melt("name", var_name="Dataset", value_name="MAE")

# Compute the average force MAE per model
df_avg = df.set_index("name").mean(axis=1).reset_index()
df_avg.columns = ["name", "Average MAE"]

# Plot the average force MAE per model
avg_barplot = sns.barplot(
    data=df_avg, x="name", y="Average MAE", color="gray", alpha=0.85, ax=ax
)

offsets = [
    (-0.2, 0.0025),
    (-0.2, 0.00925),
    (-0.2, 0.00485),
    (-0.2, 0.007),
]

# Add text annotations for the average MAE on the bars
for p, (offset_x, offset_y) in zip(avg_barplot.patches, offsets):
    avg_mae = p.get_height()
    avg_barplot.annotate(
        f"Average:\n{avg_mae:.3f}",
        ((p.get_x() + p.get_width() / 2.0) + offset_x, avg_mae + offset_y),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=10,
        color="black",
        # weight="bold",
    )

# Plot the per-dataset MAEs with thinner bars
sns.barplot(
    data=df_plot,
    x="name",
    y="MAE",
    hue="Dataset",
    ax=ax,
    dodge=True,
    edgecolor="black",
    linewidth=1,
    ci=None,
    alpha=0.85,
)

plt.xticks(rotation=45)
plt.ylabel("Force MAE (eV/Å)")
plt.title("Force MAE Metrics on JMP Pre-training Datasets")

plt.xlabel("")
# Adjust the legend and show the plot
plt.show()
