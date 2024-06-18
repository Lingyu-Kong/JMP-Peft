# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import rich
import yaml

run_dirs = Path("/mnt/datasets/jmppeft/").glob("run_*/")
all_runs = [(run_dir / "metrics.csv", run_dir / "hparams.yaml") for run_dir in run_dirs]
print(f"Found {len(all_runs)} runs")
# %%
metrics_path, hparams_path = all_runs[0]
df = pd.read_csv(metrics_path)
df


@dataclass
class RunData:
    batches_per_sec: float
    samples_per_sec: float

    @classmethod
    def from_df(cls, df: pd.DataFrame, batch_size: int):
        batches_per_sec = df["train/batches_per_sec"].dropna()
        samples_per_sec = batches_per_sec * batch_size

        return cls(batches_per_sec.mean(), samples_per_sec.mean())


# %%


@dataclass
class RunInfo:
    model: str
    size: Literal["base", "large", "xl"]
    nodes: int
    gc: bool
    batch_size: int

    @property
    def gpus(self):
        return self.nodes * 8

    @classmethod
    def from_hparams(cls, hparams):
        model_name = hparams["backbone"]["name"]
        match model_name:
            case "graphormer3d":
                model_size = {
                    12: "base",
                    24: "large",
                    48: "xl",
                }[hparams["backbone"]["layers"]]
            case "gemnet":
                model_size = {
                    4: "base",
                    6: "large",
                    8: "xl",
                }[hparams["backbone"]["num_blocks"]]
            case _:
                raise ValueError(f"Unknown model name: {model_name}")

        for name_part in hparams["name_parts"]:
            if name_part.startswith("nodes_"):
                node_count = int(name_part.split("_")[1])
                break
        else:
            raise ValueError("Could not find node in name_parts")

        gc = hparams["gradient_checkpointing"]
        batch_size = hparams["batch_size"]
        return cls(model_name, model_size, node_count, gc, batch_size)


hparams = yaml.unsafe_load(hparams_path.read_text())
run = RunInfo.from_hparams(hparams)
run


# %%
@dataclass
class Run:
    info: RunInfo
    data: RunData

    @classmethod
    def from_paths(cls, metrics_path: Path, hparams_path: Path):
        if not metrics_path.exists() or not hparams_path.exists():
            return None

        hparams = yaml.unsafe_load(hparams_path.read_text())
        if not hparams:
            return None

        info = RunInfo.from_hparams(hparams)

        df = pd.read_csv(metrics_path)
        data = RunData.from_df(df, info.batch_size)
        return cls(info, data)


runs = [
    run
    for metrics_path, hparams_path in all_runs
    if (run := Run.from_paths(metrics_path, hparams_path)) is not None
]
rich.print(runs, len(runs))

# %%
# Create a DataFrame with the data
data = {
    "model": [run.info.model for run in runs],
    "size": [run.info.size for run in runs],
    "gpus": [run.info.gpus for run in runs],
    "gc": [run.info.gc for run in runs],
    "batch_size": [run.info.batch_size for run in runs],
    "batches_per_sec": [run.data.batches_per_sec for run in runs],
    "samples_per_sec": [run.data.samples_per_sec for run in runs],
}
df = pd.DataFrame(data)
df = df.sort_values(["size", "gpus", "model", "gc"])
df.to_csv("all_runs.csv", index=False)
df

# %%
# Plot the samples per second
import matplotlib.pyplot as plt
import seaborn as sns

df_plot = df.copy()
df_plot = pd.concat(
    [
        df_plot[~df_plot["gc"]],
        df_plot[(df_plot["gc"] & (df_plot["size"] == "xl"))],
    ]
)
df_plot = df_plot[df_plot["gpus"] == 8]

sns.set_theme()
plt.figure(figsize=(12, 8))
sns.barplot(data=df_plot, x="model", y="samples_per_sec", hue="size", ci="sd")
plt.ylabel("Samples per second")
plt.title("Samples per second by model and size (1 node)")
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is already defined

# Plot the samples per second for graphormer base as a function of # of GPUs
df_plot = df.copy()
df_plot = df_plot[df_plot["model"] == "graphormer3d"]
df_plot = df_plot[df_plot["size"] == "base"]
df_plot = df_plot[df_plot["gc"]]
df_plot = df_plot.sort_values("gpus")
# Manually add a new row for 1 GPU
df_plot = pd.concat(
    [
        # df_plot[df_plot["gpus"] == 4],
        pd.DataFrame(
            {
                "model": ["graphormer3d"],
                "size": ["base"],
                "gc": [True],
                "batch_size": [8],
                "gpus": [1],
                "samples_per_sec": [2.4150665040137174 * 8],
            }
        ),
        df_plot,
        # df_plot[df_plot["gpus"] != 8],
    ]
)
# Group by model, size, gc, batch_size and gpus, and take the mean of samples_per_sec
df_plot = (
    df_plot.groupby(["model", "size", "gc", "batch_size", "gpus"]).mean().reset_index()
)

# We draw this as a bar plot.
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=df_plot, x="gpus", y="samples_per_sec")
# Show the number of samples per second on top of the bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )

# Draw an arrow from each bar to the next one, showing the relative speedup
for i in range(len(df_plot) - 1):
    start = ax.patches[i]
    end = ax.patches[i + 1]
    # ^ Gives index out of bounds error, so we use the following instead

    ax.annotate(
        "",
        xy=(end.get_x() + end.get_width() / 2.0, end.get_height()),
        xytext=(start.get_x() + start.get_width() / 2.0, start.get_height()),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    # Write the speedup factor
    print(df_plot.iloc[i + 1]["samples_per_sec"], df_plot.iloc[i]["samples_per_sec"])
    factor = df_plot.iloc[i + 1]["samples_per_sec"] / df_plot.iloc[i]["samples_per_sec"]
    # Middle of the arrow
    x = (end.get_x() + start.get_x() + end.get_width()) / 2.0
    y = (end.get_height() + start.get_height()) / 2.0
    ax.annotate(
        f"{factor:.2f}",
        (x, y),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
        color="red",
    )


plt.ylabel("Samples per second")
plt.title("Samples per second for Graphormer base (gradient checkpointing, 1 node)")
plt.show()

# %%
import numpy as np

# Seaborn paper style
sns.set_theme(style="white")


df_plot = pd.DataFrame.from_records(
    [
        (0, "graphormer3d", "base", True, 8, 1, 19.32053203),
        (1, "graphormer3d", "base", True, 8, 8, 57.436137),
        (2, "graphormer3d", "base", True, 8, 64, 146.24811746),
        (3, "graphormer3d", "base", True, 8, 512, 674.34040015),
    ],
    columns=[
        "index",
        "model",
        "size",
        "gc",
        "batch_size",
        "gpus",
        "samples_per_sec",
    ],
)
df_plot2 = pd.DataFrame.from_records(
    [
        (0, "graphormer3d", "base", True, 8, 4, 40.2),
        (1, "graphormer3d", "base", True, 8, 8, 65.7),
        (2, "graphormer3d", "base", True, 8, 16, 113.7),
        (3, "graphormer3d", "base", True, 8, 32, 180.6),
        (4, "graphormer3d", "base", True, 8, 64, 314.16),
    ],
    columns=[
        "index",
        "model",
        "size",
        "gc",
        "batch_size",
        "gpus",
        "samples_per_sec",
    ],
)


# Scatter plot version of above
fig, ax = plt.subplots()

sns.lineplot(
    data=df_plot,
    x="gpus",
    y="samples_per_sec",
    marker="o",
    dashes=False,
    label="Frontier",
    ax=ax,
)
sns.lineplot(
    data=df_plot2,
    x="gpus",
    y="samples_per_sec",
    marker="o",
    dashes=False,
    label="Perlmutter",
    ax=ax,
)
# Also plot the "perfect scaling" line
ax.plot(
    df_plot["gpus"],
    df_plot["samples_per_sec"].iloc[0] * df_plot["gpus"],
    linestyle="--",
    color="black",
    label="Perfect scaling",
)

ax.set_ylabel("Samples per second")
ax.set_xlabel("Number of GPUs")
# ax.set_title("Samples per second for Graphormer base (gradient checkpointing)")

# ax.set_xscale("log")
# Focus on the y scale of our data
# ax.set_ylim(0, 799)

# Legend
ax.legend()

plt.show()

# %%
import numpy as np

dataset_size = 2_000_000

# Seaborn paper style
sns.set_theme(style="white")


df_plot = pd.DataFrame.from_records(
    [
        # (0, "graphormer3d", "base", True, 8, 1, 19.32053203),
        (1, "graphormer3d", "base", True, 8, 8, 57.436137),
        (2, "graphormer3d", "base", True, 8, 64, 146.24811746),
        (3, "graphormer3d", "base", True, 8, 512, 674.34040015),
    ],
    columns=[
        "index",
        "model",
        "size",
        "gc",
        "batch_size",
        "gpus",
        "samples_per_sec",
    ],
)
df_plot2 = pd.DataFrame.from_records(
    [
        # (0, "graphormer3d", "base", True, 8, 4, 40.2),
        (1, "graphormer3d", "base", True, 8, 8, 65.7),
        (2, "graphormer3d", "base", True, 8, 16, 113.7),
        (3, "graphormer3d", "base", True, 8, 32, 180.6),
        (4, "graphormer3d", "base", True, 8, 64, 314.16),
    ],
    columns=[
        "index",
        "model",
        "size",
        "gc",
        "batch_size",
        "gpus",
        "samples_per_sec",
    ],
)

df_plot["total_runtime"] = dataset_size / df_plot["samples_per_sec"]
df_plot2["total_runtime"] = dataset_size / df_plot2["samples_per_sec"]

df_plot["speedup"] = df_plot["total_runtime"].iloc[0] / df_plot["total_runtime"]
df_plot2["speedup"] = df_plot2["total_runtime"].iloc[0] / df_plot2["total_runtime"]

# df_plot["gpus"] = df_plot["gpus"] // 4
# df_plot2["gpus"] = df_plot2["gpus"] // 8
df_plot3 = pd.DataFrame.from_records(
    [
        {"nodes": 1, "speedup": 1},
        {"nodes": 2, "speedup": 2},
        {"nodes": 4, "speedup": 3.5},
        {"nodes": 8, "speedup": 7.0},
        {"nodes": 16, "speedup": 11.5},
        {"nodes": 32, "speedup": 17.9},
        {"nodes": 64, "speedup": 27.5},
    ]
)
df_plot3["gpus"] = df_plot3["nodes"] * 8

# Scatter plot version of above
fig, ax = plt.subplots()

sns.lineplot(
    data=df_plot,
    x="gpus",
    y="speedup",
    marker="o",
    dashes=False,
    label="Frontier",
    ax=ax,
)
sns.lineplot(
    data=df_plot2,
    x="gpus",
    y="speedup",
    marker="o",
    dashes=False,
    label="Perlmutter",
    ax=ax,
)
sns.lineplot(
    data=df_plot3,
    x="gpus",
    y="speedup",
    marker="o",
    dashes=False,
    label="Fast DimeNet++",
    ax=ax,
)
# Also plot the "perfect scaling" line
ax.plot(
    df_plot["gpus"],
    df_plot["speedup"].iloc[0] * df_plot["gpus"],
    linestyle="--",
    color="black",
    label="Perfect scaling",
)

ax.set_ylabel("Speedup")
ax.set_xlabel("Number of GPUs")
# ax.set_title("Samples per second for Graphormer base (gradient checkpointing)")

# ax.set_xscale("log")
# Focus on the y scale of our data
ax.set_ylim(0, 50)

# Legend
ax.legend()

ax.set_title("Speedup for 1 Epoch of 2M Dataset, Graphormer Base")

plt.show()
