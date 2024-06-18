# %%
import pandas as pd

df = pd.read_csv(
    "/workspaces/repositories/jmp-peft/scripts/wandb_export_2024-06-14T12_20_48.230-04_00.csv"
)
df

# %%
df["nodes"] = df["Name"].apply(lambda x: int(x.rsplit("_", 1)[-1]))
df["gpus"] = df["nodes"] * 4
df

# %%
df["Per-GPU FLOPs for 1 Batch"] = (
    df["train/device/flops_per_sec"] * df["train/device/batches_per_sec"]
)
df["Total FLOPs for 1 Batch"] = df["Per-GPU FLOPs for 1 Batch"] * df["gpus"]
df

# %%
df_new = df[["Name", "gpus", "Total FLOPs for 1 Batch", "Per-GPU FLOPs for 1 Batch"]]
df_new.to_csv("flops.csv", index=False)
df_new
