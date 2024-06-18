# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv(
    "/workspaces/repositories/jmp-peft/scripts/wandb_export_2024-06-14T12_20_48.230-04_00.csv"
)

# Process the data
df["nodes"] = df["Name"].apply(lambda x: int(x.rsplit("_", 1)[-1]))
df["gpus"] = df["nodes"] * 4

df["Per-GPU FLOPs for 1 Batch"] = (
    df["train/device/flops_per_sec"] * df["train/device/batches_per_sec"]
)
df["Total FLOPs for 1 Batch"] = df["Per-GPU FLOPs for 1 Batch"] * df["gpus"]

# Prepare data for the regression model
X = df["gpus"].values.reshape(-1, 1)
y = df["Total FLOPs for 1 Batch"].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict FLOPs for 512 GPUs
gpus_to_predict = np.array([[512]])
predicted_flops = model.predict(gpus_to_predict)

# Display the result
print(f"Predicted Total FLOPs for 512 GPUs: {predicted_flops[0]:.3e}")

# Show the dataframe with the new columns
df

# %%
