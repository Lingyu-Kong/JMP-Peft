# %%
from pathlib import Path

base_paths = [
    # Path(
    #     "/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft-realruns-final"
    # ),
    Path("/lustre/orion/mat265/world-shared/nimashoghi/repositories/jmp-peft/"),
]
base_paths

# %%
paths = [
    f
    for base_path in base_paths
    for f in base_path.glob("lltrainer/*/log/wandb/wandb/wandb/latest-run/")
]
paths

# %%
project = "jmp-proposal"
commands = []
for path in paths:
    command = f"wandb sync {path} --project {project} --include-synced"
    # print(command)
    commands.append(command)

print("; ".join(commands))
