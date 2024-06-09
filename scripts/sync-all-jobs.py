# %%
from pathlib import Path

base_path = Path(
    "/lustre/orion/mat265/world-shared/nimashoghi/projectdata/jmppeft-realruns-final"
)
base_path

# %%
paths = list(base_path.glob("lltrainer/*/log/wandb/wandb/wandb/latest-run/"))
paths

# %%
project = "jmp-proposal"
for path in paths:
    command = f"wandb sync {path} --project {project} --include-synced"
    print(command)
