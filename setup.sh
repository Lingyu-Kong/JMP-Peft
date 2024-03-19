mamba create -n jmp-peft python=3.11
conda activate jmp-peft

# Install PyTorch
mamba install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

# Install PyTorch Geometric
mamba install -y -c pyg pyg pytorch-scatter pytorch-sparse pytorch-cluster

# Pydantic needs to be installed from GitHub (for now)
pip install git+https://github.com/pydantic/pydantic.git

# Install other packages
mamba install -y \
    -c conda-forge \
    numpy matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    einops wandb \
    cloudpickle \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# Rich for better terminal output
pip install rich

# Install packages for datasets
pip install lmdb
mamba install -y -c conda-forge ase

# Install pymatgen + matbench-discovery
mamba install -y -c conda-forge pymatgen jarvis-tools
pip install matbench-discovery

# Install jaxtyping
pip install beartype jaxtyping

# Install LoRA
pip install loralib

# Install torchmdnet
mamba install -y -c conda-forge torchmd-net
