mamba create -n jmp-peft python=3.11
conda activate jmp-peft

# Install PyTorch
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install PyTorch Geometric
mamba install -y -c pyg pyg pytorch-scatter pytorch-sparse pytorch-cluster

# Install other packages
mamba install -y \
    -c conda-forge \
    numpy matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    einops wandb \
    cloudpickle \
    "pydantic>2" \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# Install pymatgen + matbench-discovery
mamba install -y -c conda-forge pymatgen jarvis-tools
pip install matbench-discovery

# Install jaxtyping
pip install jaxtyping
