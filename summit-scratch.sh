mamba create -n jmp-peft-scratch python=3.11
conda activate jmp-peft-scratch

# Install PyTorch
# mamba install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
# ^ Summit is on powerpc64le, so we need to install using their pre-built wheels
pip install /sw/summit/pytorch/wheel_dist/torch-2.3.0a0+giteba28a6-cp311-cp311-linux_ppc64le.whl

# Install base dependencies
mamba install -y -c conda-forge \
    numpy scipy matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    rich cloudpickle frozendict wrapt varname typing-extensions lovely-numpy requests pytest nbval \
    cmake ninja

# Install PyTorch Geometric
# mamba install -y -c pyg pyg pytorch-scatter pytorch-sparse pytorch-cluster
pip install torch_geometric

# Pydantic needs to be installed from GitHub (for now)
pip install git+https://github.com/pydantic/pydantic.git

# Install other PyTorch packages from pip
pip install pytorch-lightning torchmetrics lightning einops wandb lovely-tensors
pip install beartype jaxtyping

# Install packages for datasets
pip install lmdb ase
