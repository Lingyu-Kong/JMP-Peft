mamba create -n jmp-peft-py310 python=3.10
conda activate jmp-peft-py310

# Install PyTorch
# mamba install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
# ^ Summit is on powerpc64le, so we need to install using their pre-built wheels
pip install /sw/summit/pytorch/wheel_dist/torch-2.3.0a0+giteba28a6-cp310-cp310-linux_ppc64le.whl

# Install base dependencies
mamba install -y -c conda-forge \
    numpy scipy matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    rich cloudpickle frozendict wrapt varname typing-extensions lovely-numpy requests pytest nbval

# Install PyTorch Geometric
pip install torch_geometric torch-scatter torch-sparse torch-cluster

# Pydantic needs to be installed from GitHub (for now)
pip install git+https://github.com/pydantic/pydantic.git

# Install other PyTorch packages from pip
pip install pytorch-lightning torchmetrics lightning einops wandb lovely-tensors
pip install beartype jaxtyping

# Install packages for datasets
pip install ase lmdb
