mamba create -n jmp python=3.11
conda activate jmp

# Frontier uses ROCm, so we need to install PyTorch using their pre-built wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Install base dependencies.
# We don't use conda anymore, so we need to install these packages using pip.
pip install frozendict wandb lovely-tensors matplotlib ipykernel numba pytorch-lightning cloudpickle einops pytest wrapt numpy nbformat nbval torchmetrics typing-extensions varname pandas pyyaml plotly tqdm requests ipywidgets scikit-learn seaborn lovely-numpy networkx sympy lightning pymatgen ase jarvis-tools pydantic

# Installed pip dependencies
pip install rich lmdb beartype jaxtyping ruff

# Install ll (needs to be done manually)
echo "Please install the 'll' package manually"

# Install PyG
mkdir -p /tmp/jmp-pyg-build/
pushd /tmp/jmp-pyg-build/
wget https://github.com/Looong01/pyg-rocm-build/releases/download/5/torch-2.2-rocm-5.7-py310-linux_x86_64.zip
unzip torch-2.2-rocm-5.7-py310-linux_x86_64.zip
pip install torch-2.2-rocm-5.7-py310-linux_x86_64/*
popd
rm -rf /tmp/jmp-pyg-build/
apptainer instance start --rocm --writable-tmpfs ./my_container.sif my_container
