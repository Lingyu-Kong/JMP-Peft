mamba create -n rocm53 python=3.10
conda activate rocm53

# Frontier uses ROCm, so we need to install PyTorch using their pre-built wheels
pip install "torch==2.0.1" --index-url https://download.pytorch.org/whl/rocm5.3

# Install PyG dependencies
pip install torch_geometric

#########################################################################
# Build pytorc_sparse and pytorch_scatter manually.
# This is due to this issue: https://github.com/ROCm/ROCm/issues/2783
# #######################################################################
pushd ../

git clone --recursive https://github.com/rusty1s/pytorch_scatter.git
pushd pytorch_scatter
MAX_JOBS=16 python ./setup.py install
popd

git clone --recursive https://github.com/rusty1s/pytorch_sparse.git
pushd pytorch_sparse
MAX_JOBS=16 python ./setup.py install
popd

popd

# Install base dependencies.
# We don't use conda anymore, so we need to install these packages using pip.
pip install frozendict wandb lovely-tensors matplotlib ipykernel numba pytorch-lightning cloudpickle einops pytest wrapt numpy nbformat nbval torchmetrics typing-extensions varname pandas pyyaml plotly tqdm requests ipywidgets scikit-learn seaborn lovely-numpy networkx sympy lightning pymatgen ase jarvis-tools pydantic

# Installed pip dependencies
pip install rich lmdb beartype jaxtyping ruff

# Install ll (needs to be done manually)
echo "Please install the 'll' package manually"


# salloc command:
salloc --nodes 1 --account mat265 --time 01:00:00 --qos debug

#!/bin/bash

conda deactivate || true
conda activate rocm53

module reset
module load craype-accel-amd-gfx90a
module load PrgEnv-gnu
module load rocm
module load amd-mixed

echo "Loaded ROCm 5.3.0 environment"
