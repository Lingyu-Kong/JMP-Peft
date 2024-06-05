mamba create -n py311 python=3.11
conda activate py311

# Frontier uses ROCm, so we need to install PyTorch using their pre-built wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Install PyG dependencies
pip install torch_geometric

#########################################################################
# Build pytorc_sparse and pytorch_scatter manually.
# This is due to this issue: https://github.com/ROCm/ROCm/issues/2783
# #######################################################################
pushd ../

git clone --recursive https://github.com/rusty1s/pytorch_scatter
pushd pytorch_scatter
MAX_JOBS=8 python ./setup.py install
popd

git clone --recursive https://github.com/rusty1s/pytorch_sparse
pushd pytorch_sparse
MAX_JOBS=8 python ./setup.py install
popd

popd

# Install base dependencies.
# We don't use conda anymore, so we need to install these packages using pip.
pip install frozendict wandb lovely-tensors matplotlib ipykernel numba pytorch-lightning cloudpickle einops pytest wrapt numpy nbformat nbval torchmetrics typing-extensions varname pandas pyyaml plotly tqdm requests ipywidgets scikit-learn seaborn lovely-numpy networkx sympy lightning pymatgen ase jarvis-tools pydantic

# Installed pip dependencies
pip install rich lmdb beartype jaxtyping ruff

# Install ll (needs to be done manually)
echo "Please install the 'll' package manually"
