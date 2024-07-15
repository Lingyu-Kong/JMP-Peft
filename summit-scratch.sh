export HF_DATASETS_CACHE="/gpfs/alpine2/proj-shared/mat273/nimashoghi/hf-datasets-cache" # in ~/.bashrc

mamba create -n jmp-310 python=3.10
conda activate jmp-310

# Install PyTorch
# mamba install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
# ^ Summit is on powerpc64le, so we need to install using their pre-built wheels
pip install /sw/summit/pytorch/wheel_dist/torch-*-cp310-cp310*

# Install base dependencies
mamba install -y -c conda-forge \
    "numpy<2" scipy matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pydantic rich cloudpickle frozendict wrapt varname typing-extensions "lovely-numpy==0.2.12" requests pytest nbval pyarrow \
    cmake ninja

# Install PyTorch Geometric
pip install torch_geometric

# Build pytorc_sparse and pytorch_scatter manually.
git clone --recursive https://github.com/rusty1s/pytorch_scatter.git
pushd pytorch_scatter
MAX_JOBS=16 python ./setup.py install
popd

git clone --recursive https://github.com/rusty1s/pytorch_sparse.git
pushd pytorch_sparse
MAX_JOBS=16 python ./setup.py install
popd

# Install other PyTorch packages from pip
pip install pytorch-lightning torchmetrics lightning einops wandb lovely-tensors
pip install beartype jaxtyping

# Install packages for datasets
pip install lmdb ase datasets

# Install other packages
pip install tabulate pysnooper
