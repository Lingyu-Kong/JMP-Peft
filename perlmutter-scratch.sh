export HF_DATASETS_CACHE="/global/cfs/cdirs/m3641/Nima/hf-datasets-cache" # in ~/.bashrc

mamba create -n jmp-310-pytorch230 python=3.10
conda activate jmp-310-pytorch230

# Install PyTorch
mamba install -c pytorch -c nvidia "pytorch==2.2.*" torchvision torchaudio pytorch-cuda=12.1

# Install PyG
mamba install -c pyg pyg pytorch-scatter pytorch-sparse pytorch-cluster

# Install other packages
mamba install -c conda-forge \
    "numpy<2" matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    einops wandb \
    cloudpickle pydantic \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# Rich for better terminal output
pip install rich lmdb ase pymatgen matbench-discovery beartype jaxtyping e3nn tabulate pysnooper
