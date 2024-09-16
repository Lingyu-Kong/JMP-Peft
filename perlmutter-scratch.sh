export HF_DATASETS_CACHE="/global/cfs/cdirs/m3641/Nima/hf-datasets-cache" # in ~/.bashrc

# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# chmod +x Miniconda3-latest-Linux-x86_64.sh
# yes yes | ./Miniconda3-latest-Linux-x86_64.sh
# source ~/miniconda3/bin/activate
# yes yes | conda update conda


# conda create -n jmp-310-pytorch230 python=3.10 -y
# conda activate jmp-310-pytorch230
conda activate jmp
# conda install mamba -n base -c conda-forge

# Install PyTorch
mamba install -c pytorch -c nvidia "pytorch==2.2.*" torchvision torchaudio pytorch-cuda=12.1 -y

# Install PyG
mamba install -c pyg pyg pytorch-scatter pytorch-sparse pytorch-cluster -y

# Install other packages
mamba install -y -c conda-forge \
    "numpy<2" matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    einops wandb \
    cloudpickle pydantic \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# Rich for better terminal output
pip install rich lmdb ase pymatgen matbench-discovery beartype jaxtyping e3nn tabulate pysnooper matplotlib
pip install nshtrainer==0.29.1 ==0.17.0
pip install -e .