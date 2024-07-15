export HF_DATASETS_CACHE="/global/cfs/cdirs/m3641/Nima/hf-datasets-cache" # in ~/.bashrc

mamba create -n jmp-310 python=3.10
conda activate jmp-310

# Install base dependencies
mamba install -y -c conda-forge -c pytorch -c nvidia -c pyg \
    pytorch torchvision torchaudio pytorch-cuda=12.1 \
    pyg pyg pytorch-scatter pytorch-sparse pytorch-cluster \
    "numpy<2" matplotlib seaborn sympy pandas numba scikit-learn plotly nbformat ipykernel ipywidgets tqdm pyyaml networkx \
    pytorch-lightning torchmetrics lightning \
    einops wandb \
    cloudpickle pydantic \
    frozendict wrapt varname typing-extensions lovely-tensors lovely-numpy requests pytest nbval

# Rich for better terminal output
pip install rich lmdb ase pymatgen matbench-discovery beartype jaxtyping e3nn tabulate pysnooper
