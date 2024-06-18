FROM rocm/rocm-terminal:6.0.2

USER root

# Install dependencies for building C++ code
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip

# Install Miniforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b && \
    rm Miniforge3-Linux-x86_64.sh

# Set up Miniforge
ENV PATH="/root/miniforge3/bin:${PATH}"

# Update the base environment to Python 3.10
RUN conda install -y python=3.10

# Install PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install PyTorch Geometric dependencies
RUN cd /tmp && \
    git clone --recursive https://github.com/rusty1s/pytorch_scatter.git && \
    cd pytorch_scatter && \
    python setup.py install && \
    cd ..

RUN cd /tmp && \
    git clone --recursive https://github.com/rusty1s/pytorch_sparse.git && \
    cd pytorch_sparse && \
    python setup.py install && \
    cd ..

RUN cd /tmp && \
    git clone --recursive https://github.com/rusty1s/pytorch_cluster.git && \
    cd pytorch_cluster && \
    python setup.py install && \
    cd ..

# Install PyG
RUN pip install torch_geometric

# Install other dependencies
RUN pip install lightning pytorch-lightning torchmetrics "torch==2.4.0.dev20240520+rocm6.0"
RUN pip install frozendict wandb lovely-tensors matplotlib ipykernel numba cloudpickle einops pytest wrapt numpy nbformat typing-extensions varname pandas pyyaml plotly tqdm requests ipywidgets scikit-learn seaborn lovely-numpy networkx sympy
RUN pip install pymatgen ase jarvis-tools pydantic rich lmdb beartype jaxtyping ruff
