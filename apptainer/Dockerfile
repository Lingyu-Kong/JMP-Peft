FROM rocm/dev-ubuntu-22.04:5.7.1-complete

# Install dependencies
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates curl git

# Install Miniforge3
ENV MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
RUN wget $MINIFORGE_URL -O Miniforge3.sh && \
    /bin/bash Miniforge3.sh -b -p /opt/conda && \
    rm Miniforge3.sh

# Add Miniforge3 to PATH
RUN /opt/conda/condabin/conda init bash && source ~/.bashrc

# Create conda env and install dependencies
RUN mamba create -n pt22 python=3.11 && \
    conda activate pt22 && \
    pip install "torch==2.2.2" torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
