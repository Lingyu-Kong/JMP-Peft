#!/bin/bash

conda deactivate || true

module reset
module load craype-accel-amd-gfx90a
module load PrgEnv-gnu
module load rocm
module load amd-mixed
module load cray-python
module load omniperf

# conda deactivate || true
# conda activate rocm53

echo "Loaded ROCm 5.3.0 environment"
