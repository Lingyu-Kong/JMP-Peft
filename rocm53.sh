#!/bin/bash

# conda deactivate || true
# conda activate rocm53

module reset
module load craype-accel-amd-gfx90a
module load PrgEnv-gnu
module load rocm
module load amd-mixed
echo "Loaded ROCm 5.3.0 environment"
