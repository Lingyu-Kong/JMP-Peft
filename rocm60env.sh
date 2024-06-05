#!/bin/bash

conda deactivate || true
conda activate rocm60

module reset
module load craype-accel-amd-gfx90a
module load PrgEnv-gnu
module load rocm
module load amd-mixed

echo "Loaded ROCm 6.0.0 environment"
