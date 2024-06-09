#!/bin/bash

conda deactivate || true
conda activate rocm60

module reset
module load PrgEnv-gnu/8.4.0
module load craype-accel-amd-gfx90a
module load rocm/6.0
module load amd-mixed/6.0.0

echo "Loaded ROCm 6.0 environment"
