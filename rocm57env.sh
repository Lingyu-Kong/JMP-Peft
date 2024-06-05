#!/bin/bash

conda deactivate || true
conda activate rocm57

module purge
module load PrgEnv-gnu
module load rocm/5.7.1

echo "Loaded ROCm 5.7.1 environment"
