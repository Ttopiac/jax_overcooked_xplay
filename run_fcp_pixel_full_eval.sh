#!/bin/bash
# Full evaluation of FCP pixel ego (Run 4) against all Run 3 partners.
# Includes H/M/L tier cross-play for population partners.
set -e
cd "$(dirname "$0")"
source activate jax_overcooked
export PATH="/home/nathan/anaconda3/envs/jax_overcooked/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

echo "=== Full FCP Pixel Ego Evaluation ==="

# Run the comprehensive Python eval script
python eval_fcp_pixel_full.py 2>&1 | tee reports/fcp_pixel_full_eval.txt

echo "=== Done ==="
