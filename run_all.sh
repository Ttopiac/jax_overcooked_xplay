#!/bin/bash
# Run all methods (SP, MEP, BCP, FCP) with 3 seeds on all 5 standard
# Overcooked layouts for 2-agent play.
#
# Layouts (from zero-shot coordination literature):
#   cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit
#
# Usage: bash run_all.sh [TOTAL_TIMESTEPS]

set -e

TIMESTEPS="${1:-5000000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Use JAX-bundled CUDA toolkit to avoid ptxas version mismatch
export PATH="/home/nathan/anaconda3/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"

LAYOUTS="cramped_room asymm_advantages coord_ring forced_coord counter_circuit"

echo "=========================================="
echo " Overcooked XPlay Training (JAX)"
echo " Timesteps per phase: $TIMESTEPS"
echo " Layouts: $LAYOUTS"
echo " Seeds: 0 1 2"
echo "=========================================="

for LAYOUT in $LAYOUTS; do
    for SEED in 0 1 2; do
        echo ""
        echo "===== Layout=$LAYOUT  Seed=$SEED ====="

        echo "--- Self-Play (SP) ---"
        python train_sp.py \
            --seed "$SEED" --layout "$LAYOUT" --total-timesteps "$TIMESTEPS"

        echo "--- Maximum Entropy Population (MEP) ---"
        python train_mep.py \
            --seed "$SEED" --layout "$LAYOUT" --total-timesteps "$TIMESTEPS"

        echo "--- Best Cross-Play (BCP) ---"
        python train_bcp.py \
            --seed "$SEED" --layout "$LAYOUT" \
            --sp-total-timesteps "$TIMESTEPS" --bcp-total-timesteps "$TIMESTEPS"

        echo "--- Fictitious Co-Play (FCP) ---"
        python train_fcp.py \
            --seed "$SEED" --layout "$LAYOUT" \
            --sp-total-timesteps "$TIMESTEPS" --fcp-total-timesteps "$TIMESTEPS"
    done
done

echo ""
echo "=========================================="
echo " All training runs complete."
echo "=========================================="
