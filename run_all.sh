#!/bin/bash
# Run all methods (SP, MEP, BCP, FCP) with 3 seeds on all 5 standard
# Overcooked layouts for 2-agent play. 120M timesteps per method.
#
# Workflow per layout-seed:
#   1. Train SP population (16 agents in parallel, 60M each) - shared by BCP & FCP
#   2. Train SP agent (120M)
#   3. Train MEP (120M, pop=16)
#   4. Train BCP agent (60M, loads shared SP pop)
#   5. Train FCP agent (60M, loads shared SP pop)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# CUDA workarounds
export PATH="/home/nathan/anaconda3/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

LAYOUTS="cramped_room asymm_advantages coord_ring forced_coord counter_circuit"
POP_SIZE=16
SP_POP_TIMESTEPS=60000000    # 60M per agent in population
SP_TIMESTEPS=120000000       # 120M for SP method
MEP_TIMESTEPS=120000000      # 120M for MEP method
BCP_TIMESTEPS=60000000       # 60M for BCP agent (+ 60M shared pop = 120M total)
FCP_TIMESTEPS=60000000       # 60M for FCP agent (+ 60M shared pop = 120M total)

echo "=========================================="
echo " Overcooked XPlay Training (JAX)"
echo " SP pop:  ${POP_SIZE} agents × ${SP_POP_TIMESTEPS} (parallel vmap)"
echo " SP:      ${SP_TIMESTEPS}"
echo " MEP:     ${MEP_TIMESTEPS} (pop=${POP_SIZE})"
echo " BCP:     ${BCP_TIMESTEPS} (+ shared pop)"
echo " FCP:     ${FCP_TIMESTEPS} (+ shared pop)"
echo " Layouts: ${LAYOUTS}"
echo " Seeds:   0 1 2"
echo "=========================================="

for LAYOUT in $LAYOUTS; do
    for SEED in 0 1 2; do
        echo ""
        echo "===== Layout=$LAYOUT  Seed=$SEED ====="

        echo "--- [1/5] SP Population (${POP_SIZE} agents, parallel) ---"
        python train_sp_population.py \
            --seed "$SEED" --layout "$LAYOUT" \
            --total-timesteps "$SP_POP_TIMESTEPS" --pop-size "$POP_SIZE"

        echo "--- [2/5] Self-Play (SP) ---"
        python train_sp.py \
            --seed "$SEED" --layout "$LAYOUT" --total-timesteps "$SP_TIMESTEPS"

        echo "--- [3/5] Maximum Entropy Population (MEP) ---"
        python train_mep.py \
            --seed "$SEED" --layout "$LAYOUT" --total-timesteps "$MEP_TIMESTEPS" \
            --pop-size "$POP_SIZE"

        echo "--- [4/5] Best Cross-Play (BCP) ---"
        python train_bcp.py \
            --seed "$SEED" --layout "$LAYOUT" --total-timesteps "$BCP_TIMESTEPS"

        echo "--- [5/5] Fictitious Co-Play (FCP) ---"
        python train_fcp.py \
            --seed "$SEED" --layout "$LAYOUT" --total-timesteps "$FCP_TIMESTEPS"
    done
done

echo ""
echo "=========================================="
echo " All training runs complete."
echo "=========================================="
