#!/bin/bash
# Resume Run 2 from Step 6 (FCP) — Steps 0-5 already done.
set -e
cd "$(dirname "$0")"
source activate jax_overcooked
export PATH="/home/nathan/anaconda3/envs/jax_overcooked/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

echo "=========================================="
echo " Run 2 — Resuming from Step 6 (FCP)"
echo "=========================================="

# --- Step 6: FCP ---
echo ""
echo "=== Step 6: FCP ==="
python train_fcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 --wandb

echo "--- Eval: FCP ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/fcp_n2x1_hetero_seed${SEED}.pkl \
        --output reports/eval_results/fcp_ego_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Final: Cross-play evaluation ---
echo ""
echo "=== Final: Cross-play evaluation ==="

for EGO_METHOD in sp mep16 mep8 bcp fcp; do
    echo "  Evaluating $EGO_METHOD vs all partners..."
    for EGO_SEED in 0 1 2; do
        case $EGO_METHOD in
            sp)    EGO_PATH="checkpoints/sp_n2_seed${EGO_SEED}.pkl" ;;
            mep16) EGO_PATH="checkpoints/mep16_seed${EGO_SEED}.pkl" ;;
            mep8)  EGO_PATH="checkpoints/mep8_seed${EGO_SEED}.pkl" ;;
            bcp)   EGO_PATH="checkpoints/bcp_n2x1_hetero_seed${EGO_SEED}.pkl" ;;
            fcp)   EGO_PATH="checkpoints/fcp_n2x1_hetero_seed${EGO_SEED}.pkl" ;;
        esac

        for P_SEED in 3 4; do
            LAYOUTS="cramped_room,asymm_advantages,coord_ring,forced_coord,counter_circuit"
            [ "$EGO_METHOD" = "bcp" ] && LAYOUTS="cramped_room,asymm_advantages,coord_ring"

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/sp_pop_seed${P_SEED}.pkl --partner-type pop \
                --layouts $LAYOUTS \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_sp_s${P_SEED}.json

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/mep16_pop_seed${P_SEED}.pkl --partner-type pop \
                --layouts $LAYOUTS \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_mep16_s${P_SEED}.json

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/mep8_pop_seed${P_SEED}.pkl --partner-type pop \
                --layouts $LAYOUTS \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_mep8_s${P_SEED}.json

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/bc_seed${P_SEED}.pkl --partner-type bc \
                --layouts cramped_room,asymm_advantages,coord_ring \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_bc_s${P_SEED}.json
        done
    done
    python print_tables.py
done

echo ""
echo "=========================================="
echo " FINAL RESULTS"
echo "=========================================="
python print_tables.py

echo ""
echo " All training and evaluation complete."
echo " Results in reports/eval_results/"
