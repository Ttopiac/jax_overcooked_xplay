#!/bin/bash
# FCP ego training with pixel observations (IMPALA shallow CNN).
# Ego: 8×8 pixel CNN, Partners: egocentric MLP from Run 3's SP Pop.
# 6 seeds, 100M steps. Estimated ~4.3h.
set -e
cd "$(dirname "$0")"
source activate jax_overcooked
export PATH="/home/nathan/anaconda3/envs/jax_overcooked/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

mkdir -p checkpoints reports/eval_results

LOGFILE="reports/run4_fcp_log.txt"
log() { echo "[$(date '+%Y-%m-%d %H:%M')] $1" | tee -a "$LOGFILE"; }

log "=== FCP Pixel Ego Training (6 seeds, 100M steps) ==="

# --- Train FCP ego ---
log "Training FCP ego..."
python train_fcp.py --seeds 0,1,2,3,4,5 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 \
    --obs-encoding pixel --wandb

log "FCP ego training done."

# --- Eval: FCP ego self-play ---
log "Evaluating FCP ego self-play..."
for SEED in 0 1 2 3 4 5; do
    python evaluate.py --mode self-play-ego \
        --agent-path checkpoints/fcp_n2x1_hetero_seed${SEED}.pkl \
        --output reports/eval_results/fcp_ego_selfplay_pixel_seed${SEED}.json
done

# --- Eval: FCP ego cross-play vs Run 3 partners ---
log "Evaluating FCP ego cross-play..."
for EGO_SEED in 0 1 2 3 4 5; do
    EGO_PATH="checkpoints/fcp_n2x1_hetero_seed${EGO_SEED}.pkl"
    for P_SEED in 3 4; do
        python evaluate.py --mode cross-play --ego-path $EGO_PATH \
            --partner-path checkpoints/sp_pop_seed${P_SEED}.pkl --partner-type pop \
            --output reports/eval_results/cross_fcp_pixel_s${EGO_SEED}_vs_sp_s${P_SEED}.json

        python evaluate.py --mode cross-play --ego-path $EGO_PATH \
            --partner-path checkpoints/mep16_pop_seed${P_SEED}.pkl --partner-type pop \
            --output reports/eval_results/cross_fcp_pixel_s${EGO_SEED}_vs_mep16_s${P_SEED}.json

        python evaluate.py --mode cross-play --ego-path $EGO_PATH \
            --partner-path checkpoints/mep8_pop_seed${P_SEED}.pkl --partner-type pop \
            --output reports/eval_results/cross_fcp_pixel_s${EGO_SEED}_vs_mep8_s${P_SEED}.json

        python evaluate.py --mode cross-play --ego-path $EGO_PATH \
            --partner-path checkpoints/bc_seed${P_SEED}.pkl --partner-type bc \
            --layouts cramped_room,asymm_advantages,coord_ring \
            --output reports/eval_results/cross_fcp_pixel_s${EGO_SEED}_vs_bc_s${P_SEED}.json
    done
done

log "Evaluation done."
python print_tables.py | tee -a "$LOGFILE"

log "=== FCP Pixel Run Complete ==="
