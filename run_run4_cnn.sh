#!/bin/bash
# Run 4: Human-equivalent pixel observations + IMPALA ResNet CNN.
# Agents see the same rendered kitchen image a human sees (32x32 tile sprites).
# All methods use --obs-encoding pixel with IMPALACNNActorCritic (~1.6M params).
# Trains SP Pop, MEP Pop+Ego, and FCP from scratch. 100M steps each.
# BCP is skipped (BC partners use egocentric encoding, incompatible with pixel mode).
set -e
cd "$(dirname "$0")"
source activate jax_overcooked
export PATH="/home/nathan/anaconda3/envs/jax_overcooked/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

mkdir -p checkpoints reports/eval_results

LOGFILE="reports/run4_log.txt"
log() { echo "[$(date '+%Y-%m-%d %H:%M')] $1" | tee -a "$LOGFILE"; }

log "=========================================="
log " Run 4 — Pixel obs (160x288x3) + IMPALA CNN"
log "=========================================="

# --- Step 1: SP Population (pixel obs + IMPALA CNN) ---
log ""
log "=== Step 1: SP Population (pixel obs + IMPALA CNN) ==="
python train_sp_population.py --seeds 0,1,2,3,4 --pop-size 16 --num-checkpoints 35 \
    --total-timesteps 100000000 --obs-encoding pixel --batch-size 4 --wandb
log "Step 1 done."

# --- Step 2: Evaluate SP pop per layout (for FCP) ---
log ""
log "=== Step 2: Evaluate SP pop per layout ==="
python evaluate_pop_per_layout.py --seeds 0,1,2,3,4
log "Step 2 done."

# --- Step 3: SP Pop self-play eval ---
log ""
log "=== Step 3: SP Pop self-play eval ==="
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-pop --agent-path checkpoints/sp_pop_seed${SEED}.pkl \
        --output reports/eval_results/sp_pop_selfplay_pixel_seed${SEED}.json
done
python print_tables.py
log "Step 3 done."

# --- Step 4: MEP-16 (pixel obs + IMPALA CNN) ---
log ""
log "=== Step 4: MEP-16 Population + Ego (pixel obs) ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 100000000 --ego-timesteps 100000000 \
    --pop-size 16 --epoch-updates 1 --num-agents 2 --num-unseen 1 \
    --obs-encoding pixel --wandb
log "Step 4 done."

log "--- Eval: MEP-16 ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/mep16_seed${SEED}.pkl \
        --output reports/eval_results/mep16_ego_selfplay_pixel_seed${SEED}.json
done
python print_tables.py

# --- Step 5: MEP-8 (pixel obs + IMPALA CNN) ---
log ""
log "=== Step 5: MEP-8 Population + Ego (pixel obs) ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 100000000 --ego-timesteps 100000000 \
    --pop-size 8 --epoch-updates 1 --num-agents 2 --num-unseen 1 \
    --obs-encoding pixel --wandb
log "Step 5 done."

log "--- Eval: MEP-8 ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/mep8_seed${SEED}.pkl \
        --output reports/eval_results/mep8_ego_selfplay_pixel_seed${SEED}.json
done
python print_tables.py

# --- Step 6: BCP (pixel ego + egocentric BC partners) ---
log ""
log "=== Step 6: BCP (pixel ego, egocentric BC partners) ==="
python train_bcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 \
    --obs-encoding pixel --bc-anneal-horizon 8000 --wandb
log "Step 6 done."

log "--- Eval: BCP ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/bcp_n2x1_hetero_seed${SEED}.pkl \
        --layouts cramped_room,asymm_advantages,coord_ring \
        --output reports/eval_results/bcp_ego_selfplay_pixel_seed${SEED}.json
done
python print_tables.py

# --- Step 7: FCP (pixel obs + IMPALA CNN) ---
log ""
log "=== Step 7: FCP (pixel obs) ==="
python train_fcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 \
    --obs-encoding pixel --wandb
log "Step 7 done."

log "--- Eval: FCP ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/fcp_n2x1_hetero_seed${SEED}.pkl \
        --output reports/eval_results/fcp_ego_selfplay_pixel_seed${SEED}.json
done
python print_tables.py

# --- Step 8: Cross-play evaluation ---
log ""
log "=== Step 8: Cross-play evaluation (pixel methods) ==="

for EGO_METHOD in mep16 mep8 bcp fcp; do
    log "  Evaluating $EGO_METHOD vs all pixel-trained partners..."
    for EGO_SEED in 0 1 2; do
        case $EGO_METHOD in
            mep16) EGO_PATH="checkpoints/mep16_seed${EGO_SEED}.pkl" ;;
            mep8)  EGO_PATH="checkpoints/mep8_seed${EGO_SEED}.pkl" ;;
            bcp)   EGO_PATH="checkpoints/bcp_n2x1_hetero_seed${EGO_SEED}.pkl" ;;
            fcp)   EGO_PATH="checkpoints/fcp_n2x1_hetero_seed${EGO_SEED}.pkl" ;;
        esac

        LAYOUTS="cramped_room,asymm_advantages,coord_ring,forced_coord,counter_circuit"
        [ "$EGO_METHOD" = "bcp" ] && LAYOUTS="cramped_room,asymm_advantages,coord_ring"

        for P_SEED in 3 4; do
            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/sp_pop_seed${P_SEED}.pkl --partner-type pop \
                --layouts $LAYOUTS \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_sp_s${P_SEED}_pixel.json

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/mep16_pop_seed${P_SEED}.pkl --partner-type pop \
                --layouts $LAYOUTS \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_mep16_s${P_SEED}_pixel.json

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/mep8_pop_seed${P_SEED}.pkl --partner-type pop \
                --layouts $LAYOUTS \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_mep8_s${P_SEED}_pixel.json

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/bc_seed${P_SEED}.pkl --partner-type bc \
                --layouts cramped_room,asymm_advantages,coord_ring \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_bc_s${P_SEED}_pixel.json
        done
    done
done

log ""
log "=========================================="
log " FINAL RESULTS"
log "=========================================="
python print_tables.py

log ""
log " Run 4 complete. Results in reports/eval_results/"
