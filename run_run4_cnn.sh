#!/bin/bash
# Run 4: Full grid observation + CNN (pixel-based input).
# All methods use --obs-encoding full with CNNActorCritic instead of MLP.
# Trains SP Pop, MEP Pop+Ego, and FCP from scratch.
# BCP is skipped (BC partners use egocentric encoding, incompatible with full mode).
set -e
cd "$(dirname "$0")"
source activate jax_overcooked
export PATH="/home/nathan/anaconda3/envs/jax_overcooked/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

mkdir -p checkpoints reports/eval_results

echo "=========================================="
echo " Run 4 — Full grid observation + CNN"
echo "=========================================="

# --- Step 1: SP Population (full obs + CNN) ---
echo ""
echo "=== Step 1: SP Population (full obs + CNN) ==="
python train_sp_population.py --seeds 0,1,2,3,4 --pop-size 16 --num-checkpoints 35 \
    --total-timesteps 100000000 --obs-encoding full --batch-size 8 --wandb

# --- Step 2: Evaluate SP pop per layout (for FCP) ---
echo ""
echo "=== Step 2: Evaluate SP pop per layout ==="
python evaluate_pop_per_layout.py --seeds 0,1,2,3,4

# --- Step 3: SP Ego eval ---
echo ""
echo "=== Step 3: SP Ego self-play eval ==="
# SP ego checkpoints don't exist for full mode yet - use SP pop final params as ego
# (SP ego = single agent trained via self-play, but we can evaluate pop's best agents)
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-pop --agent-path checkpoints/sp_pop_seed${SEED}.pkl \
        --obs-encoding full \
        --output reports/eval_results/sp_pop_selfplay_cnn_seed${SEED}.json
done

# --- Step 4: MEP-16 (full obs + CNN) ---
echo ""
echo "=== Step 4: MEP-16 Population + Ego (full obs + CNN) ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 100000000 --ego-timesteps 100000000 \
    --pop-size 16 --epoch-updates 1 --num-agents 2 --num-unseen 1 \
    --obs-encoding full --wandb

echo "--- Eval: MEP-16 ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/mep16_seed${SEED}.pkl \
        --obs-encoding full \
        --output reports/eval_results/mep16_ego_selfplay_cnn_seed${SEED}.json
done

# --- Step 5: MEP-8 (full obs + CNN) ---
echo ""
echo "=== Step 5: MEP-8 Population + Ego (full obs + CNN) ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 100000000 --ego-timesteps 100000000 \
    --pop-size 8 --epoch-updates 1 --num-agents 2 --num-unseen 1 \
    --obs-encoding full --wandb

echo "--- Eval: MEP-8 ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/mep8_seed${SEED}.pkl \
        --obs-encoding full \
        --output reports/eval_results/mep8_ego_selfplay_cnn_seed${SEED}.json
done

# --- Step 6: FCP (full obs + CNN) ---
echo ""
echo "=== Step 6: FCP (full obs + CNN) ==="
python train_fcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 \
    --obs-encoding full --wandb

echo "--- Eval: FCP ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/fcp_n2x1_hetero_seed${SEED}.pkl \
        --obs-encoding full \
        --output reports/eval_results/fcp_ego_selfplay_cnn_seed${SEED}.json
done

# --- Step 7: Cross-play evaluation ---
echo ""
echo "=== Step 7: Cross-play evaluation (CNN methods) ==="

for EGO_METHOD in mep16 mep8 fcp; do
    echo "  Evaluating $EGO_METHOD vs all CNN-trained partners..."
    for EGO_SEED in 0 1 2; do
        case $EGO_METHOD in
            mep16) EGO_PATH="checkpoints/mep16_seed${EGO_SEED}.pkl" ;;
            mep8)  EGO_PATH="checkpoints/mep8_seed${EGO_SEED}.pkl" ;;
            fcp)   EGO_PATH="checkpoints/fcp_n2x1_hetero_seed${EGO_SEED}.pkl" ;;
        esac

        for P_SEED in 3 4; do
            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/sp_pop_seed${P_SEED}.pkl --partner-type pop \
                --obs-encoding full \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_sp_s${P_SEED}_cnn.json

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/mep16_pop_seed${P_SEED}.pkl --partner-type pop \
                --obs-encoding full \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_mep16_s${P_SEED}_cnn.json

            python evaluate.py --mode cross-play --ego-path $EGO_PATH \
                --partner-path checkpoints/mep8_pop_seed${P_SEED}.pkl --partner-type pop \
                --obs-encoding full \
                --output reports/eval_results/cross_${EGO_METHOD}_s${EGO_SEED}_vs_mep8_s${P_SEED}_cnn.json
        done
    done
done

echo ""
echo "=========================================="
echo " FINAL RESULTS"
echo "=========================================="
python print_tables.py

echo ""
echo " Run 4 complete. Results in reports/eval_results/"
