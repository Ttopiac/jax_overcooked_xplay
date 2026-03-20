#!/bin/bash
# Run 3: Retrain ego agents only with per-env partner sampling.
# Reuses SP Pop, SP Ego, MEP Pop, and BC checkpoints from Run 2.
# Only retrains: MEP-16 ego, MEP-8 ego, BCP ego, FCP ego.
set -e
cd "$(dirname "$0")"
source activate jax_overcooked
export PATH="/home/nathan/anaconda3/envs/jax_overcooked/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

mkdir -p checkpoints reports/eval_results

echo "=========================================="
echo " Run 3 — Per-env partner sampling (ego retraining only)"
echo "=========================================="

# Verify Run 2 prerequisites exist
for f in checkpoints/sp_pop_seed0.pkl checkpoints/sp_pop_seed1.pkl checkpoints/sp_pop_seed2.pkl \
         checkpoints/sp_pop_seed3.pkl checkpoints/sp_pop_seed4.pkl \
         checkpoints/bc_seed0.pkl checkpoints/bc_seed3.pkl checkpoints/bc_seed4.pkl \
         checkpoints/mep16_pop_seed0.pkl checkpoints/mep16_pop_seed3.pkl \
         checkpoints/mep8_pop_seed0.pkl checkpoints/mep8_pop_seed3.pkl; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing prerequisite: $f"
        exit 1
    fi
done
echo "All Run 2 prerequisites found."

# --- Step 1: MEP-16 Ego (retraining with per-env partners) ---
echo ""
echo "=== Step 1: MEP-16 Ego (per-env partner sampling) ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --skip-pop --ego-timesteps 100000000 \
    --pop-size 16 --epoch-updates 1 --num-agents 2 --num-unseen 1 --wandb

echo "--- Eval: MEP-16 ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/mep16_seed${SEED}.pkl \
        --output reports/eval_results/mep16_ego_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Step 2: MEP-8 Ego (retraining with per-env partners) ---
echo ""
echo "=== Step 2: MEP-8 Ego (per-env partner sampling) ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --skip-pop --ego-timesteps 100000000 \
    --pop-size 8 --epoch-updates 1 --num-agents 2 --num-unseen 1 --wandb

echo "--- Eval: MEP-8 ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/mep8_seed${SEED}.pkl \
        --output reports/eval_results/mep8_ego_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Step 3: BCP (retraining with per-env partners) ---
echo ""
echo "=== Step 3: BCP (per-env partner sampling) ==="
python train_bcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 \
    --bc-anneal-horizon 8000 --wandb

echo "--- Eval: BCP ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/bcp_n2x1_hetero_seed${SEED}.pkl \
        --layouts cramped_room,asymm_advantages,coord_ring \
        --output reports/eval_results/bcp_ego_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Step 4: FCP (retraining with per-env partners) ---
echo ""
echo "=== Step 4: FCP (per-env partner sampling) ==="
python train_fcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 --wandb

echo "--- Eval: FCP ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/fcp_n2x1_hetero_seed${SEED}.pkl \
        --output reports/eval_results/fcp_ego_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Step 5: Cross-play evaluation (all methods) ---
echo ""
echo "=== Step 5: Cross-play evaluation ==="

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
echo " Run 3 complete. Results in reports/eval_results/"
