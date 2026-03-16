#!/bin/bash
# Resume Run 2 from Step 1 (BC already done).
set -e
cd "$(dirname "$0")"
source activate jax_overcooked
export PATH="/home/nathan/anaconda3/envs/jax_overcooked/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

mkdir -p checkpoints reports/eval_results

echo "=========================================="
echo " Run 2 — Resuming from Step 1"
echo "=========================================="

# --- Step 1: SP Population ---
echo ""
echo "=== Step 1: SP Population (80 agents, batch_size=8) ==="
python train_sp_population.py --seeds 0,1,2,3,4 \
    --total-timesteps 100000000 --pop-size 16 --num-agents 2 --batch-size 8 --wandb

echo "--- Eval: SP Pop self-play (seeds 3,4) ---"
for SEED in 3 4; do
    python evaluate.py --mode self-play-pop --agent-path checkpoints/sp_pop_seed${SEED}.pkl \
        --agent-type pop --output reports/eval_results/sp_pop_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Step 2: SP Ego ---
echo ""
echo "=== Step 2: SP Ego ==="
python train_sp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --wandb

echo "--- Eval: SP ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/sp_n2_seed${SEED}.pkl \
        --output reports/eval_results/sp_ego_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Step 3: MEP-16 ---
echo ""
echo "=== Step 3: MEP-16 ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 100000000 --ego-timesteps 100000000 \
    --pop-size 16 --epoch-updates 1 --num-agents 2 --num-unseen 1 --wandb

echo "--- Eval: MEP-16 Pop self-play (seeds 3,4) ---"
for SEED in 3 4; do
    python evaluate.py --mode self-play-pop --agent-path checkpoints/mep16_pop_seed${SEED}.pkl \
        --agent-type pop --output reports/eval_results/mep16_pop_selfplay_seed${SEED}.json
done
echo "--- Eval: MEP-16 ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/mep16_seed${SEED}.pkl \
        --output reports/eval_results/mep16_ego_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Step 4: MEP-8 ---
echo ""
echo "=== Step 4: MEP-8 ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 100000000 --ego-timesteps 100000000 \
    --pop-size 8 --epoch-updates 1 --num-agents 2 --num-unseen 1 --wandb

echo "--- Eval: MEP-8 Pop self-play (seeds 3,4) ---"
for SEED in 3 4; do
    python evaluate.py --mode self-play-pop --agent-path checkpoints/mep8_pop_seed${SEED}.pkl \
        --agent-type pop --output reports/eval_results/mep8_pop_selfplay_seed${SEED}.json
done
echo "--- Eval: MEP-8 ego self-play ---"
for SEED in 0 1 2; do
    python evaluate.py --mode self-play-ego --agent-path checkpoints/mep8_seed${SEED}.pkl \
        --output reports/eval_results/mep8_ego_selfplay_seed${SEED}.json
done
python print_tables.py

# --- Step 5: BCP ---
echo ""
echo "=== Step 5: BCP ==="
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
                \
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
