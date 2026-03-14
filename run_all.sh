#!/bin/bash
# Run all methods (SP, MEP, BCP, FCP) with 3 seeds on 2-agent Overcooked.
# Usage: bash run_all.sh [LAYOUT] [TOTAL_TIMESTEPS]

set -e

LAYOUT="${1:-cramped_room}"
TIMESTEPS="${2:-5000000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo " Layout:     $LAYOUT"
echo " Timesteps:  $TIMESTEPS per phase"
echo "=========================================="

for SEED in 0 1 2; do
    echo ""
    echo "===== Seed $SEED ====="

    echo "--- Self-Play (SP) ---"
    python "$SCRIPT_DIR/train_sp.py" \
        --seed "$SEED" --layout "$LAYOUT" --total-timesteps "$TIMESTEPS"

    echo "--- Maximum Entropy Population (MEP) ---"
    python "$SCRIPT_DIR/train_mep.py" \
        --seed "$SEED" --layout "$LAYOUT" --total-timesteps "$TIMESTEPS"

    echo "--- Best Cross-Play (BCP) ---"
    python "$SCRIPT_DIR/train_bcp.py" \
        --seed "$SEED" --layout "$LAYOUT" \
        --sp-total-timesteps "$TIMESTEPS" --bcp-total-timesteps "$TIMESTEPS"

    echo "--- Fictitious Co-Play (FCP) ---"
    python "$SCRIPT_DIR/train_fcp.py" \
        --seed "$SEED" --layout "$LAYOUT" \
        --sp-total-timesteps "$TIMESTEPS" --fcp-total-timesteps "$TIMESTEPS"
done

echo ""
echo "=========================================="
echo " All training runs complete."
echo "=========================================="
