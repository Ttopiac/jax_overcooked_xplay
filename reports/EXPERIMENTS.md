# Experiment Index

## Run 1 — Baseline (Carroll-style)

- **Date**: 2026-03-14
- **Report**: [`reports/run1_report.md`](run1_report.md)
- **Checkpoints**: `run1_backup/checkpoints/` (110 files)
- **Eval results**: `run1_backup/eval_results/` (144 JSON files)
- **Key settings**: 3×64 MLP, 128 steps, 8 envs/layout, Carroll reward, no position shuffle
- **Best cross-play**: BCP (79.1)
- **Resume**: Restore `run1_backup/checkpoints/` to `checkpoints/`, then run eval scripts

## Run 2 — multiHRI-matched

- **Date**: 2026-03-16
- **Report**: [`reports/run2_report.md`](run2_report.md)
- **Checkpoints**: `checkpoints/` (32 files, current)
- **Eval results**: `reports/eval_results/` (141 JSON files, current)
- **Training log**: `reports/run2_log.txt`
- **Key settings**: 2×256 MLP, 500 steps, 42 envs/layout, originaler reward, slot shuffle
- **Best cross-play**: MEP-16 (126.6)
- **Resume**:
  - Full retrain: `bash run_all.sh`
  - From SP Pop: `bash run_from_step1.sh`
  - FCP + eval only: `bash run_from_step6.sh`

## Run 3 — Per-layout H/M/L + eval-based MEP sampling

- **Date**: 2026-03-20
- **Report**: [`reports/run3_report.md`](run3_report.md)
- **Checkpoints**: `checkpoints/` (current)
- **Eval results**: `reports/eval_results/` (current)
- **Key settings**: Same as Run 2 + per-layout H/M/L FCP, eval-based MEP sampling, 3 ckpts/agent (init/mid/final)
- **Best cross-play**: MEP-8 (133.2)
- **Resume**:
  - Full retrain: `bash run_all.sh`
  - From SP Pop: `bash run_from_step1.sh`
  - FCP + eval only: `bash run_from_step6.sh`

## Run 4 — Full grid observation + CNN (pixel-based input)

- **Date**: 2026-03-21
- **Report**: [`reports/run4_report.md`](run4_report.md)
- **Cross-run analysis**: [`reports/cross_run_analysis.md`](cross_run_analysis.md)
- **Checkpoints**: `checkpoints/` (current)
- **Eval results**: `reports/eval_results/` (current)
- **Key settings**: CNNActorCritic, `--obs-encoding full` (26x5x9), 10M timesteps, pop_size=8, 2-3 seeds
- **Best cross-play**: FCP (~45) — all methods significantly worse than MLP runs due to insufficient training and lack of egocentric inductive bias
- **Conclusion**: Egocentric 7x7 + MLP (Run 3) remains far superior; CNN needs 100M+ steps and data augmentation to compete

## Settings Comparison

- **Side-by-side**: [`reports/settings_comparison.md`](settings_comparison.md)

## Conda Environment

- **Name**: `jax_overcooked`
- **Spec**: `environment.yml`
- **Activate**: `conda activate jax_overcooked`
- **Required env vars** (set automatically in run scripts):
  ```bash
  export PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
  export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
  ```
