# CLAUDE.md

Instructions for working in this repo. Keep this file lean — it's loaded into every session.

## What this repo is

JAX-based 2-player Overcooked zero-shot coordination (ZSC) training + evaluation. Methods: SP, SP Population, MEP, BCP, FCP, all built on a shared N-XPlay framework (`nxplay.py`). End goal: prepare pixel-observation agents for sim-to-real transfer to dual-arm robotic cooking.

## Layout

| Path | What it is |
|------|------------|
| `train_sp.py`, `train_sp_population.py`, `train_mep.py`, `train_bc.py`, `train_bcp.py`, `train_fcp.py` | Training scripts — one per method |
| `networks.py` | `ActorCritic` (MLP), `CNNActorCritic`, `IMPALACNNActorCritic` (pixel) |
| `obs_encoding.py` | `ObsEncoder` with modes `egocentric` / `full` / `pixel`; pixel uses `SpriteAtlas` with palette indexing |
| `overcooked_env/` | N-agent Overcooked extended from JaxMARL |
| `ppo.py`, `reward_shaping.py`, `nxplay.py` | Shared PPO, dense shaping, N-XPlay utilities |
| `evaluate.py`, `evaluate_pop_per_layout.py`, `eval_ego_vs_ego.py`, `eval_fcp_pixel_full.py` | Evaluation scripts |
| `checkpoints/` | All `.pkl` checkpoints. See `checkpoints/CLAUDE.md` |
| `reports/` | Numbered per-run reports. See `reports/README.md` |
| `run_*.sh` | Entry-point shell scripts for each run |

## Environment

- Conda env: `jax_overcooked` (see `environment.yml`). CUDA 12, JAX 0.4.38, Flax 0.10.4.
- Single GPU (~24 GB VRAM). Pixel training fits with palette-indexed obs + IMPALA shallow CNN.

## Core conventions

- **Observation modes** (`obs_encoding.py`): `egocentric` (7×7×26 = 1274-dim, default), `full` (26×5×9), `pixel` (palette indices decoded to (3, 40, 72) at 8 px/tile).
- **Checkpoints** are pickles. Config is stored in `data["config"]` (including `obs_encoding`, `hidden_dims`). See `checkpoints/CLAUDE.md` for naming + safety.
- **Layouts**: `cramped_room`, `asymm_advantages`, `coord_ring`, `forced_coord`, `counter_circuit`. BCP uses only the first three (BC dataset coverage).
- **H/M/L partner tiers** (FCP, evals): computed **per layout** via `categorize_by_skill_per_layout` — 1 agent per tier per layout, picked from per-layout self-play scores. Never use global tiers.
- **Runs** are numbered (Run 1 → Run 4). Each run freezes hyperparameters; changes go in a new run. Reports live under `reports/0N_run{N}_*/`.

## Session-safety rules

This repo has many long-running experiments. Work can be wiped by one bad command. When in doubt, ask.

- **Never overwrite checkpoints** without confirming a backup exists. See `checkpoints/CLAUDE.md`.
- **Never `git reset --hard`, `git clean -fd`, or force-push** without explicit user instruction.
- **Training runs are long** (hours to days). Before starting a new run, confirm: (a) the previous run is backed up, (b) the script won't overwrite active checkpoints, (c) the user has agreed to start.
- **Evaluation is cheap** (minutes); training is expensive. Prefer re-running eval over re-running training.
- **Canonical results**: `reports/05_run3_vs_run4_crossplay_eval/crossplay_eval.md` has the full cross-play comparison. `reports/04_run4_cnn_pixel_human_equivalent/report.md` has the Run 4 architecture journey and settings.

## Git

- Do NOT include `Co-Authored-By` lines in commit messages.
- Commit after each meaningful milestone (training done, eval done, report updated).
- Never commit `.pkl` checkpoints, wandb runs, or `__pycache__`.

## Terminology

- **SP**: Self-play. **SP Pop**: population of 16 self-play agents with 35 saved checkpoints each.
- **MEP**: Max Entropy Population. **MEP-16**, **MEP-8**: population size. Phase 1 trains the pop with entropy regularization; Phase 2 trains ego against it.
- **BCP**: Behavior-Cloning-Partner Play (ego trained against a BC policy fit to human data).
- **FCP**: Fictitious Co-Play (ego trained against H/M/L tiers from the SP population).
- **N-XPlay**: N-agent game, X unseen partners; see `nxplay.py`.
- **Ego vs Ego**: cross-play between ego agents from different methods.
