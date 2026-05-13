# Comparison: This Repo vs JaxMARL Baselines

This document compares our zero-shot coordination (ZSC) codebase with the [JaxMARL](https://github.com/flairox/jaxmarl) Overcooked baselines (`baselines/IPPO/ippo_ff_overcooked.py`, `ippo_cnn_overcooked.py`). Both are verified by reading the source code.

Our repo extends the JaxMARL Overcooked environment with ZSC training methods, pixel observations, and cross-play evaluation infrastructure.

---

## Observation

| | JaxMARL | Ours |
|---|---------|------|
| Raw env obs | (H, W, 26) per agent, ego/teammate channels swapped | Same (inherited from JaxMARL) |
| FF processing | Flatten raw obs to vector | Egocentric 7×7 crop centered on agent + rotation so agent faces south, then flatten to 1274-dim. Not in JaxMARL. |
| CNN processing | Feed raw (H, W, 26) directly to CNN | Feed raw (H, W, 26) padded to (26, 5, 9) to CNN |
| Pixel rendering | None | Pre-rendered 8×8 tile sprites matching the human visualizer. Palette-indexed (76 colors, 1 byte/pixel). CNN decodes palette → RGB at forward time. |
| Encoding modes | One (raw env obs, used differently per script) | Three: `egocentric` (7×7 rotated), `full` (padded grid), `pixel` (sprite images) — selected via `--obs-encoding` |

## Network Architecture

| | JaxMARL | Ours |
|---|---------|------|
| MLP | 2×64 Dense, tanh or relu | 2×256 Dense, relu |
| MLP actor/critic | Separate backbones (both start from input `x`) | Separate backbones (same pattern) |
| CNN (grid obs) | 3 conv layers (32@5×5, 32@3×3, 32@3×3), shared backbone → Dense(64) | 3 conv layers (32@3×3, 64@3×3, 64@3×3), shared backbone → Dense(256) |
| CNN (pixel obs) | None | IMPALA shallow: Conv(16, 8×8, stride=4) → Conv(32, 4×4, stride=2) → Dense(256), with palette decode. Architecture from Espeholt et al., ICML 2018, Figure 3. |
| Param count (MLP) | ~17K (2×64) | ~367K (2×256) |
| Param count (pixel CNN) | N/A | ~185K |

## Training

| | JaxMARL | Ours |
|---|---------|------|
| Algorithm | IPPO — shared params between both game agents | PPO — shared params in SP/SP Pop self-play; separate params for ego vs partner in FCP/BCP/MEP ego |
| PPO code | Inline in each script | Shared `ppo.py` (Transition, GAE, ppo_update) imported by all training scripts |
| Reward shaping | Inline annealing: `reward + shaped_reward * anneal(t)` | Separate `reward_shaping.py` with configurable modes (shaped, sparse, originaler) and independent annealing schedule |
| Training methods | Self-play only | SP, SP Population (16 agents × 35 checkpoints), MEP (population with entropy bonus + ego with eval-based sampling), BCP (ego vs BC partners), FCP (ego vs H/M/L population tiers) |
| Population training | None | SP Pop trains 16 diverse agents; MEP Pop trains with cross-entropy bonus across population |
| Partner framework | None (both agents always share one policy) | `nxplay.py`: N-agent game with X unseen partner slots, per-env partner sampling, position shuffling |
| H/M/L partner tiers | None | Per-layout skill categorization via `categorize_by_skill_per_layout`: H = best agent on that layout, M = closest to 50% of H, L = worst. Different agents per layout. |
| Dual encoder | None | Pixel CNN ego + egocentric MLP partners — different observation encodings and network architectures for ego and partner within the same training run |
| Config system | Hydra YAML files | argparse CLI flags |
| Default timesteps | 5M | 100M |
| Default parallel envs | 16–64 | 42 per layout × 5 layouts = 210 |
| Progress monitoring | None during scan | `jax.debug.print` inside `jax.lax.scan` at configurable intervals |
| Resumable training | None | `--resume` flag with opt_state + update_idx saved in checkpoints |

## Evaluation

| | JaxMARL | Ours |
|---|---------|------|
| Eval method | Training episode rewards logged to wandb; 1 rollout saved as GIF | Dedicated eval with deterministic actions (argmax), 96 episodes per layout per pairing |
| Self-play eval | Training reward only | Separate eval function, per-layout breakdown |
| Cross-play eval | None | Ego vs population partners with H/M/L tiers per layout; ego-vs-ego matrix across all methods |
| Population eval | None | `evaluate_pop_per_layout.py` — evaluates all population checkpoints in self-play per layout to produce skill scores |
| Mid-training eval | None | `--num-evals` saves intermediate checkpoint params inside the scan for post-training evaluation at multiple training points |
| Eval scripts | None | `evaluate.py` (self-play, cross-play), `evaluate_pop_per_layout.py`, `eval_ego_vs_ego.py`, `eval_fcp_pixel_full.py` |

## File Structure

| | JaxMARL | Ours |
|---|---------|------|
| Training | 1 file per env variant (`ippo_ff_overcooked.py`, `ippo_cnn_overcooked.py`) | 1 file per ZSC method (`train_sp.py`, `train_sp_population.py`, `train_mep.py`, `train_bc.py`, `train_bcp.py`, `train_fcp.py`) |
| Shared utilities | None (each script is self-contained) | `ppo.py`, `obs_encoding.py`, `reward_shaping.py`, `nxplay.py`, `wandb_utils.py` |
| Environment | `jaxmarl/environments/overcooked/` (2-agent) | `overcooked_env/` (forked, extended to N agents) |
| Evaluation | None | 4 dedicated eval scripts |
| Reports | None | Numbered per-run reports with tables, surveys, and raw JSON results |

---

## Summary

JaxMARL provides a clean IPPO self-play baseline for Overcooked. Our repo extends it into a full ZSC research platform: diverse population training, multiple coordination methods (FCP, MEP, BCP), pixel observations for sim-to-real transfer, and comprehensive cross-play evaluation with per-layout H/M/L partner tiers.
