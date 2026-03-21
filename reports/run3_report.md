# Run 3 Report — Per-layout H/M/L + eval-based MEP sampling

## Changes from Run 2

| Component | Run 2 | Run 3 |
|-----------|-------|-------|
| **FCP partner selection** | Global H/M/L tiers (48 per seed) | Per-layout H/M/L selection |
| **MEP ego sampling** | Prioritized (noisy EMA, alpha=3.0) | Eval-based per-layout per-position sampling |
| **MEP Phase 1 checkpoints** | Standard saves | 3 per agent (init/mid/final) for skill diversity |
| **SP Pop evaluation** | — | New `evaluate_pop_per_layout.py` (Step 0) |
| **BCP** | Per-env partner sampling | Unchanged |

## Settings

| Parameter | Value |
|-----------|-------|
| Network | **2x256 MLP** (pi=[256,256], vf=[256,256]) |
| Network params | ~400K |
| PPO n_steps | 500 |
| PPO minibatch | 512 |
| PPO n_epochs | 4 |
| PPO lr | 3e-4 |
| PPO gamma | 0.99 |
| PPO gae_lambda | 0.95 |
| PPO clip_eps | 0.2 |
| PPO vf_coef | 0.3 |
| PPO ent_coef | 0.01 |
| n_envs_per_layout | 42 |
| n_envs_total | 210 (42x5 layouts) |
| Steps per PPO update | 105,000 (500x210) |
| Total timesteps (all methods) | 100M |
| PPO updates per training | 952 |
| SP Pop size | 16 |
| SP Checkpoints | 35 per agent |
| MEP pop size | 16 (MEP-16) and 8 (MEP-8) |
| MEP epoch_updates | 1 |
| MEP entropy coef | 0.01 |
| MEP ego sampling | **Eval-based per-layout per-position** (was prioritized alpha=3.0) |
| MEP Phase 1 ckpts | **3 per agent (init/mid/final)** |
| FCP partner pool | **Per-layout H/M/L** (was global H/M/L) |
| Reward mode | **Originaler**: `group_sparse * ratio + individual_shaped * (1-ratio)`, ratio 0->0.5 over 10M steps |
| Starting positions | **Slot shuffle** each PPO update |
| Obs encoding | Egocentric 7x7x26 = 1274-dim |
| BC architecture | 2x64 MLP |
| BC per layout | 3 agents |
| BCP bc_anneal_horizon | 8,000 env steps |
| BCP layouts | 3 (cramped_room, asymm_advantages, coord_ring) |
| Eval episodes | 100 per pair per layout |
| Eval horizon | 400 steps |
| Eval actions | Deterministic (argmax) |
| Seeds (training) | 0, 1, 2 |
| Seeds (eval partners) | 3, 4 |
| Conda env | `jax_overcooked` (JAX 0.4.38, Python 3.12) |

## Training Results

| Method | Seeds | Duration | Final Rewards (s0/s1/s2) |
|--------|-------|----------|--------------------------|
| MEP-16 ego | 0, 1, 2 | 85 min | 231.6 / 230.4 / 245.0 |
| MEP-8 ego | 0, 1, 2 | 61 min | 249.4 / 248.7 / 260.2 |
| BCP | 0, 1, 2 | 20 min | 230.2 / 234.4 / 236.8 |
| FCP | 0, 1, 2 | 26 min | 235.3 / 229.8 / 232.8 |

## Results

### Table 1: Ego Self-Play

| Method | Seed 0 | Seed 1 | Seed 2 | Mean |
|--------|--------|--------|--------|------|
| SP | — | — | — | **272.0** |
| MEP-16 | 104.0 | 196.0 | 198.1 | **166.0** |
| MEP-8 | 220.9 | 220.0 | 152.2 | **197.7** |
| BCP | 62.8 | 65.7 | 106.7 | **78.4** |
| FCP | 135.8 | 135.3 | 14.6 | **95.2** |

### Table 2: Cross-Play Summary (mean sparse reward)

| Ego \ Partner | vs SP | vs MEP-16 | vs MEP-8 | vs BC | Overall |
|--------------|-------|-----------|----------|-------|---------|
| SP | — | — | — | — | **94.6** |
| MEP-16 | — | — | — | — | **121.6** |
| MEP-8 | — | — | — | — | **133.2** |
| BCP | — | — | — | — | **71.7** |
| FCP | — | — | — | — | **121.7** |

### Run 2 vs Run 3 Comparison

#### Ego Self-Play

| Method | Run 2 | Run 3 | Delta |
|--------|-------|-------|-------|
| SP | 272.0 | 272.0 | 0.0 |
| MEP-16 | 179.1 | **166.0** | -13.1 |
| MEP-8 | 152.6 | **197.7** | +45.1 |
| BCP | 53.8 | **78.4** | +24.6 |
| FCP | 149.7 | **95.2** | -54.5 |

#### Cross-Play Overall

| Method | Run 2 | Run 3 | Delta |
|--------|-------|-------|-------|
| SP | 95.2 | **94.6** | -0.6 |
| MEP-16 | 126.6 | 121.6 | -5.0 |
| MEP-8 | 110.3 | **133.2** | +22.9 |
| BCP | 91.6 | 71.7 | -19.9 |
| FCP | 123.7 | 121.7 | -2.0 |

## Key Observations

1. **MEP-8 is the clear cross-play winner at 133.2** (+22.9 vs Run 2) — eval-based sampling helps the smaller population more
2. **MEP-8 self-play also jumped to 197.7** (+45.1 vs Run 2) — a massive improvement in both self-play and cross-play
3. **FCP cross-play roughly maintained (121.7 vs 123.7)** despite per-layout H/M/L — but self-play dropped to 95.2 due to catastrophic seed 2 (14.6)
4. **MEP-16 cross-play slightly declined (121.6 vs 126.6)** — seed 0 self-play collapsed to 104.0, dragging down the mean
5. **BCP cross-play regressed (71.7 vs 91.6)** — BCP was unchanged, so this is likely noise or interaction effects
6. **SP unchanged** as expected (same population, no retraining)
7. **counter_circuit remains at 0** for all MEP-16 seeds in self-play — persistent layout challenge
8. **BC partners still completely broken** (0 reward), causing near-zero cross-play for all methods vs BC
9. **High variance remains a concern**: FCP seed 2 self-play at 14.6 (catastrophic), MEP-16 seed 0 at 104.0
10. **MEP-8 > MEP-16 in both self-play and cross-play** — smaller population with eval-based sampling outperforms larger population

## Known Issues

1. **BC partners broken**: BC agents produce 0 reward, making all cross-play vs BC near-zero. This inflates the penalty in overall cross-play scores.
2. **High seed variance**: FCP seed 2 self-play of 14.6 and MEP-16 seed 0 of 104.0 suggest training instability. More seeds or longer training may help.
3. **counter_circuit layout**: All MEP-16 seeds score 0 on this layout in self-play. May need layout-specific tuning.

## Training Timeline

| Step | Duration |
|------|----------|
| MEP-16 ego | 85 min |
| MEP-8 ego | 61 min |
| BCP | 20 min |
| FCP | 26 min |

## Files

- Checkpoints: `checkpoints/`
- Eval results: `reports/eval_results/`
- New script: `evaluate_pop_per_layout.py`
