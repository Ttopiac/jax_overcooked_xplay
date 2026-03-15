# Run 1 Report — 2-Player Overcooked

## Settings

| Parameter | Value |
|-----------|-------|
| Network | **3×64 MLP** (pi=[64,64,64], vf=[64,64,64]) |
| Network params | ~180K |
| PPO n_steps | 128 |
| PPO minibatch | 512 |
| PPO n_epochs | 4 |
| PPO lr | 3e-4 |
| PPO gamma | 0.99 |
| PPO gae_lambda | 0.95 |
| PPO clip_eps | 0.2 |
| PPO vf_coef | 0.3 |
| PPO ent_coef | 0.01 |
| n_envs_per_layout | 8 |
| n_envs_total | 40 (8×5 layouts) |
| Steps per PPO update | 5,120 (128×40) |
| SP Pop total | 60M |
| SP Ego total | 120M |
| MEP/FCP/BCP ego total | 60M |
| SP Pop size | 16 |
| SP Checkpoints | 15 per agent |
| MEP pop size | 16 (MEP-16) and 8 (MEP-8) |
| MEP epoch_updates | 20 (~102K steps/round) |
| MEP entropy coef | 0.01 |
| MEP ego sampling | Prioritized (alpha=3.0) |
| Reward mode | Carroll: `sparse + 1.0 * shaped` (no annealing) |
| Starting positions | **Fixed** (no shuffle, no randomization) |
| Obs encoding | Egocentric 7×7×26 = 1274-dim |
| BC architecture | 2×64 MLP |
| BC per layout | 3 agents |
| BCP bc_anneal_horizon | 8,000 env steps |
| BCP layouts | 3 (cramped_room, asymm_advantages, coord_ring) |
| Eval episodes | 100 per pair per layout |
| Eval horizon | 400 steps |
| Eval actions | Deterministic (argmax) |
| Seeds (training) | 0, 1, 2 |
| Seeds (eval partners) | 3, 4 |

## Results

### Table 1: Population Self-Play (mean sparse reward)

| Population | Mean |
|-----------|------|
| SP Pop (seeds 3,4) | 153.5 |
| MEP-16 Pop (seeds 3,4) | 119.0 |
| MEP-8 Pop (seeds 3,4) | 142.1 |
| BC (seeds 3,4) | 0.0 |

### Table 2: Ego Self-Play

| Method | cramped_room | asymm_adv | coord_ring | forced_coord | counter_circuit | Mean |
|--------|-------------|-----------|------------|-------------|----------------|------|
| SP | 226.7 | 315.8 | 198.2 | 180.1 | 71.9 | **198.6** |
| MEP-16 | 225.0 | 187.2 | 185.9 | 0.0 | 0.0 | **119.6** |
| MEP-8 | 219.2 | 246.7 | 187.7 | 0.6 | 0.0 | **130.8** |
| BCP | 28.2 | 286.7 | 113.3 | — | — | **142.7** |
| FCP | 84.9 | 202.5 | 193.3 | 91.0 | 5.2 | **115.4** |

### Table 3: Cross-Play Summary

| Ego \ Partner | vs SP | vs MEP-16 | vs MEP-8 | vs BC | Overall |
|--------------|-------|-----------|----------|-------|---------|
| SP | 44.5 | 63.7 | 60.3 | 0.0 | **42.1** |
| MEP-16 | 53.0 | 82.7 | 92.8 | 2.2 | **57.7** |
| MEP-8 | 48.3 | 82.3 | 95.5 | 0.2 | **56.6** |
| BCP | 69.4 | 77.4 | 86.4 | 83.1 | **79.1** |
| FCP | 52.2 | 77.2 | 110.1 | 15.9 | **63.8** |

## Key Observations

1. SP has highest self-play (198.6) but worst cross-play (42.1) — classic overfitting
2. BCP wins cross-play (79.1) — human-like BC partners help generalization
3. MEP agents fail on forced_coord and counter_circuit (0.0 self-play)
4. All methods except BCP score ~0 vs BC partners
5. Overall cross-play rewards are low compared to self-play
6. Network was too small (3×64) and rollouts too short (128 steps)
7. No position randomization — agents always started at same fixed positions
8. Carroll reward (constant sparse+shaped) instead of multiHRI's ratio blending
