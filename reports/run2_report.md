# Run 2 Report — 2-Player Overcooked (multiHRI-matched)

## Settings

| Parameter | Value |
|-----------|-------|
| Network | **2×256 MLP** (pi=[256,256], vf=[256,256]) |
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
| n_envs_total | 210 (42×5 layouts) |
| Steps per PPO update | 105,000 (500×210) |
| SP Pop total | 100M |
| SP Ego total | 100M |
| MEP/FCP/BCP ego total | 100M |
| SP Pop size | 16 |
| SP Checkpoints | 35 per agent |
| MEP pop size | 16 (MEP-16) and 8 (MEP-8) |
| MEP epoch_updates | 1 (~105K steps/round) |
| MEP entropy coef | 0.01 |
| MEP ego sampling | Prioritized (alpha=3.0) |
| Reward mode | **Originaler**: `group_sparse×ratio + shaped×(1-ratio)`, ratio: 0→0.5 over 10M steps |
| Starting positions | **Fixed layout + slot shuffle** each PPO update (multiHRI p_idx style) |
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
| SP Pop (seeds 3,4) | |
| MEP-16 Pop (seeds 3,4) | |
| MEP-8 Pop (seeds 3,4) | |
| BC (seeds 3,4) | |

### Table 2: Ego Self-Play

| Method | cramped_room | asymm_adv | coord_ring | forced_coord | counter_circuit | Mean |
|--------|-------------|-----------|------------|-------------|----------------|------|
| SP | | | | | | |
| MEP-16 | | | | | | |
| MEP-8 | | | | | | |
| BCP | | | | — | — | |
| FCP | | | | | | |

### Table 3: Cross-Play Summary

| Ego \ Partner | vs SP | vs MEP-16 | vs MEP-8 | vs BC | Overall |
|--------------|-------|-----------|----------|-------|---------|
| SP | | | | | |
| MEP-16 | | | | | |
| MEP-8 | | | | | |
| BCP | | | | | |
| FCP | | | | | |

## Key Observations

*(To be filled after training)*
