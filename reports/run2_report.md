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
| Total timesteps (all methods) | 100M |
| PPO updates per training | 952 |
| SP Pop size | 16 |
| SP Checkpoints | 35 per agent |
| SP Pop GPU batch size | 8 (10 batches for 80 agents) |
| MEP pop size | 16 (MEP-16) and 8 (MEP-8) |
| MEP epoch_updates | 1 |
| MEP entropy coef | 0.01 |
| MEP ego sampling | Prioritized (alpha=3.0) |
| Reward mode | **Originaler**: `group_sparse * ratio + individual_shaped * (1-ratio)`, ratio 0→0.5 over 10M steps |
| Starting positions | **Slot shuffle** each PPO update (multiHRI-style) |
| Obs encoding | Egocentric 7×7×26 = 1274-dim |
| BC architecture | 2×64 MLP |
| BC per layout | 3 agents |
| BCP bc_anneal_horizon | 8,000 env steps |
| BCP layouts | 3 (cramped_room, asymm_advantages, coord_ring) |
| FCP partner pool | 48 per seed (16 H + 16 M + 16 L, subsampled from 560) |
| Eval episodes | 100 per pair per layout |
| Eval horizon | 400 steps |
| Eval actions | Deterministic (argmax) |
| Seeds (training) | 0, 1, 2 |
| Seeds (eval partners) | 3, 4 |
| Conda env | `jax_overcooked` (JAX 0.4.38, Python 3.12) |

## Results

### Table 1: Population Self-Play (mean sparse reward)

| Population | Mean |
|-----------|------|
| SP Pop (seeds 3,4) | 268.5 |
| MEP-16 Pop (seeds 3,4) | 189.5 |
| MEP-8 Pop (seeds 3,4) | 246.9 |
| BC (seeds 3,4) | 0.0 |

### Table 2: Ego Self-Play

| Method | cramped_room | asymm_adv | coord_ring | forced_coord | counter_circuit | Mean |
|--------|-------------|-----------|------------|-------------|----------------|------|
| SP | 260.0 | 500.0 | 240.0 | 200.0 | 160.0 | **272.0** |
| MEP-16 | 246.7 | 253.3 | 200.0 | 186.7 | 8.8 | **179.1** |
| MEP-8 | 223.7 | 195.6 | 116.9 | 186.7 | 40.0 | **152.6** |
| BCP | 26.6 | 73.3 | 61.4 | — | — | **53.8** |
| FCP | 238.9 | 211.0 | 143.4 | 115.1 | 40.0 | **149.7** |

### Table 3: Cross-Play Summary

| Ego \ Partner | vs SP | vs MEP-16 | vs MEP-8 | vs BC | Overall |
|--------------|-------|-----------|----------|-------|---------|
| SP | 162.9 | 56.7 | 160.2 | 0.8 | **95.2** |
| MEP-16 | 165.6 | 156.6 | 181.7 | 2.4 | **126.6** |
| MEP-8 | 127.5 | 137.1 | 176.5 | 0.0 | **110.3** |
| BCP | 93.6 | 94.1 | 82.8 | 95.9 | **91.6** |
| FCP | 177.8 | 114.6 | 180.4 | 22.1 | **123.7** |

## Key Observations

1. SP self-play dramatically improved (272 vs Run 1's 199) — larger network + more data per update
2. **MEP-16 leads cross-play (126.6)** — best zero-shot coordination overall
3. **FCP close second (123.7)** — strong cross-play with diverse SP population partners
4. SP cross-play improved 2× (95.2 vs 42.1) but still overfits (high self-play, mediocre cross-play)
5. MEP-16 counter_circuit still weak (8.8 self-play) — layout-specific challenge persists
6. BCP self-play dropped (53.8 vs Run 1's 142.7) — may need tuning with originaler reward
7. All methods except BCP still score ~0 vs BC partners — fundamental gap between RL and human-like play
8. MEP-8 pop outperforms MEP-16 pop in self-play (246.9 vs 189.5) — fewer agents = more training per agent
9. FCP partner pool was subsampled to 48 (from 560) due to GPU memory constraints

## Known Issues

1. **Cross-play JSON bug**: `evaluate.py` line 373 saves only the last partner's per-layout results, not the mean across all partners. The console output (`overall_mean`) is correct, but the saved JSON and therefore the Table 4 numbers from `print_tables.py` are based on the last partner only.
2. **Single partner per PPO update**: All 210 parallel envs use the same partner during training. multiHRI samples a different partner for each of the 210 envs per update. This reduces partner diversity during training and likely hurts cross-play performance. Fixed in Run 3.

## How to Resume

- **Retrain all**: `bash run_all.sh`
- **Retrain from SP Pop onward** (BC already done): `bash run_from_step1.sh`
- **Retrain FCP + re-evaluate**: `bash run_from_step6.sh`
- **Re-evaluate only**: Run the cross-play section of `run_from_step6.sh` (skip FCP training)

## Training Timeline

| Step | Duration |
|------|----------|
| BC Training | ~4 min |
| SP Population | ~10 hours |
| SP Ego | 24 min |
| MEP-16 (pop+ego) | ~60 min |
| MEP-8 (pop+ego) | ~60 min |
| BCP | 19 min |
| FCP | 18 min |
| Cross-play eval | ~3 hours |
| **Total** | **~18 hours** |

## Files

- Checkpoints: `checkpoints/` (32 files)
- Eval results: `reports/eval_results/` (141 JSON files)
- Training log: `reports/run2_log.txt`
