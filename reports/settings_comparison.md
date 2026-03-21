# Settings Comparison: Run 1 vs Run 2 vs Run 3

| Parameter | Run 1 | Run 2 | Run 3 | Change (Run 2 -> 3) |
|-----------|-------|-------|-------|---------------------|
| **Network** | 3x64 (180K params) | 2x256 (400K params) | 2x256 (400K params) | Same |
| **PPO n_steps** | 128 | 500 | 500 | Same |
| **n_envs** | 40 (8x5) | 210 (42x5) | 210 (42x5) | Same |
| **Steps/PPO update** | 5,120 | 105,000 | 105,000 | Same |
| **Total timesteps** | 60-120M (varied) | 100M (all methods) | 100M (all methods) | Same |
| **Reward mode** | Carroll (sparse+shaped) | Originaler (ratio blend) | Originaler (ratio blend) | Same |
| **Position shuffle** | None (fixed) | Slot shuffle each update | Slot shuffle each update | Same |
| **MEP epoch_updates** | 20 | 1 | 1 | Same |
| **MEP ego sampling** | Prioritized (alpha=3.0) | Prioritized (alpha=3.0) | **Eval-based per-layout per-position** | New in Run 3 |
| **MEP Phase 1 ckpts** | Standard | Standard | **3 per agent (init/mid/final)** | New in Run 3 |
| **SP Checkpoints** | 15 | 35 | 35 | Same |
| **FCP pool** | 560 (all ckpts) | 48 (global H/M/L) | **Per-layout H/M/L** | New in Run 3 |
| **SP Pop eval** | — | — | **evaluate_pop_per_layout.py (Step 0)** | New in Run 3 |
| PPO lr | 3e-4 | 3e-4 | 3e-4 | Same |
| PPO gamma/gae/clip/vf/ent | 0.99/0.95/0.2/0.3/0.01 | Same | Same | Same |
| Pop size | 16 | 16 | 16 | Same |
| BC architecture | 2x64 | 2x64 | 2x64 | Same |
| BCP anneal | 8K steps | 8K steps | 8K steps | Same |
| Eval (episodes/horizon) | 100/400 | 100/400 | 100/400 | Same |

## Results Comparison

### Ego Self-Play (mean across layouts)

| Method | Run 1 | Run 2 | Run 3 | Delta (R2->R3) |
|--------|-------|-------|-------|----------------|
| SP | 198.6 | 272.0 | 272.0 | 0.0 |
| MEP-16 | 119.6 | **179.1** | 166.0 | -13.1 |
| MEP-8 | 130.8 | 152.6 | **197.7** | +45.1 |
| BCP | 142.7 | 53.8 | **78.4** | +24.6 |
| FCP | 115.4 | **149.7** | 95.2 | -54.5 |

### Cross-Play Overall (mean sparse reward)

| Method | Run 1 | Run 2 | Run 3 | Delta (R2->R3) |
|--------|-------|-------|-------|----------------|
| SP | 42.1 | 95.2 | 94.6 | -0.6 |
| MEP-16 | 57.7 | **126.6** | 121.6 | -5.0 |
| MEP-8 | 56.6 | 110.3 | **133.2** | +22.9 |
| BCP | 79.1 | **91.6** | 71.7 | -19.9 |
| FCP | 63.8 | 123.7 | 121.7 | -2.0 |

### Key Takeaways

1. **MEP-8 now leads cross-play (133.2)** — eval-based sampling is a big win for the smaller population (+22.9)
2. **MEP-8 self-play also jumped to 197.7** (+45.1) — improvements in both metrics simultaneously
3. **FCP cross-play held steady (~122)** despite per-layout H/M/L, but self-play dropped due to catastrophic seed 2
4. **MEP-16 slightly declined in both metrics** — seed 0 collapse to 104.0 suggests instability
5. **BCP regressed in cross-play (71.7)** — unchanged method, likely noise or evaluation interaction
6. **vs BC partners remains near-zero** for all RL methods — persistent challenge across all runs
7. **High variance is the main concern** — FCP seed 2 (14.6 SP) and MEP-16 seed 0 (104.0 SP) are outliers
