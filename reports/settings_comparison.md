# Settings Comparison: Run 1 vs Run 2

| Parameter | Run 1 | Run 2 | Change |
|-----------|-------|-------|--------|
| **Network** | 3×64 (180K params) | 2×256 (400K params) | Matched multiHRI |
| **PPO n_steps** | 128 | 500 | Matched multiHRI |
| **n_envs** | 40 (8×5) | 210 (42×5) | Matched multiHRI |
| **Steps/PPO update** | 5,120 | 105,000 | 20× more data |
| **Total timesteps** | 60-120M (varied) | 100M (all methods) | Standardized |
| **Reward mode** | Carroll (sparse+shaped) | Originaler (ratio blend 0→0.5) | Matched multiHRI |
| **Position shuffle** | None (fixed) | Slot shuffle each update | Matched multiHRI |
| **MEP epoch_updates** | 20 | 1 | Matched multiHRI |
| **SP Checkpoints** | 15 | 35 | Matched multiHRI |
| **FCP pool size** | 560 (all ckpts) | 48 (subsampled H/M/L) | GPU memory constraint |
| PPO lr | 3e-4 | 3e-4 | Same |
| PPO gamma/gae/clip/vf/ent | 0.99/0.95/0.2/0.3/0.01 | Same | Same |
| Pop size | 16 | 16 | Same |
| BC architecture | 2×64 | 2×64 | Same |
| BCP anneal | 8K steps | 8K steps | Same |
| Eval (episodes/horizon) | 100/400 | 100/400 | Same |

## Results Comparison

### Ego Self-Play (mean across layouts)

| Method | Run 1 | Run 2 | Δ |
|--------|-------|-------|---|
| SP | 198.6 | **272.0** | +73.4 |
| MEP-16 | 119.6 | **179.1** | +59.5 |
| MEP-8 | 130.8 | **152.6** | +21.8 |
| BCP | 142.7 | 53.8 | -88.9 |
| FCP | 115.4 | **149.7** | +34.3 |

### Cross-Play Overall (mean sparse reward)

| Method | Run 1 | Run 2 | Δ |
|--------|-------|-------|---|
| SP | 42.1 | **95.2** | +53.1 |
| MEP-16 | 57.7 | **126.6** | +68.9 |
| MEP-8 | 56.6 | **110.3** | +53.7 |
| BCP | 79.1 | **91.6** | +12.5 |
| FCP | 63.8 | **123.7** | +59.9 |

### Key Takeaways

1. **All methods improved in cross-play** — multiHRI-matched params work
2. **MEP-16 now leads cross-play** (126.6) — was 3rd in Run 1
3. **BCP self-play regressed** — originaler reward may not suit BC-partner training
4. **SP cross-play doubled** — still overfits, but much stronger baseline
5. **vs BC partners remains near-zero** for all RL methods — persistent challenge
