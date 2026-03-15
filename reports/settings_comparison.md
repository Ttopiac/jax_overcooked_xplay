# Settings Comparison: Run 1 vs multiHRI vs Run 2

| Parameter | Run 1 | multiHRI | Run 2 | Run 1→2 Change |
|-----------|-------|----------|-------|----------------|
| **Network** | 3×64 (180K) | 2×256 (400K) | 2×256 (400K) | Fixed ✓ |
| **PPO n_steps** | 128 | 500 | 500 | Fixed ✓ |
| **n_envs** | 40 (8×5) | 210 | 210 (42×5) | Fixed ✓ |
| **Steps/PPO update** | 5,120 | 105,000 | 105,000 | Fixed ✓ |
| **SP Pop total** | 60M | 100M | 100M | Fixed ✓ |
| **SP Ego total** | 120M | 100M | 100M | Fixed ✓ |
| **FCP/MEP ego total** | 60M | 100M | 100M | Fixed ✓ |
| **Pop size** | 16 | 8 | 16 | Keep 16 (our choice) |
| **Checkpoints** | 15 | 35 | 35 | Fixed ✓ |
| **Reward** | Carroll (sparse+shaped) | Originaler (ratio blend) | Originaler | Fixed ✓ |
| **Reward magnifier** | None | None (Originaler) | None | Same |
| **Ratio schedule** | No ratio | 0→0.5 over 10M | 0→0.5 over 10M | Fixed ✓ |
| **Position random** | None (fixed) | random p_idx | slot shuffle | Fixed ✓ |
| **MEP epoch** | 20 updates | 1 learn() call | 1 update | Fixed ✓ |
| PPO lr | 3e-4 | 3e-4 | 3e-4 | Same |
| PPO gamma | 0.99 | 0.99 | 0.99 | Same |
| PPO gae_lambda | 0.95 | 0.95 | 0.95 | Same |
| PPO clip_eps | 0.2 | 0.2 | 0.2 | Same |
| PPO vf_coef | 0.3 | 0.3 | 0.3 | Same |
| PPO ent_coef | 0.01 | 0.01 | 0.01 | Same |
| PPO epochs | 4 | 4 | 4 | Same |
| PPO minibatch | 512 | 500 | 512 | ~Same |
| MEP entropy coef | 0.01 | 0.01 | 0.01 | Same |
| MEP priority alpha | 3.0 | 3.0 | 3.0 | Same |
| BC architecture | 2×64 | 2×64 | 2×64 | Same |
| BCP bc_anneal | 8K steps | ~8K steps | 8K steps | Same |
| Eval horizon | 400 | 400 | 400 | Same |
| Obs encoding | ego 7×7×26 | ego 7×7×27 | ego 7×7×26 | See below |

## Remaining Differences (Run 2 vs multiHRI)

| # | Difference | Impact |
|---|-----------|--------|
| 1 | 26 vs 27 obs channels (missing "goal" channel) | **NONE** — goal channel is all zeros during training |
| 2 | SB3 feature extractor | **NONE** — confirmed SB3 uses Flatten, no CNN |
| 3 | Plate pickup shaped reward: +3 (JaxMARL) vs +0 (overcooked_ai) | **LOW** — slightly more shaped signal |
| 4 | Auto-reset terminal obs handling | **MINIMAL** — standard JAX approach, 0.25% of transitions |
| 5 | SB3 PyTorch vs JAX PPO | **LOW** — same algorithm, different framework |
| 6 | Pop size 16 vs multiHRI's 8 | **INTENTIONAL** — our choice for more diverse populations |
