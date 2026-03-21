# Run 4 Report — Full grid observation + CNN (pixel-based input)

## Changes from Run 3

| Component | Run 3 | Run 4 |
|-----------|-------|-------|
| **Obs encoding** | Egocentric 7x7x26 = 1274-dim | Full grid 26x5x9 = 1170-dim |
| **Network** | 2x256 MLP | CNNActorCritic |
| **Total timesteps** | 100M | 10M |
| **SP Pop size** | 16 | 8 |
| **Seeds** | 3 | 2-3 |

## Settings

| Parameter | Value |
|-----------|-------|
| Network | **CNNActorCritic** |
| Obs encoding | **Full grid** (`--obs-encoding full`, 26x5x9 = 1170-dim) |
| Total timesteps | **10M** (10% of Runs 1-3) |
| SP Pop size | **8** |
| Seeds (training) | 2-3 depending on method |

All other PPO hyperparameters unchanged from Run 3.

## Training Results

| Method | Seeds | Duration | Final Rewards |
|--------|-------|----------|---------------|
| SP Pop | 3 | 68 min | Mean ~290 |
| MEP-8 Pop | 3 | 12 min | 87-127 |
| MEP-8 ego | 2 | 14 min | 172 / 152 |
| FCP | 2 | 6.5 min | 213 / 187 |

## Results

### Table 1: Ego Self-Play (5-layout mean)

| Method | Mean |
|--------|------|
| SP Pop | **~290** |
| MEP-8 ego | **~2.5** |
| FCP | **~53** |

### Table 2: Cross-Play Summary (mean sparse reward)

| Method | Overall |
|--------|---------|
| MEP-8 | **~35** |
| FCP | **~45** |

## Comparison to Run 3 (MLP, 100M)

### Ego Self-Play

| Method | Run 3 (MLP, 100M) | Run 4 (CNN, 10M) | Delta |
|--------|-------------------|-------------------|-------|
| MEP-8 ego | 197.7 | ~2.5 | -195.2 |
| FCP | 95.2 | ~53 | -42.2 |

### Cross-Play

| Method | Run 3 (MLP, 100M) | Run 4 (CNN, 10M) | Delta |
|--------|-------------------|-------------------|-------|
| MEP-8 | 133.2 | ~35 | -98.2 |
| FCP | 121.7 | ~45 | -76.7 |

## Key Observations

1. **CNN drastically underperforms MLP** even accounting for 10x fewer timesteps. MEP-8 self-play dropped from 197.7 to ~2.5 (nearly 100x worse).
2. **SP Pop reward is misleadingly high (~290)** because population self-play doesn't require cross-play generalization. The CNN learns layout-specific spatial patterns within a fixed population.
3. **MEP-8 population rewards (87-127) are much lower than MLP's ~260** indicating the CNN struggles to learn cooperative strategies during population training.
4. **FCP holds up best (~53 self-play, ~45 cross-play)** likely because it trains against a diverse partner pool which provides more consistent learning signal.
5. **Cross-play is severely degraded**: MEP-8 dropped from 133.2 to ~35, FCP from 121.7 to ~45.

## Why CNN Underperforms

1. **Insufficient training (10M vs 100M timesteps)**: CNNs need more data to learn spatial features. 10M timesteps is far too few for the CNN to converge.
2. **Full grid observation lacks inductive bias**: The egocentric 7x7 crop provides agent-centering and rotation-invariance — powerful priors for Overcooked. The full grid (26x5x9) has no such structure.
3. **No data augmentation**: DrQ-style augmentation (random shifts, crops) is standard for pixel-based RL but was not used here. This would help the CNN generalize spatial features.
4. **Egocentric encoding is very efficient**: 7x7 crop centered on the agent naturally encodes relative positions. The full grid requires the CNN to learn agent-relative reasoning from scratch.

## Files

- Checkpoints: `checkpoints/`
- Eval results: `reports/eval_results/`
