# Timing Test: FCP Ego with IMPALA Shallow CNN

Architecture: IMPALA shallow (2 conv layers, 185K params)
Obs: 8×8 tiles, palette-indexed (1 byte/pixel)
Ego: pixel CNN, Partners: egocentric MLP
Seeds: 6, envs/layout: 42, steps: 500

## Results (1M timesteps, 9 updates)

| Metric | Value |
|--------|-------|
| JIT compile | ~23s |
| Scan time (9 updates, 6 seeds) | 147.7s |
| Per update | 16.41s |
| Per agent per update | 2.735s |
| Est. 100M steps (6 seeds) | 4.3h |
| Est. per agent (100M) | 0.7h |

## Architecture comparison (per-agent per-update)

| Architecture | Per-agent/update | Est. SP Pop (100M) |
|-------------|-----------------|-------------------|
| 4-block IMPALA ResNet (batch_size=10) | 32.88s | 696h |
| **IMPALA shallow (6 seeds vmap)** | **2.74s** | **4.3h (FCP only)** |

IMPALA shallow is **12× faster** per agent than the deep IMPALA.

## SP Pop pixel training estimate

Total CNN evaluations: 80 agents × 952 updates × 1.05M evals/update = 80 billion
At measured throughput (192K CNN-evals/sec): **~116 hours**

Population-based training does not scale with pixel CNN on single GPU.
