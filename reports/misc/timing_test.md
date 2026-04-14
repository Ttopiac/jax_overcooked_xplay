# Timing Test: FCP Ego with IMPALA Shallow CNN

Architecture: IMPALA shallow (2 conv layers, 185K params)
Obs: 8×8 tiles, palette-indexed (1 byte/pixel)
Ego: pixel CNN, Partners: egocentric MLP
Seeds: 6, envs/layout: 42, steps: 500

| Metric | Value |
|--------|-------|
| JIT compile | 23.3s |
| Scan time (9 updates) | 147.7s |
| Per update | 16.41s |
| Per agent per update | 2.735s |
| Est. 100M steps (6 seeds) | 4.3h |
| Est. per agent (100M) | 0.7h |
