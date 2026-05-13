# Run 4 Report — Human-Equivalent Pixel Observations + IMPALA Shallow CNN

## Goal

Agents see the same rendered kitchen image a human sees — preparing for future sim-to-real transfer in human-robot collaboration. Replace engineered egocentric features (1274-dim) with raw pixel observations.

## Changes from Run 3

| Component | Run 3 | Run 4 |
|-----------|-------|-------|
| **Obs encoding** | Egocentric 7×7×26 = 1274-dim | **Pixel 8×8 tiles, palette-indexed** (1, 40, 72) uint8 |
| **Network** | 2×256 MLP (367K params) | **IMPALA shallow CNN** (185K params) |
| **What was trained** | SP Pop, MEP-16 ego, MEP-8 ego, BCP, FCP | **FCP ego only** (6 seeds) |
| **Partners** | Egocentric MLP (same arch as ego) | **Dual encoder**: pixel CNN ego + egocentric MLP partners from Run 3's SP Pop |
| **Total timesteps** | 100M | 100M |
| **Seeds** | 3 per method | 6 (FCP only) |

## Architecture Journey

### Problem: pixel obs are 108× larger than egocentric

Each pixel observation is 138,240 floats (3×160×288 at 32px tiles) vs 1,274 for egocentric. Storing these in the `jax.lax.scan` trajectory overwhelms 24GB VRAM.

### Iteration 1: 4-block IMPALA ResNet (32×32 tiles)
- 20 conv layers, 1.6M params, input (3, 160, 288)
- **Result**: OOM at batch_size=1 with 2 envs/layout. Even with reduced settings, each PPO update took ~33s → SP Pop estimated at 696 hours. Abandoned.

### Iteration 2: Palette indexing
- Discovered sprites use only 76 unique RGB colors
- Store 1 byte per pixel (palette index) instead of 12 bytes (float32 RGB) → **12× memory savings**
- CNN decodes palette → RGB at forward-pass time via gather operation
- Enabled 16×16 tiles with batch_size=2, but still too slow (~116h for SP Pop)

### Iteration 3: Tile size reduction (16px → 8px)
- 8×8 tiles: (3, 40, 72) input, 4× fewer pixels than 16×16
- Palette-indexed: (1, 40, 72) uint8 = 2,880 bytes per obs
- batch_size=10 fits in VRAM

### Iteration 4: IMPALA shallow CNN (final architecture)
- Based on IMPALA paper (Espeholt et al., ICML 2018, Figure 3, shallow model)
- **Conv(16, 8×8, stride=4) → ReLU → Conv(32, 4×4, stride=2) → ReLU → Dense(256)**
- 185K params, 2 conv layers (vs 20 in the deep IMPALA)
- Handles free-position objects (no tile grid assumption) — suitable for sim-to-real
- Per-agent per-update: 2.7s (vs 33s with deep IMPALA) → FCP 6 seeds in 4.3 hours

### Why not deeper CNN?
- IMPALA deep (4-block ResNet) was designed for Atari/DMLab where objects move freely at multiple scales
- Overcooked is a simple tile grid with 64 possible tile types — multi-scale processing is unnecessary
- Procgen paper (Cobbe et al., ICML 2020) showed larger models help for diverse environments, but Overcooked has fixed layouts
- IMPALA shallow was the standard for single-task RL in the original paper

### Why not tile-stride CNN?
- Conv(32, kernel=8, stride=8) would exploit the tile grid structure for maximum speed
- Rejected because it encodes a tile-grid inductive bias that won't transfer to real camera images
- Real kitchen cameras don't produce tile grids — objects can be at any position

## Final Settings

| Parameter | Value |
|-----------|-------|
| Network | **IMPALACNNActorCritic** (IMPALA shallow, 185K params) |
| Obs encoding | **pixel** — 8×8 tile sprites, palette-indexed (76 colors) |
| Obs shape | (1, 40, 72) uint8 palette indices → decoded to (3, 40, 72) float32 RGB |
| Tile rendering | Pre-rendered `SpriteAtlas` using JaxMARL `OvercookedVisualizer` primitives |
| Dual encoder | Ego: pixel CNN. Partners: egocentric MLP (Run 3's SP Pop) |
| Total timesteps | 100M |
| FCP ego seeds | 0, 1, 2, 3, 4, 5 |
| Envs/layout | 42 |
| Num steps | 500 |
| batch_size | N/A (6 seeds vmapped, ~4.3h total) |
| All PPO hyperparams | Unchanged from Run 3 |

## Training Results

| Method | Seeds | Duration | Final Train Reward (shaped) | Final Sparse Reward |
|--------|-------|----------|---------------------------|-------------------|
| FCP pixel ego | 6 | 4.3h | ~206 | ~177 |

Training rewards improved steadily (shaped: 3→206, sparse: 27→177 over 952 updates). The CNN learned to coordinate with egocentric MLP partners.

## Why SP Pop / MEP Pop Were Not Trained

SP Pop pixel training estimated at ~116 hours on single RTX 3090 (80 agents × 952 updates × 2 game agents per step with CNN). Population-based methods don't scale with pixel CNN because every env step requires CNN forward passes for all agents.

FCP was chosen because:
1. Only trains 1 ego agent (not 80 population members)
2. Partners are frozen MLP agents (no CNN cost for partners via dual encoder)
3. Directly measures zero-shot coordination ability

## Evaluation Results

Full evaluation: 867 pairings across all Run 3 and Run 4 ego agents, H/M/L population tiers, BC, and ego-vs-ego cross-play. See `reports/05_run3_vs_run4_crossplay_eval/crossplay_eval.md` for complete tables.

### Self-Play

| Method | cramped | asymm | coord | forced | counter | Mean |
|--------|---------|-------|-------|--------|---------|------|
| R3 FCP (MLP) | 162 | 160 | 73 | 55 | 25 | **95.3** |
| **R4 FCP pixel** | 0 | 12 | 0 | 0 | 33 | **9.2** |

Self-play fails because the pixel ego trained with MLP partners — it learned to complement MLP behavior, not its own pixel-based behavior.

### Cross-Play vs Unseen Populations (mean across layouts)

| Method | vs SP H | vs SP M | vs SP L | vs M16 H | vs M8 H | vs BC | vs Egos | XP Overall |
|--------|---------|---------|---------|----------|---------|-------|---------|------------|
| R3 FCP (MLP) | 175 | 100 | 12 | 139 | 169 | 47 | 112 | 103 |
| **R4 FCP pixel** | **174** | **98** | **3** | **143** | **166** | **1** | **106** | **96** |

Pixel CNN matches MLP on high-skill partner cross-play (SP H: 174 vs 175, M16 H: 143 vs 139, M8 H: 166 vs 169).

### Key Findings

1. **Pixel CNN matches MLP on cross-play with strong partners** — the 8×8 pixel image carries enough information for coordination
2. **Self-play fails** (9.2 vs 95.3) — dual-encoder training mismatch, not a CNN capacity issue
3. **Training reward was high** (sparse 177) — the CNN learned, it just learned to work with MLP partners specifically
4. **Population training is the bottleneck** — pixel CNN per-step cost makes SP Pop impractical on single GPU
5. **IMPALA shallow is sufficient** — 2 conv layers handle 8×8 tile images; deeper architectures waste compute on multi-scale processing this domain doesn't need

## Files

- Training script: `run_fcp_pixel.sh`
- Eval scripts: `eval_fcp_pixel_full.py`, `eval_ego_vs_ego.py`
- Checkpoints: `checkpoints/fcp_n2x1_hetero_seed{0-5}.pkl` (pixel CNN)
- Full eval report: `reports/05_run3_vs_run4_crossplay_eval/crossplay_eval.md`
- Timing test: `reports/04_run4_cnn_pixel_human_equivalent/timing_test.md`
- Training log: `reports/04_run4_cnn_pixel_human_equivalent/fcp_log.txt`
