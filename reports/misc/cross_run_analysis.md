# Cross-Run Analysis — Runs 1-4

## Overview

| Run | Date | Key Change | Network | Obs Encoding | Timesteps | Best Cross-Play |
|-----|------|-----------|---------|-------------|-----------|----------------|
| 1 | 2026-03-14 | Baseline (Carroll-style) | 3x64 MLP | Egocentric 7x7 | 60-120M | BCP (79.1) |
| 2 | 2026-03-16 | multiHRI-matched settings | 2x256 MLP | Egocentric 7x7 | 100M | MEP-16 (126.6) |
| 3 | 2026-03-20 | Per-layout H/M/L + eval-based MEP | 2x256 MLP | Egocentric 7x7 | 100M | MEP-8 (133.2) |
| 4 | 2026-03-21 | Full grid + CNN | CNNActorCritic | Full grid 26x5x9 | 10M | FCP (~45) |

## Self-Play Comparison (5-layout mean)

| Method | Run 1 (60M, 3x64 MLP) | Run 2 (100M, 2x256 MLP) | Run 3 (100M, 2x256 MLP) | Run 4 (10M, CNN) |
|--------|------------------------|--------------------------|--------------------------|------------------|
| SP Pop | 198.6 | 272.0 | 272.0 | ~290 (pop only) |
| MEP-16 ego | 119.6 | 179.1 | 166.0 | N/A |
| MEP-8 ego | 130.8 | 152.6 | 197.7 | ~2.5 |
| BCP | 142.7 | 53.8 | 78.4 | N/A |
| FCP | 115.4 | 149.7 | 95.2 | ~53 |

## Cross-Play Comparison (overall mean)

| Method | Run 1 | Run 2 | Run 3 | Run 4 (CNN) |
|--------|-------|-------|-------|-------------|
| SP | 42.1 | 95.2 | 94.6 | N/A |
| MEP-16 | 57.7 | 126.6 | 121.6 | N/A |
| MEP-8 | 56.6 | 110.3 | 133.2 | ~35 |
| BCP | 79.1 | 91.6 | 71.7 | N/A |
| FCP | 63.8 | 123.7 | 121.7 | ~45 |

## Key Findings

### 1. MEP-8 is the best cross-play method

MEP-8 cross-play progression: 56.6 -> 110.3 -> **133.2** (Runs 1-3). The combination of eval-based sampling (Run 3) and smaller population size gives MEP-8 the best zero-shot coordination. Smaller populations mean more training per agent, and eval-based sampling ensures the ego trains against its weakest partners.

### 2. Per-layout improvements (Run 3) helped FCP the most

FCP cross-play: 63.8 -> 123.7 -> 121.7. The big jump was Run 1 to Run 2 (+60 points) from the multiHRI-matched settings. Per-layout H/M/L selection in Run 3 maintained this level. FCP benefits from partner skill diversity (H/M/L tiers) but is sensitive to seed variance (Run 3 seed 2 collapsed to 14.6 self-play).

### 3. Eval-based MEP sampling improved MEP cross-play by ~23

MEP-8: 110.3 (Run 2) -> 133.2 (Run 3) = +22.9. The eval-based per-layout per-position sampling ensures the ego agent focuses training on partner-layout combinations where it performs worst, rather than relying on noisy EMA estimates.

### 4. MLP + egocentric encoding is far superior to CNN + full grid

Run 4 demonstrates that the CNN with full grid observation dramatically underperforms the MLP with egocentric encoding, even beyond what the 10x fewer timesteps would explain:
- MEP-8 self-play: 197.7 (MLP) vs ~2.5 (CNN) — nearly 100x worse
- MEP-8 cross-play: 133.2 (MLP) vs ~35 (CNN) — nearly 4x worse

The egocentric 7x7 crop provides agent-centering and rotation-invariance that the full grid lacks. These inductive biases are extremely valuable for Overcooked.

### 5. Population diversity (MEP) matters more than partner skill diversity (FCP H/M/L)

Both MEP-8 and FCP achieve similar cross-play in Run 3 (~133 vs ~122), but MEP achieves this through population diversity (agents trained against each other in a maximum-entropy framework) while FCP relies on curated partner tiers. MEP's approach is more robust as it doesn't depend on the quality of the SP population's skill spread.

### 6. All methods still struggle with hard layouts

Across all 4 runs, certain layouts consistently fail:
- **counter_circuit**: ~0 reward for MEP-16 ego across ALL runs
- **forced_coord**: Very low or zero for many methods, especially CNN

## Persistent Problems

### BC partners are broken

BC agents produce 0.0 reward across all 4 runs. This means all vs-BC cross-play scores are near-zero, which significantly drags down overall cross-play numbers. Every method's overall cross-play is penalized by approximately 25% due to this (since BC is one of four partner types in the cross-play matrix).

### counter_circuit consistently fails

This layout requires a specific asymmetric workflow: one agent loads onions while the other delivers. Without explicit asymmetric role discovery, both agents attempt the same actions and deadlock. MEP-16 scores 0 on this layout in self-play across all runs. Even SP Pop only achieves 160 (vs 500 on asymm_advantages).

### forced_coord is fragile

Agents must pass items through a narrow corridor. This requires precise timing and handoff coordination that is hard to learn from random exploration. Many method-seed combinations score 0 on this layout.

### High seed variance

Training is unstable across seeds:
- FCP seed 2 in Run 3: 14.6 self-play (catastrophic collapse)
- MEP-16 seed 0 in Run 3: 104.0 vs seeds 1,2 at ~197
- This variance makes it difficult to draw conclusions from 3-seed experiments

## Architecture and Encoding Analysis

| Property | Egocentric 7x7 + MLP | Full Grid + CNN |
|----------|----------------------|-----------------|
| Input dims | 1274 (7x7x26) | 1170 (26x5x9) |
| Agent-centered | Yes (always at center) | No (must learn) |
| Rotation-invariant | Yes (always facing up) | No (must learn) |
| Spatial locality | Implicit (nearby cells nearby in vector) | Must be learned by CNN |
| Params needed | ~400K (2x256 MLP) | More (CNN + FC layers) |
| Data efficiency | High | Low |
| Inductive bias | Strong (for Overcooked) | Weak |

The egocentric encoding essentially solves the "where am I?" problem for the agent, letting the MLP focus entirely on "what should I do?". The full grid CNN must first learn to locate itself, understand relative positions, and then learn behavior — a much harder problem.

## Progression of Best Cross-Play Method

| Run | Best Method | Score | Why |
|-----|------------|-------|-----|
| 1 | BCP | 79.1 | Small network, poor training — BC partners helped generalize |
| 2 | MEP-16 | 126.6 | Larger network + proper settings — population diversity wins |
| 3 | MEP-8 | 133.2 | Eval-based sampling — smaller population + smart sampling beats larger |
| 4 | FCP | ~45 | CNN too undertrained — FCP's diverse partners help more when agent is weak |

## Recommendations

1. **Stick with egocentric encoding + MLP** for Overcooked. The inductive bias is too valuable to give up.
2. **If CNN is desired**: Train for at least 100M timesteps, add DrQ-style data augmentation, and consider using the egocentric crop as CNN input instead of the full grid.
3. **Fix BC partners**: The broken BC agents are a systematic problem that penalizes all cross-play scores. Either retrain BC with better settings or remove from evaluation.
4. **Address layout-specific failures**: counter_circuit and forced_coord need targeted approaches — possibly asymmetric role training or curriculum learning.
5. **Increase seeds**: 3 seeds is insufficient given the high variance. Use 5+ seeds for more reliable comparisons.
6. **MEP-8 with eval-based sampling is the recommended method** for zero-shot coordination in Overcooked.
