# Full Evaluation Results: Run 3 (MLP) vs Run 4 (Pixel CNN)

## H/M/L Definition (per layout, per population)

For a given population (e.g., SP Pop seed 3 with 560 agents) and a given layout (e.g., cramped_room):
- **H** = the 1 agent with the highest self-play score on that layout
- **M** = the 1 agent whose score is closest to 50% of H's score on that layout
- **L** = the 1 agent with the lowest score on that layout

Different layouts pick different agents as H/M/L. This matches exactly how FCP training selects partners via `categorize_by_skill_per_layout`. The same definition is applied to SP Pop, MEP-16 Pop, and MEP-8 Pop for consistent evaluation.

## Agents

### Run 3 Ego Agents (MLP, egocentric encoding, 2×256 hidden)

| Method | Seeds | Path |
|--------|-------|------|
| SP ego | 0, 1, 2 | `sp_n2_seed{s}.pkl` |
| MEP-16 ego | 0, 1, 2 | `mep16_seed{s}.pkl` |
| MEP-8 ego | 0, 1, 2 | `mep8_seed{s}.pkl` |
| BCP ego | 0, 1, 2 | `run3_backup/bcp_n2x1_hetero_seed{s}.pkl` |
| FCP ego | 0, 1, 2 | `run3_backup/fcp_n2x1_hetero_seed{s}.pkl` |

### Run 4 Ego Agents (Pixel CNN, 8×8 tiles palette-indexed, IMPALA shallow 185K params)

| Method | Seeds | Path |
|--------|-------|------|
| FCP pixel ego | 0, 1, 2, 3, 4, 5 | `fcp_n2x1_hetero_seed{s}.pkl` |

### Unseen Partner Populations (seeds 3, 4 ONLY)

| Pop | Seed | Total agents | layout_scores |
|-----|------|-------------|---------------|
| SP Pop | 3 | 560 (16×35 ckpts) | yes |
| SP Pop | 4 | 560 (16×35 ckpts) | yes |
| MEP-16 Pop | 3 | 48 (16×3 ckpts) | yes |
| MEP-16 Pop | 4 | 48 (16×3 ckpts) | yes |
| MEP-8 Pop | 3 | 24 (8×3 ckpts) | yes |
| MEP-8 Pop | 4 | 24 (8×3 ckpts) | yes |
| BC | 3 | 9 (3 layouts × 3) | N/A (no tiers) |
| BC | 4 | 9 (3 layouts × 3) | N/A (no tiers) |

### Dual Encoder Rules

| Ego type | Partner type | Ego encoder | Partner encoder |
|----------|-------------|-------------|-----------------|
| MLP ego | MLP pop/ego | egocentric | egocentric |
| MLP ego | BC | egocentric | egocentric |
| Pixel ego | MLP pop/ego | pixel (8×8 palette) | egocentric |
| Pixel ego | BC | pixel (8×8 palette) | egocentric |

### Evaluation Protocol

- 96 episodes per pairing per layout (3 batches × 32 parallel envs)
- Deterministic actions (argmax of policy logits)
- Metric: mean sparse reward (number of soups delivered × 20)

## Pairing Count

### Per ego seed, partner evals:

| Partner pop | Seed 3 | Seed 4 | Subtotal |
|-------------|--------|--------|----------|
| SP Pop (H, M, L) | 3 | 3 | 6 |
| MEP-16 Pop (H, M, L) | 3 | 3 | 6 |
| MEP-8 Pop (H, M, L) | 3 | 3 | 6 |
| **Pop total** | **9** | **9** | **18** |
| BC (all 9 agents) | 9 | 9 | 18 |
| **Grand total per ego seed** | **18** | **18** | **36** |

### Total evals:

| Section | Ego agents | × evals each | Total pairings |
|---------|-----------|-------------|----------------|
| Run 3 (5 methods × 3 seeds) | 15 | × 36 | 540 |
| Run 4 (1 method × 6 seeds) | 6 | × 36 | 216 |
| Ego-vs-ego (6 pixel × 15 Run3 egos) | — | — | 90 |
| Self-play | — | — | 21 |
| **Grand total** | | | **867** |

---

## Table A: Self-Play

Each ego seed plays with a copy of itself. Same encoder for both agents. Averaged across seeds.

| Method | # seeds | cramped | asymm | coord | forced | counter | Mean |
|--------|---------|---------|-------|-------|--------|---------|------|
| Run 3 SP | 3 | | | | | | |
| Run 3 MEP-16 | 3 | | | | | | |
| Run 3 MEP-8 | 3 | | | | | | |
| Run 3 BCP | 3 | | | | | | |
| Run 3 FCP | 3 | | | | | | |
| **Run 4 FCP pixel** | 6 | | | | | | |

---

## Tables B1-B5: Cross-Play vs Populations

One table per layout. Each cell computed as:

1. Ego seed `s` plays with population seed `p`'s H/M/L agent for this layout → one score
2. Average across pop seeds 3 and 4 → score per ego seed
3. Average across all ego seeds for that method → **final cell value**

For BC: no H/M/L tiers. Ego plays with all 9 BC agents from each seed, averages across agents, then across pop seeds 3,4, then across ego seeds.

### Table B1: cramped_room

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L | BC |
|---------------|------|------|------|-------|-------|-------|------|------|------|----|
| Run 3 SP (avg s0,1,2 × p3,4) | | | | | | | | | | |
| Run 3 MEP-16 (avg s0,1,2 × p3,4) | | | | | | | | | | |
| Run 3 MEP-8 (avg s0,1,2 × p3,4) | | | | | | | | | | |
| Run 3 BCP (avg s0,1,2 × p3,4) | | | | | | | | | | |
| Run 3 FCP (avg s0,1,2 × p3,4) | | | | | | | | | | |
| **Run 4 FCP pixel (avg s0-5 × p3,4)** | | | | | | | | | | |

### Table B2: asymm_advantages

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L | BC |
|---------------|------|------|------|-------|-------|-------|------|------|------|----|
| Run 3 SP | | | | | | | | | | |
| Run 3 MEP-16 | | | | | | | | | | |
| Run 3 MEP-8 | | | | | | | | | | |
| Run 3 BCP | | | | | | | | | | |
| Run 3 FCP | | | | | | | | | | |
| **Run 4 FCP pixel** | | | | | | | | | | |

### Table B3: coord_ring

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L | BC |
|---------------|------|------|------|-------|-------|-------|------|------|------|----|
| Run 3 SP | | | | | | | | | | |
| Run 3 MEP-16 | | | | | | | | | | |
| Run 3 MEP-8 | | | | | | | | | | |
| Run 3 BCP | | | | | | | | | | |
| Run 3 FCP | | | | | | | | | | |
| **Run 4 FCP pixel** | | | | | | | | | | |

### Table B4: forced_coord

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L |
|---------------|------|------|------|-------|-------|-------|------|------|------|
| Run 3 SP | | | | | | | | | |
| Run 3 MEP-16 | | | | | | | | | |
| Run 3 MEP-8 | | | | | | | | | |
| Run 3 BCP | — | — | — | — | — | — | — | — | — |
| Run 3 FCP | | | | | | | | | |
| **Run 4 FCP pixel** | | | | | | | | | |

(No BC column — BC only trained on 3 layouts. BCP ego also only trained on 3 layouts, marked —.)

### Table B5: counter_circuit

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L |
|---------------|------|------|------|-------|-------|-------|------|------|------|
| Run 3 SP | | | | | | | | | |
| Run 3 MEP-16 | | | | | | | | | |
| Run 3 MEP-8 | | | | | | | | | |
| Run 3 BCP | — | — | — | — | — | — | — | — | — |
| Run 3 FCP | | | | | | | | | |
| **Run 4 FCP pixel** | | | | | | | | | |

(No BC column. BCP ego marked —.)

---

## Table C: Cross-Play vs Run 3 Ego Agents

Pixel FCP ego (Run 4) plays against each Run 3 ego agent. Dual encoder: pixel for ego, egocentric for partner.

How each cell is computed:
1. Pixel ego seed `s` plays with Run 3 ego seed `p` → per-layout scores
2. Average across Run 3 partner seeds 0, 1, 2 → score per pixel seed per layout
3. Average across all 6 pixel seeds → **final cell value**

| Layout | vs SP ego | vs FCP ego | vs MEP-16 ego | vs MEP-8 ego | vs BCP ego |
|--------|-----------|------------|--------------|-------------|------------|
| cramped | | | | | |
| asymm | | | | | |
| coord | | | | | |
| forced | | | | | (—) |
| counter | | | | | (—) |
| **Mean** | | | | | |

BCP ego only trained on cramped, asymm, coord — forced and counter marked (—).

---

## Table D: Overall Summary

Mean across all 5 layouts (3 layouts for BCP-related cells).

| Method | Self-play | vs SP H | vs SP M | vs SP L | vs M16 H | vs M16 M | vs M16 L | vs M8 H | vs M8 M | vs M8 L | vs BC | vs Egos | XP Overall |
|--------|-----------|---------|---------|---------|----------|----------|----------|---------|---------|---------|-------|---------|------------|
| Run 3 SP | | | | | | | | | | | | — | |
| Run 3 MEP-16 | | | | | | | | | | | | — | |
| Run 3 MEP-8 | | | | | | | | | | | | — | |
| Run 3 BCP | | | | | | | | | | | | — | |
| Run 3 FCP | | | | | | | | | | | | — | |
| **Run 4 FCP pixel** | | | | | | | | | | | | | |

- "vs Egos" = pixel FCP ego vs Run 3 ego agents (Table C mean). Only applicable to Run 4.
- "XP Overall" = mean of all cross-play columns (vs pop H/M/L + vs BC + vs Egos where applicable).

---

## Results

*To be filled after evaluation completes.*
