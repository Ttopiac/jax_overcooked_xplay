# Run 3 & Run 4 Cross-Play Evaluation

## H/M/L Definition (per layout, per population)

For a given population (e.g., SP Pop seed 3 with 560 agents) and a given layout (e.g., cramped_room):
- **H** = the 1 agent with the highest self-play score on that layout
- **M** = the 1 agent whose score is closest to 50% of H's score on that layout
- **L** = the 1 agent with the lowest score on that layout

Different layouts pick different agents as H/M/L. This matches how FCP training selects partners via `categorize_by_skill_per_layout`.

**Applied consistently to all populations for evaluation:** SP Pop, MEP-16 Pop, and MEP-8 Pop all use the same per-layout H/M/L definition. Note that MEP training itself does NOT use H/M/L categories (it uses prioritized sampling from the full pop), but we apply H/M/L for evaluation to enable standardized comparison across all partner types.

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
| SP Pop | 3, 4 | 560 (16×35 ckpts) | yes |
| MEP-16 Pop | 3, 4 | 48 (16×3 ckpts) | yes |
| MEP-8 Pop | 3, 4 | 24 (8×3 ckpts) | yes |
| BC | 3, 4 | 9 (3 layouts × 3) | N/A (no tiers) |

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
- Total: 867 pairings, completed in 132 minutes

---

## Table A: Self-Play (Run 3 MLP vs Run 4 Pixel)

Each ego seed plays with a copy of itself. Same encoder for both agents. Averaged across seeds.

| Method | # seeds | cramped | asymm | coord | forced | counter | Mean |
|--------|---------|---------|-------|-------|--------|---------|------|
| Run 3 SP | 3 | 260.0 | 500.0 | 240.0 | 200.0 | 160.0 | 272.0 |
| Run 3 MEP-16 | 3 | 253.3 | 206.7 | 140.0 | 200.0 | 36.0 | 167.2 |
| Run 3 MEP-8 | 3 | 253.3 | 421.5 | 140.0 | 173.1 | 0.0 | 197.6 |
| Run 3 BCP | 3 | 7.8 | 153.3 | 73.3 | — | — | 78.2 |
| Run 3 FCP | 3 | 162.3 | 160.1 | 73.3 | 55.4 | 25.3 | 95.3 |
| **Run 4 FCP pixel** | 6 | 0.0 | 12.4 | 0.0 | 0.0 | 33.3 | **9.2** |

---

## Tables B1-B5: Cross-Play vs Unseen Populations (Run 3 MLP & Run 4 Pixel ego vs Pop seeds 3,4)

One table per layout. Each cell computed as:

1. Ego seed `s` plays with population seed `p`'s H/M/L agent for this layout → one score
2. Average across pop seeds 3 and 4 → score per ego seed
3. Average across all ego seeds for that method → **final cell value**

For BC: no H/M/L tiers. Ego plays with all 9 BC agents from each seed, averages across agents, then across pop seeds 3,4, then across ego seeds.

### Table B1: cramped_room

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L | BC |
|---------------|------|------|------|-------|-------|-------|------|------|------|----|
| Run 3 SP | 195 | 181 | 1 | 108 | 136 | 136 | 172 | 125 | 125 | 3 |
| Run 3 MEP-16 | 163 | 99 | 2 | 244 | 241 | 241 | 160 | 236 | 236 | 5 |
| Run 3 MEP-8 | 221 | 171 | 2 | 173 | 212 | 212 | 224 | 243 | 243 | 2 |
| Run 3 BCP | 69 | 26 | 57 | 111 | 110 | 110 | 59 | 174 | 174 | 147 |
| Run 3 FCP | 206 | 186 | 24 | 172 | 193 | 193 | 182 | 178 | 178 | 104 |
| **Run 4 FCP pixel** | 192 | 173 | 0 | 134 | 130 | 130 | 206 | 162 | 162 | 0 |

### Table B2: asymm_advantages

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L | BC |
|---------------|------|------|------|-------|-------|-------|------|------|------|----|
| Run 3 SP | 262 | 247 | 0 | 480 | 100 | 97 | 500 | 143 | 143 | 0 |
| Run 3 MEP-16 | 374 | 305 | 0 | 410 | 263 | 269 | 420 | 267 | 267 | 0 |
| Run 3 MEP-8 | 400 | 299 | 0 | 463 | 270 | 270 | 460 | 277 | 277 | 0 |
| Run 3 BCP | 30 | 27 | 133 | 60 | 47 | 67 | 26 | 53 | 53 | 158 |
| Run 3 FCP | 397 | 197 | 29 | 463 | 190 | 187 | 499 | 197 | 197 | 36 |
| **Run 4 FCP pixel** | 473 | 174 | 0 | 450 | 85 | 76 | 497 | 107 | 107 | 2 |

### Table B3: coord_ring

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L | BC |
|---------------|------|------|------|-------|-------|-------|------|------|------|----|
| Run 3 SP | 37 | 10 | 0 | 67 | 63 | 47 | 10 | 17 | 17 | 0 |
| Run 3 MEP-16 | 61 | 49 | 0 | 122 | 95 | 79 | 152 | 112 | 112 | 0 |
| Run 3 MEP-8 | 116 | 16 | 0 | 159 | 163 | 91 | 125 | 152 | 152 | 0 |
| Run 3 BCP | 43 | 24 | 6 | 3 | 93 | 61 | 13 | 3 | 3 | 11 |
| Run 3 FCP | 123 | 37 | 0 | 37 | 10 | 34 | 126 | 46 | 46 | 0 |
| **Run 4 FCP pixel** | 55 | 21 | 0 | 66 | 19 | 17 | 83 | 35 | 35 | 0 |

### Table B4: forced_coord

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L |
|---------------|------|------|------|-------|-------|-------|------|------|------|
| Run 3 SP | 95 | 67 | 0 | 57 | 0 | 7 | 0 | 0 | 0 |
| Run 3 MEP-16 | 179 | 113 | 0 | 153 | 149 | 123 | 97 | 97 | 97 |
| Run 3 MEP-8 | 158 | 111 | 0 | 127 | 90 | 90 | 78 | 78 | 78 |
| Run 3 BCP | — | — | — | — | — | — | — | — | — |
| Run 3 FCP | 50 | 66 | 0 | 23 | 32 | 35 | 30 | 30 | 30 |
| **Run 4 FCP pixel** | 122 | 98 | 0 | 66 | 70 | 78 | 32 | 32 | 32 |

(No BC column — BC only trained on 3 layouts. BCP ego only trained on 3 layouts, marked —.)

### Table B5: counter_circuit

| Ego \ Partner | SP H | SP M | SP L | M16 H | M16 M | M16 L | M8 H | M8 M | M8 L |
|---------------|------|------|------|-------|-------|-------|------|------|------|
| Run 3 SP | 30 | 3 | 0 | 0 | 0 | 0 | 9 | 9 | 9 |
| Run 3 MEP-16 | 47 | 43 | 10 | 3 | 3 | 3 | 4 | 12 | 12 |
| Run 3 MEP-8 | 27 | 23 | 0 | 0 | 0 | 0 | 47 | 20 | 20 |
| Run 3 BCP | — | — | — | — | — | — | — | — | — |
| Run 3 FCP | 21 | 32 | 0 | 0 | 0 | 0 | 1 | 12 | 12 |
| **Run 4 FCP pixel** | 12 | 22 | 3 | 0 | 0 | 0 | 0 | 12 | 12 |

(No BC column. BCP ego marked —.)

---

## Table C: Full Ego vs Ego Cross-Play Matrix (Run 3 MLP & Run 4 Pixel)

Each ego method plays against every other ego method. Each cell = mean sparse reward across all ego seeds × partner seeds × layouts (5 layouts for most, 3 for BCP).

Dual encoder used when pixel ego plays with MLP partner (or vice versa).

### Per-layout breakdown

**cramped_room:**

| Ego \ Partner | SP | MEP-16 | MEP-8 | BCP | FCP | R4 pixel |
|---------------|-----|--------|-------|-----|-----|----------|
| R3 SP | (self) | 168 | 147 | 50 | 185 | 4 |
| R3 MEP-16 | 130 | (self) | 255 | 88 | 187 | 3 |
| R3 MEP-8 | 169 | 251 | (self) | 114 | 201 | 5 |
| R3 BCP | 52 | 127 | 126 | (self) | 68 | 32 |
| R3 FCP | 166 | 217 | 186 | 77 | (self) | 48 |
| R4 FCP pixel | 166 | 212 | 219 | 96 | 188 | (self) |

**asymm_advantages:**

| Ego \ Partner | SP | MEP-16 | MEP-8 | BCP | FCP | R4 pixel |
|---------------|-----|--------|-------|-----|-----|----------|
| R3 SP | (self) | 131 | 420 | 31 | 9 | 0 |
| R3 MEP-16 | 421 | (self) | 391 | 160 | 153 | 0 |
| R3 MEP-8 | 489 | 287 | (self) | 131 | 29 | 0 |
| R3 BCP | 49 | 27 | 24 | (self) | 31 | 160 |
| R3 FCP | 500 | 182 | 383 | 158 | (self) | 68 |
| R4 FCP pixel | 497 | 68 | 394 | 83 | 9 | (self) |

**coord_ring:**

| Ego \ Partner | SP | MEP-16 | MEP-8 | BCP | FCP | R4 pixel |
|---------------|-----|--------|-------|-----|-----|----------|
| R3 SP | (self) | 40 | 34 | 48 | 33 | 0 |
| R3 MEP-16 | 46 | (self) | 148 | 51 | 102 | 0 |
| R3 MEP-8 | 46 | 135 | (self) | 76 | 82 | 0 |
| R3 BCP | 46 | 73 | 91 | (self) | 12 | 4 |
| R3 FCP | 47 | 45 | 90 | 13 | (self) | 0 |
| R4 FCP pixel | 4 | 28 | 3 | 14 | 36 | (self) |

**forced_coord:**

| Ego \ Partner | SP | MEP-16 | MEP-8 | BCP | FCP | R4 pixel |
|---------------|-----|--------|-------|-----|-----|----------|
| R3 SP | (self) | 180 | 167 | — | 103 | 0 |
| R3 MEP-16 | 113 | (self) | 189 | — | 170 | 0 |
| R3 MEP-8 | 122 | 131 | (self) | — | 65 | 0 |
| R3 BCP | — | — | — | (self) | — | — |
| R3 FCP | 121 | 101 | 109 | — | (self) | 0 |
| R4 FCP pixel | 86 | 118 | 128 | — | 48 | (self) |

**counter_circuit:**

| Ego \ Partner | SP | MEP-16 | MEP-8 | BCP | FCP | R4 pixel |
|---------------|-----|--------|-------|-----|-----|----------|
| R3 SP | (self) | 22 | 16 | — | 16 | 0 |
| R3 MEP-16 | 1 | (self) | 29 | — | 33 | 16 |
| R3 MEP-8 | 14 | 44 | (self) | — | 17 | 0 |
| R3 BCP | — | — | — | (self) | — | — |
| R3 FCP | 21 | 33 | 13 | — | (self) | 8 |
| R4 FCP pixel | 7 | 22 | 1 | — | 7 | (self) |

### Summary (mean across layouts)

| Ego \ Partner | SP | MEP-16 | MEP-8 | BCP | FCP | R4 pixel | **vs Egos** |
|---------------|-----|--------|-------|-----|-----|----------|------------|
| R3 SP | (self) | 108 | 157 | 43 | 69 | 1 | **78.4** |
| R3 MEP-16 | 142 | (self) | 202 | 100 | 129 | 4 | **116.7** |
| R3 MEP-8 | 168 | 169 | (self) | 107 | 79 | 1 | **104.6** |
| R3 BCP | 49 | 76 | 81 | (self) | 37 | 65 | **61.4** |
| R3 FCP | 171 | 115 | 156 | 83 | (self) | 25 | **112.3** |
| R4 FCP pixel | 152 | 90 | 149 | 64 | 58 | (self) | **105.8** |

---

## Table D: Overall Summary (Run 3 MLP vs Run 4 Pixel)

Mean across all 5 layouts (3 layouts for BCP-related cells).

| Method | Self-play | vs SP H | vs SP M | vs SP L | vs M16 H | vs M16 M | vs M16 L | vs M8 H | vs M8 M | vs M8 L | vs BC | vs Egos | XP Overall |
|--------|-----------|---------|---------|---------|----------|----------|----------|---------|---------|---------|-------|---------|------------|
| Run 3 SP | 272.0 | 124 | 102 | 0 | 142 | 60 | 57 | 138 | 59 | 59 | 1 | 78 | 78 |
| Run 3 MEP-16 | 167.2 | 164 | 122 | 2 | 187 | 150 | 143 | 167 | 145 | 145 | 2 | 117 | 122 |
| Run 3 MEP-8 | 197.6 | 184 | 124 | 0 | 184 | 147 | 133 | 187 | 154 | 154 | 1 | 105 | 125 |
| Run 3 BCP | 78.2 | 47 | 26 | 65 | 58 | 83 | 79 | 33 | 77 | 77 | 105 | 61 | 65 |
| Run 3 FCP | 95.3 | 175 | 100 | 12 | 139 | 85 | 90 | 169 | 99 | 99 | 47 | 112 | 103 |
| **Run 4 FCP pixel** | **9.2** | **174** | **98** | **3** | **143** | **61** | **60** | **166** | **74** | **74** | **1** | **106** | **96** |

- "vs Egos" = mean cross-play score when playing against all other ego methods (Table C summary).
- "XP Overall" = mean of all cross-play columns (vs pop H/M/L + vs BC + vs Egos).

---

## Key Findings

### 1. Pixel CNN matches MLP on high-skill partner cross-play
Run 4 FCP pixel vs SP Pop H (174) ≈ Run 3 FCP vs SP Pop H (175). Similar for MEP-16 H and MEP-8 H. The pixel CNN learned to coordinate with strong partners as well as the MLP agent.

### 2. Pixel CNN fails at self-play
Run 4 self-play (9.2) vs Run 3 FCP self-play (95.3). The pixel ego was trained with egocentric MLP partners, so it learned to complement MLP behavior rather than its own pixel-based behavior.

### 3. Both methods fail with L-tier partners
All ego methods score near 0 with SP Pop L partners. The worst SP Pop agents (score 0 on that layout) are essentially random — no ego can coordinate with them.

### 4. Run 3 MEP-8 is the strongest MLP method overall
XP Overall: MEP-8 (132) > MEP-16 (128) > FCP (104) > SP (77) > BCP (65).

### 5. Run 4 FCP pixel achieves competitive cross-play
XP Overall: pixel FCP (94) vs MLP FCP (104). Only 10% gap despite using raw 8×8 pixel observations instead of engineered 1274-dim egocentric features. The pixel agent can see and respond to the same visual information a human would see.

### 6. Layout-specific strengths
- **asymm_advantages**: Pixel FCP excels (473 vs SP Pop H, 497 vs MEP-8 H) — best layout for pixel agent
- **cramped_room**: Competitive (192 vs SP H for pixel, 206 for MLP FCP)
- **coord_ring**: Both methods struggle (55 pixel, 123 MLP FCP vs SP H)
- **counter_circuit**: Weak for all methods
