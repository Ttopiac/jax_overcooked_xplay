# Evaluation Tables — 2-Player Games

## Table 1: Self-Play Performance (Population & BC Agents)

Measures individual agent competence. Each agent plays with a copy of itself.
Mean sparse reward per episode (deliveries × 20) across 100 episodes.

### SP Population (seeds 3,4 — unseen eval partners)

| Agent | cramped_room | asymm_adv | coord_ring | forced_coord | counter_circuit | Mean |
|-------|-------------|-----------|------------|-------------|----------------|------|
| SP_s3_0 | | | | | | |
| SP_s3_1 | | | | | | |
| ... | | | | | | |
| SP_s3_15 | | | | | | |
| SP_s4_0 | | | | | | |
| ... | | | | | | |
| SP_s4_15 | | | | | | |
| **SP-HIGH (avg)** | | | | | | |
| **SP-MED (avg)** | | | | | | |
| **SP-LOW (avg)** | | | | | | |

### MEP-16 Population (seeds 3,4)

| Agent | cramped_room | asymm_adv | coord_ring | forced_coord | counter_circuit | Mean |
|-------|-------------|-----------|------------|-------------|----------------|------|
| MEP16_s3_0 | | | | | | |
| ... | | | | | | |
| **MEP16-HIGH (avg)** | | | | | | |
| **MEP16-MED (avg)** | | | | | | |
| **MEP16-LOW (avg)** | | | | | | |

### MEP-8 Population (seeds 3,4)

| Agent | cramped_room | asymm_adv | coord_ring | forced_coord | counter_circuit | Mean |
|-------|-------------|-----------|------------|-------------|----------------|------|
| **MEP8-HIGH (avg)** | | | | | | |
| **MEP8-MED (avg)** | | | | | | |
| **MEP8-LOW (avg)** | | | | | | |

### BC Agents (seeds 3,4)

| Agent | cramped_room | asymm_adv | coord_ring | Mean |
|-------|-------------|-----------|------------|------|
| BC_s3_cramped_0 | | | | |
| BC_s3_cramped_1 | | | | |
| BC_s3_cramped_2 | | | | |
| ... | | | | |
| **BC (avg)** | | | | |

---

## Table 2: Ego Self-Play Performance

Each ego agent (trained with seeds 0,1,2) plays with itself.
Validates that the ego agent learned basic coordination.

| Ego Method | cramped_room | asymm_adv | coord_ring | forced_coord | counter_circuit | Mean |
|-----------|-------------|-----------|------------|-------------|----------------|------|
| SP (avg ± se) | | | | | | |
| MEP-16 (avg ± se) | | | | | | |
| MEP-8 (avg ± se) | | | | | | |
| BCP (avg ± se) | | | | N/A | N/A | |
| FCP (avg ± se) | | | | | | |

---

## Table 3: Cross-Play Performance (Ego vs Unseen Partners)

Main evaluation. Each ego agent (3 seeds) paired with unseen partners (seeds 3,4).
Mean sparse reward ± SE across ego seeds × partners × 100 episodes per layout.

### vs SP Partners

| Ego \ Skill | SP-LOW | SP-MED | SP-HIGH | SP-All |
|------------|--------|--------|---------|--------|
| SP | | | | |
| MEP-16 | | | | |
| MEP-8 | | | | |
| BCP | | | | |
| FCP | | | | |

*(One table per layout, 5 total. BCP has N/A for forced_coord, counter_circuit)*

### vs MEP-16 Partners

| Ego \ Skill | MEP16-LOW | MEP16-MED | MEP16-HIGH | MEP16-All |
|------------|-----------|-----------|------------|-----------|
| SP | | | | |
| MEP-16 | | | | |
| MEP-8 | | | | |
| BCP | | | | |
| FCP | | | | |

### vs MEP-8 Partners

| Ego \ Skill | MEP8-LOW | MEP8-MED | MEP8-HIGH | MEP8-All |
|------------|----------|----------|-----------|----------|
| SP | | | | |
| MEP-16 | | | | |
| MEP-8 | | | | |
| BCP | | | | |
| FCP | | | | |

### vs BC Partners (3 layouts only)

| Ego \ Layout | cramped_room | asymm_adv | coord_ring | Mean |
|-------------|-------------|-----------|------------|------|
| SP | | | | |
| MEP-16 | | | | |
| MEP-8 | | | | |
| BCP | | | | |
| FCP | | | | |

---

## Table 4: Overall Summary

Mean sparse reward across all layouts, all unseen partner types, all skill levels.

| Ego Method | vs SP | vs MEP-16 | vs MEP-8 | vs BC | Overall |
|-----------|-------|-----------|----------|-------|---------|
| SP | | | | | |
| MEP-16 | | | | | |
| MEP-8 | | | | | |
| BCP | | | | | |
| FCP | | | | | |

---

## Evaluation Schedule

After each training step, run evaluation and fill in results:

| After Step | Evaluate | Fill Table |
|-----------|----------|------------|
| Step 0 (BC) | BC self-play | Table 1 (BC section) |
| Step 1 (SP Pop) | SP Pop self-play (seeds 3,4) | Table 1 (SP section) |
| Step 2 (SP Ego) | SP ego self-play | Table 2 (SP row) |
| Step 3 (MEP-16) | MEP-16 Pop self-play (seeds 3,4), MEP-16 ego self-play | Table 1 (MEP-16), Table 2 (MEP-16) |
| Step 4 (MEP-8) | MEP-8 Pop self-play (seeds 3,4), MEP-8 ego self-play | Table 1 (MEP-8), Table 2 (MEP-8) |
| Step 5 (BCP) | BCP ego self-play | Table 2 (BCP) |
| Step 6 (FCP) | FCP ego self-play | Table 2 (FCP) |
| **Final** | All cross-play evaluations | Tables 3, 4 |
