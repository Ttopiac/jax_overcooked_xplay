# Training Plan — JAX Overcooked XPlay

## Overview

Train and evaluate 5 zero-shot coordination methods in the Overcooked environment:
- **SP** — Self-Play
- **MEP-16** — Maximum Entropy Population (pop=16)
- **MEP-8** — Maximum Entropy Population (pop=8)
- **BCP** — Behavioral Cloning Play (Carroll et al. 2019)
- **FCP** — Fictitious Co-Play

Each method trains an **ego agent** that is then evaluated with **unseen partners**
at three skill levels (High, Medium, Low).

---

## Step 0: Train BC Population (~5 min per seed)

```bash
python train_bc.py --seed $SEED --bc-epochs 100 --num-bc-agents 3
```

| Detail | Value |
|--------|-------|
| Data | `2019_hh_trials_all.pickle` (human-human gameplay) |
| Layouts | cramped_room (45,602), asymmetric_advantages (45,586), coordination_ring (43,178) |
| Agents per layout | 3 (different seeds for diversity) |
| Architecture | **64×64 MLP** (matching Carroll) |
| Input | Egocentric obs (1274-dim) |
| Output | 6 actions, cross-entropy loss |
| lr / batch / epochs | 1e-3 / 64 / 100 (full dataset passes) |
| Early stopping | patience=20 |
| Validation split | 15% |
| Output | `checkpoints/bc_seed{SEED}.pkl` — 3 agents × 3 layouts = 9 BC models |

**Seeds**: 0, 1, 2 (training) + 3, 4 (evaluation teammates)

---

## Step 1: SP Population — shared by FCP (~72 min per seed)

```bash
python train_sp_population.py --seed $SEED --total-timesteps 60000000 --pop-size 16
```

| Detail | Value |
|--------|-------|
| Population size | 16 agents |
| Parallelism | All 16 via `jax.vmap` |
| Timesteps per agent | 60M |
| Layouts | All 5 simultaneously |
| Envs per layout | 8 (= 40 total) |
| Rollout steps | 128 |
| PPO updates | 11,718 per agent |
| Batch size | 128 × 40 × 2 = 10,240 |
| Checkpoints | 15 per agent (240 total) |
| Training mode | Self-play (both players share params) |
| Reward | `sparse + 1.0 * shaped` |
| Output | `checkpoints/sp_pop_seed{SEED}.pkl` |

**Seeds**: 0, 1, 2 (training) + 3, 4 (evaluation teammates)

---

## Step 2: SP (Self-Play) (~62 min per seed)

```bash
python train_sp.py --seed $SEED --total-timesteps 120000000
```

| Detail | Value |
|--------|-------|
| Agents | 1 (both players share params) |
| Timesteps | 120M |
| Layouts | All 5 |
| PPO updates | 23,437 |
| Batch size | 128 × 40 × 2 = 10,240 |
| Reward | `sparse + 1.0 * shaped` |
| Output | `checkpoints/sp_seed{SEED}.pkl` |

---

## Step 3a: MEP-16 (~62 min per seed)

```bash
python train_mep.py --seed $SEED --pop-timesteps 60000000 --ego-timesteps 60000000 \
    --pop-size 16 --epoch-updates 20
```

### Phase 1 — Population (60M)

| Detail | Value |
|--------|-------|
| Population | 16 agents |
| Agent selection | 1 random agent per round |
| Updates per round | 20 PPO updates (~102K env steps) |
| Total rounds | 585 |
| Avg updates per agent | ~731 |
| Pairing | **Self-play** (paired with itself) |
| Entropy bonus | `reward += 0.01 * (-log(P_pop(a)))` |
| P_pop | Mean action prob across **all 16 agents including self** |
| Reward | `sparse + shaped + entropy_bonus` |

### Phase 2 — Ego (60M)

| Detail | Value |
|--------|-------|
| Partner selection | **Prioritized sampling** (inverse performance ranking) |
| Priority formula | `priority = (1/rank)^3.0`, normalized |
| Performance tracking | EMA (alpha=0.1) of mean reward per partner |
| Entropy bonus | **None** (standard reward only) |
| PPO updates | 11,718 |
| Reward | `sparse + 1.0 * shaped` |
| Output | `checkpoints/mep16_seed{SEED}.pkl` |

---

## Step 3b: MEP-8 (~62 min per seed)

```bash
python train_mep.py --seed $SEED --pop-timesteps 60000000 --ego-timesteps 60000000 \
    --pop-size 8 --epoch-updates 20
```

Same as MEP-16 except:
- Population of 8 agents
- Each agent gets ~1,463 PPO updates on average (~2x more than MEP-16)
- Output: `checkpoints/mep8_seed{SEED}.pkl`

---

## Step 4: BCP (~31 min per seed)

```bash
python train_bcp.py --seed $SEED --total-timesteps 60000000 --bc-anneal-horizon 8000
```

| Detail | Value |
|--------|-------|
| BC source | `checkpoints/bc_seed{SEED}.pkl` |
| BC pool | 9 agents (3 layouts × 3 agents) |
| BC architecture | **64×64 MLP** (different from ego's 256×256) |
| Training layouts | **Only 3** with human data |
| Envs | 8/layout × 3 layouts = 24 |
| Timesteps | 60M |
| PPO updates | 19,531 |
| bc_schedule | bc_factor anneals 0→1 over **8,000 env steps** |
| Partner logic | Coin flip: BC (prob=bc_factor) or self-play |
| BC partner | Random from 9-agent pool |
| Reward | `sparse + 1.0 * shaped` |
| Output | `checkpoints/bcp_seed{SEED}.pkl` |

---

## Step 5: FCP (~31 min per seed)

```bash
python train_fcp.py --seed $SEED --total-timesteps 60000000
```

| Detail | Value |
|--------|-------|
| SP population | Loaded from `checkpoints/sp_pop_seed{SEED}.pkl` |
| Checkpoints | 240 (16 agents × 15 ckpts) → categorized: |
| HIGH tier | Top 1/3 (80 checkpoints) |
| MED tier | Middle 1/3 (80 checkpoints) |
| LOW tier | Bottom 1/3 (80 checkpoints) |
| Partner selection | **Uniform random tier** (33/33/33), then random within tier |
| Timesteps | 60M |
| Layouts | All 5 |
| PPO updates | 11,718 |
| Reward | `sparse + 1.0 * shaped` |
| Output | `checkpoints/fcp_seed{SEED}.pkl` |

---

## Per-Method Timestep Budget = 120M

| Method | Population phase | Ego phase | Total |
|--------|-----------------|-----------|-------|
| SP | — | 120M | 120M |
| MEP-16 | 60M (entropy pop) | 60M (ego vs pop) | 120M |
| MEP-8 | 60M (entropy pop) | 60M (ego vs pop) | 120M |
| BCP | ~0 (BC is supervised) | 60M (ego vs BC) | ~60M + BC |
| FCP | 60M (shared SP pop) | 60M (ego vs pool) | 120M |

---

## Evaluation Protocol

### Unseen Teammate Generation

To evaluate zero-shot coordination, each ego agent is paired with **unseen partners**
that were never encountered during training. We generate these partners using
**2 additional seeds** (seeds 3, 4) beyond the 3 training seeds (0, 1, 2).

#### Unseen BC teammates (BCP evaluation only)
```bash
python train_bc.py --seed 3 --bc-epochs 100 --num-bc-agents 3
python train_bc.py --seed 4 --bc-epochs 100 --num-bc-agents 3
```
- 3 agents/layout × 3 layouts × 2 seeds = **18 unseen BC agents**
- Combined with training seeds: 9 × 3 training seeds = 27, but only seeds 3,4 are unseen
- **6 unseen BC agents per layout** (3 from seed 3 + 3 from seed 4)

#### Unseen SP teammates (all methods)
```bash
python train_sp_population.py --seed 3 --total-timesteps 60000000 --pop-size 16
python train_sp_population.py --seed 4 --total-timesteps 60000000 --pop-size 16
```
- 16 agents × 2 seeds = **32 unseen SP agents**
- Each can play all 5 layouts (trained on all 5)
- Categorize into skill levels by final checkpoint reward:
  - **HIGH**: top 1/3 (~11 agents)
  - **MED**: middle 1/3 (~11 agents)
  - **LOW**: bottom 1/3 (~10 agents)

#### Unseen MEP teammates (all methods)
```bash
python train_mep.py --seed 3 --pop-timesteps 60000000 --ego-timesteps 0 --pop-size 16
python train_mep.py --seed 4 --pop-timesteps 60000000 --ego-timesteps 0 --pop-size 16
```
- 16 agents × 2 seeds = **32 unseen MEP population agents**
- Each can play all 5 layouts
- Categorize into HIGH/MED/LOW by self-play evaluation reward

### Evaluation Matrix

Each ego agent (from seeds 0, 1, 2) is paired with each unseen teammate and
evaluated over 100 episodes per layout. Report mean ± std across 3 ego seeds.

| Ego Method | Unseen Partners | Skill Levels | Layouts | Episodes |
|------------|----------------|--------------|---------|----------|
| SP | 32 SP + 32 MEP | H/M/L | 5 | 100 each |
| MEP-16 | 32 SP + 32 MEP | H/M/L | 5 | 100 each |
| MEP-8 | 32 SP + 32 MEP | H/M/L | 5 | 100 each |
| BCP | 6 BC/layout + 32 SP + 32 MEP | H/M/L | 3 (BC) or 5 (SP/MEP) | 100 each |
| FCP | 32 SP + 32 MEP | H/M/L | 5 | 100 each |

### Skill Level Categorization

For SP and MEP unseen partners, categorize by performance:
1. Evaluate each unseen agent in **self-play** across all layouts
2. Rank by mean episodic reward
3. Split into thirds: HIGH (best), MED (middle), LOW (worst)

For BC unseen partners:
- All BC agents are roughly similar skill (trained on same data)
- Report as a single "BC" skill level (human-like play)

### Metrics to Report

**Per ego method, per layout, report these 8 metrics:**

| Metric | Partners | Count |
|--------|----------|-------|
| vs SP-LOW | Bottom 1/3 of 32 SP agents | ~10 |
| vs SP-MED | Middle 1/3 of 32 SP agents | ~11 |
| vs SP-HIGH | Top 1/3 of 32 SP agents | ~11 |
| vs MEP-LOW | Bottom 1/3 of 32 MEP agents | ~10 |
| vs MEP-MED | Middle 1/3 of 32 MEP agents | ~11 |
| vs MEP-HIGH | Top 1/3 of 32 MEP agents | ~11 |
| vs BC | 6 BC agents per layout (3 layouts only) | 6 |
| **Overall** | Mean across all unseen partners | all |

Each cell = **mean episodic reward ± standard error** across:
- 3 ego seeds × N unseen partners × 100 episodes per pair per layout

Reward is **sparse only** (number of deliveries × 20) for clean comparison.

**Example result table (per layout):**

| Ego \ Partner | SP-L | SP-M | SP-H | MEP-L | MEP-M | MEP-H | BC | Overall |
|---|---|---|---|---|---|---|---|---|
| SP | | | | | | | | |
| MEP-16 | | | | | | | | |
| MEP-8 | | | | | | | | |
| BCP | | | | | | | | |
| FCP | | | | | | | | |

---

## Shared Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Obs encoding | Egocentric 7×7, 26 channels, 1274-dim | multiHRI |
| RL network | MLP 1274→256→256→6 (sep actor/critic) | multiHRI |
| BC network | MLP 1274→64→64→6 | Carroll |
| PPO lr | 3e-4 | multiHRI |
| PPO gamma | 0.99 | All papers |
| PPO gae_lambda | 0.95 | multiHRI |
| PPO clip_eps | 0.2 | Standard |
| PPO vf_coef | 0.3 | multiHRI |
| PPO ent_coef | 0.01 | multiHRI |
| PPO epochs | 4 | multiHRI |
| Envs per layout | 8 | — |
| Steps per rollout | 128 | — |
| Reward | `sparse + 1.0 * shaped` | Carroll default |
| Shaped rewards | +3 onion→pot, +5 soup pickup, +3 plate | JaxMARL |
| Sparse reward | +20 delivery | Both |
| MEP entropy coef | 0.01 | MEP paper |
| MEP epoch_updates | 20 (~102K steps/round) | ~multiHRI |
| MEP priority alpha | 3.0 | MEP paper |
| BCP bc_anneal_horizon | 8,000 env steps | Carroll |
| BC lr | 1e-3 | Carroll |
| BC batch/epochs | 64 / 100 | Carroll |
| BC early stopping | patience=20 | Carroll |

---

## Time Estimates

### Training (seeds 0, 1, 2)

| Step | Per seed | × 3 seeds |
|------|----------|-----------|
| BC | ~5 min | ~15 min |
| SP Pop | ~72 min | ~216 min |
| SP | ~62 min | ~186 min |
| MEP-16 | ~62 min | ~186 min |
| MEP-8 | ~62 min | ~186 min |
| BCP | ~31 min | ~93 min |
| FCP | ~31 min | ~93 min |
| **Total training** | **~325 min** | **~16 hrs** |

### Unseen Teammate Generation (seeds 3, 4)

| Step | Per seed | × 2 seeds |
|------|----------|-----------|
| BC (eval) | ~5 min | ~10 min |
| SP Pop (eval) | ~72 min | ~144 min |
| MEP Pop (eval) | ~31 min | ~62 min |
| **Total eval partners** | **~108 min** | **~3.6 hrs** |

### **Grand total: ~20 hours**
