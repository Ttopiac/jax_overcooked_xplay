# Training Plan — 2-Player Games

## Overview

Train and evaluate zero-shot coordination methods with N=2 agents, X=1 unseen teammate.
All methods use N-XPlay framework: 1 ego + 1 partner per game.

**Methods**: SP, MEP-16, MEP-8, BCP, FCP
**Seeds**: 0,1,2 (training ego agents) + 3,4 (unseen eval partners)
**Layouts**: 5 (cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit)
**Exception**: BCP trains/evaluates on 3 layouts only (cramped_room, asymm_advantages, coord_ring)

---

## Architecture

| Component | Size | Source |
|-----------|------|--------|
| RL network | 3×64 MLP (1274→64→64→64→6) | Carroll & MEP papers |
| BC network | 2×64 MLP (1274→64→64→6) | Carroll |
| Obs encoding | Egocentric 7×7×26 = 1274-dim | multiHRI |
| Reward | `sparse + 1.0 * shaped` (Carroll default) | Carroll |
| PPO | lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, vf=0.3, ent=0.01, 4 epochs | multiHRI |

---

## Execution Schedule (single RTX 3090)

### Step 0: BC Training (~25 min, CPU-bound)

```bash
for SEED in 0 1 2 3 4; do
    python train_bc.py --seed $SEED --bc-epochs 100 --num-bc-agents 3
done
```

| Detail | Value |
|--------|-------|
| Architecture | 2×64 MLP |
| Data | `2019_hh_trials_all.pickle` (3 layouts) |
| Per seed | 3 agents × 3 layouts = 9 BC models |
| Total | 5 seeds × 9 = **45 BC models** |
| Training | 100 epochs, lr=1e-3, batch=64, early stopping patience=20 |
| Output | `checkpoints/bc_seed{0..4}.pkl` |

### Step 1: SP Population (~129 min, GPU)

```bash
python train_sp_population.py --seeds 0,1,2,3,4 \
    --total-timesteps 60000000 --pop-size 16 --num-agents 2
```

| Detail | Value |
|--------|-------|
| Agents vmapped | 5 seeds × 16 agents = **80 total** |
| Timesteps per agent | 60M |
| Training | Self-play (X=0), all 5 layouts, 8 envs/layout |
| Checkpoints | 15 per agent = 1,200 total |
| Output | `checkpoints/sp_pop_seed{0..4}.pkl` |
| **Used by** | FCP (partner pool), evaluation (unseen SP partners) |

### Step 2: SP Ego (~62 min, GPU)

```bash
python train_sp.py --seeds 0,1,2 \
    --total-timesteps 120000000 --num-agents 2
```

| Detail | Value |
|--------|-------|
| Agents vmapped | 3 seeds |
| Timesteps | 120M per agent |
| Training | Self-play (X=0), all 5 layouts |
| Batch size | 128 steps × 40 envs × 2 agents = 10,240 |
| Output | `checkpoints/sp_n2_seed{0..2}.pkl` |

### Step 3: MEP-16 Pop + Ego (~80 min, GPU)

```bash
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 60000000 --ego-timesteps 60000000 \
    --pop-size 16 --epoch-updates 20 --num-agents 2 --num-unseen 1
```

| Detail | Value |
|--------|-------|
| **Phase 1 (Pop)** | |
| Seeds vmapped | 5 |
| Pop size | 16 per seed |
| Per round | Pick 1 random agent, train 20 PPO updates (~102K steps) |
| Entropy bonus | `reward += 0.01 * (-log(P_pop(a)))` including self |
| All N=2 agents | Self-play, transitions from both collected |
| **Phase 2 (Ego)** | |
| Seeds vmapped | 3 (ego_seeds 0,1,2) |
| Partner selection | Prioritized sampling (inverse perf, alpha=3.0) |
| N-X=1 ego + X=1 partner | Transitions from ego only |
| No entropy bonus | Standard reward |
| Output | `checkpoints/mep16_pop_seed{0..4}.pkl`, `checkpoints/mep16_seed{0..2}.pkl` |

### Step 4: MEP-8 Pop + Ego (~80 min, GPU)

```bash
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 60000000 --ego-timesteps 60000000 \
    --pop-size 8 --epoch-updates 20 --num-agents 2 --num-unseen 1
```

Same as Step 3 but pop_size=8. Each agent gets ~2x more training per round.

| Output | `checkpoints/mep8_pop_seed{0..4}.pkl`, `checkpoints/mep8_seed{0..2}.pkl` |

### Step 5: BCP Ego (~31 min, GPU)

```bash
python train_bcp.py --seeds 0,1,2 \
    --total-timesteps 60000000 --num-agents 2 --num-unseen 1 \
    --bc-anneal-horizon 8000
```

| Detail | Value |
|--------|-------|
| Seeds vmapped | 3 |
| Layouts | **3 only** (cramped_room, asymm_advantages, coord_ring) |
| BC pool | 9 agents per seed (3 layouts × 3 agents) |
| bc_schedule | Factor anneals 0→1 over 8,000 env steps |
| Partner | Coin flip: BC (64×64) or self-play (64×64×64), based on bc_factor |
| N-X=1 ego + X=1 partner | |
| Output | `checkpoints/bcp_n2x1_hetero_seed{0..2}.pkl` |

### Step 6: FCP Ego (~31 min, GPU)

```bash
python train_fcp.py --seeds 0,1,2 \
    --total-timesteps 60000000 --num-agents 2 --num-unseen 1
```

| Detail | Value |
|--------|-------|
| Seeds vmapped | 3 |
| SP pop source | `checkpoints/sp_pop_seed{0..2}.pkl` |
| Partner pool | 240 checkpoints per seed, categorized H/M/L |
| Partner selection | Uniform random tier (33/33/33), then random within tier |
| N-X=1 ego + X=1 partner | |
| Output | `checkpoints/fcp_n2x1_hetero_seed{0..2}.pkl` |

---

## Per-Method Timestep Budget = 120M

| Method | Population phase | Ego phase | Total |
|--------|-----------------|-----------|-------|
| SP | — | 120M | 120M |
| MEP-16 | 60M (entropy pop) | 60M (ego vs pop) | 120M |
| MEP-8 | 60M (entropy pop) | 60M (ego vs pop) | 120M |
| BCP | ~0 (BC supervised) | 60M (ego vs BC) | ~60M + BC |
| FCP | 60M (shared SP pop) | 60M (ego vs pool) | 120M |

---

## Time Estimates

| Step | Task | GPU time |
|------|------|----------|
| 0 | BC (5 seeds, CPU) | ~25 min |
| 1 | SP Pop (80 agents vmapped) | ~129 min |
| 2 | SP Ego (3 seeds vmapped) | ~62 min |
| 3 | MEP-16 Pop+Ego (5+3 seeds) | ~80 min |
| 4 | MEP-8 Pop+Ego (5+3 seeds) | ~80 min |
| 5 | BCP Ego (3 seeds) | ~31 min |
| 6 | FCP Ego (3 seeds) | ~31 min |
| **Total training** | | **~438 min ≈ 7.3 hrs** |

---

## Unseen Teammate Generation (for evaluation)

Already trained in Steps 0-4 using seeds 3,4:

| Partner type | Source | Count | Layouts |
|---|---|---|---|
| **SP** | `sp_pop_seed{3,4}.pkl` | 16×2 = **32 agents** | All 5 |
| **MEP-16** | `mep16_pop_seed{3,4}.pkl` | 16×2 = **32 agents** | All 5 |
| **MEP-8** | `mep8_pop_seed{3,4}.pkl` | 8×2 = **16 agents** | All 5 |
| **BC** | `bc_seed{3,4}.pkl` | 3×3×2 = **18 agents** (6/layout) | 3 layouts |

### Skill Level Categorization

For SP and MEP unseen partners:
1. Evaluate each agent in self-play across layouts
2. Rank by mean sparse reward
3. Split into thirds: HIGH, MED, LOW

---

## Evaluation Protocol

Each ego agent (seeds 0,1,2) paired with each unseen partner.
100 episodes per pair per layout. Sparse reward only (deliveries × 20).

### Result Table (per layout)

| Ego \ Partner | SP-L | SP-M | SP-H | MEP16-L | MEP16-M | MEP16-H | MEP8-L | MEP8-M | MEP8-H | BC | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SP | | | | | | | | | | | |
| MEP-16 | | | | | | | | | | | |
| MEP-8 | | | | | | | | | | | |
| BCP | | | | | | | | | | | |
| FCP | | | | | | | | | | | |

5 tables (1 per layout). BC column only for 3 layouts with human data.
Mean ± standard error across 3 ego seeds × N unseen partners × 100 episodes.

---

## run_all.sh

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"
export PATH="/home/nathan/anaconda3/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

echo "=== Step 0: BC Training ==="
for SEED in 0 1 2 3 4; do
    python train_bc.py --seed $SEED --bc-epochs 100 --num-bc-agents 3
done

echo "=== Step 1: SP Population (80 agents) ==="
python train_sp_population.py --seeds 0,1,2,3,4 \
    --total-timesteps 60000000 --pop-size 16 --num-agents 2 --wandb

echo "=== Step 2: SP Ego ==="
python train_sp.py --seeds 0,1,2 \
    --total-timesteps 120000000 --num-agents 2 --wandb

echo "=== Step 3: MEP-16 ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 60000000 --ego-timesteps 60000000 \
    --pop-size 16 --epoch-updates 20 --num-agents 2 --num-unseen 1 --wandb

echo "=== Step 4: MEP-8 ==="
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 60000000 --ego-timesteps 60000000 \
    --pop-size 8 --epoch-updates 20 --num-agents 2 --num-unseen 1 --wandb

echo "=== Step 5: BCP ==="
python train_bcp.py --seeds 0,1,2 \
    --total-timesteps 60000000 --num-agents 2 --num-unseen 1 \
    --bc-anneal-horizon 8000 --wandb

echo "=== Step 6: FCP ==="
python train_fcp.py --seeds 0,1,2 \
    --total-timesteps 60000000 --num-agents 2 --num-unseen 1 --wandb

echo "=== All training complete ==="
```

---

## Checkpoints Produced

| Checkpoint | Seeds | Contents |
|---|---|---|
| `bc_seed{0..4}.pkl` | 5 | 3 BC agents × 3 layouts |
| `sp_pop_seed{0..4}.pkl` | 5 | 16 SP agents + 15 ckpts each |
| `sp_n2_seed{0..2}.pkl` | 3 | 1 ego agent |
| `mep16_pop_seed{0..4}.pkl` | 5 | 16 MEP agents |
| `mep16_seed{0..2}.pkl` | 3 | 1 ego agent |
| `mep8_pop_seed{0..4}.pkl` | 5 | 8 MEP agents |
| `mep8_seed{0..2}.pkl` | 3 | 1 ego agent |
| `bcp_n2x1_hetero_seed{0..2}.pkl` | 3 | 1 ego agent |
| `fcp_n2x1_hetero_seed{0..2}.pkl` | 3 | 1 ego agent |

---

## Shared Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| RL network | 3×64 MLP | Carroll & MEP papers |
| BC network | 2×64 MLP | Carroll |
| Obs encoding | Egocentric 7×7×26 (1274-dim) | multiHRI |
| PPO lr | 3e-4 | multiHRI |
| PPO gamma | 0.99 | All |
| PPO gae_lambda | 0.95 | multiHRI |
| PPO clip_eps | 0.2 | Standard |
| PPO vf_coef | 0.3 | multiHRI |
| PPO ent_coef | 0.01 | multiHRI |
| PPO epochs | 4 | multiHRI |
| Envs per layout | 8 | — |
| Steps per rollout | 128 | — |
| Reward | `sparse + 1.0 * shaped` | Carroll |
| Shaped rewards | +3 onion→pot, +5 soup pickup, +3 plate pickup | JaxMARL |
| Sparse reward | +20 delivery | Both |
| MEP entropy coef | 0.01 | MEP paper |
| MEP epoch_updates | 20 (~102K steps/round) | ~multiHRI |
| MEP priority alpha | 3.0 | MEP paper |
| BCP bc_anneal_horizon | 8,000 env steps | Carroll |
| BC lr | 1e-3 | Carroll |
| BC batch/epochs | 64 / 100 | Carroll |
| BC early stopping | patience=20 | Carroll |
| N (num_agents) | 2 | — |
| X (num_unseen) | 1 | — |
| xplay_mode | hetero (= homo when X=1) | — |
