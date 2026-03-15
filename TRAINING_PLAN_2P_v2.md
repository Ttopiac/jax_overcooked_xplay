# Training Plan v2 — 2-Player Games (multiHRI-matched)

## Parameter Comparison: multiHRI vs Ours

### Matched Parameters

| Parameter | multiHRI | Ours | Match |
|-----------|----------|------|-------|
| Network | 2×256 MLP (pi=[256,256], vf=[256,256]) | 2×256 MLP (shared ActorCritic) | YES |
| Network params | ~400K | ~400K | YES |
| PPO lr | 3e-4 | 3e-4 | YES |
| PPO gamma | 0.99 | 0.99 | YES |
| PPO gae_lambda | 0.95 | 0.95 | YES |
| PPO clip_eps | 0.2 (SB3 default) | 0.2 | YES |
| PPO vf_coef | 0.3 | 0.3 | YES |
| PPO ent_coef | 0.01 | 0.01 | YES |
| PPO n_epochs | 4 | 4 | YES |
| PPO n_steps | 500 | 500 | YES |
| PPO minibatch | 500 | 512 | ~YES |
| n_envs | 210 | 210 (42×5 layouts) | YES |
| Steps per PPO update | 105,000 (500×210) | 105,000 | YES |
| SP pop total | 100M (5M × how_long=20) | 100M | YES |
| FCP/MEP total | 100M | 100M | YES |
| SP ego total | 100M | 100M | YES |
| SP pop size | 8 (total_ego_agents) | 8 | YES |
| Checkpoints | 35 (num_of_ckpoints for 2-player) | 35 | YES |
| Eval horizon | 400 | 400 | YES |
| Reward mode | Originaler: group_sparse×ratio + shaped×(1-ratio) | Originaler mode | YES |
| Ratio schedule | 0 → 0.5 over 10M steps (dynamic_reward) | 0 → 0.5 over 10M steps | YES |
| final_sparse_r_ratio | 0.5 | 0.5 | YES |
| Position randomization | random p_idx each reset | shuffle_agent_slots each update | YES |
| MEP epoch_timesteps | 100K | 105K (500×210, 1 PPO update) | ~YES |
| MEP entropy coef | 0.01 | 0.01 | YES |

### Unmatched Parameters

| Parameter | multiHRI | Ours | Impact |
|-----------|----------|------|--------|
| Obs channels | 27 (overcooked_ai lossless) | 26 (JaxMARL) | LOW — missing "goal" channel, redundant with delivery location |
| Obs encoding source | overcooked_ai `lossless_state_encoding` | JaxMARL obs channels | LOW — same 26 core channels, different libraries |
| SB3 MultiInputPolicy | Uses CombinedExtractor (may add CNN for 3D obs) | No feature extractor, flat MLP | MEDIUM — SB3 might apply CNN to egocentric (7,7,27) obs. Unclear if it does for small spatial dims |
| Reward magnifier | Originaler returns `personal_reward` (no magnifier=3.0) | No magnifier | NONE — Originaler doesn't use magnifier |
| `group_sparse` calculation | `sum(env_info['sparse_r_by_agent'])` from overcooked_ai | `sum(rewards[k] for k in rewards)` from JaxMARL | LOW — both sum all agents' sparse rewards. JaxMARL gives same reward to both agents for delivery |
| Egocentric rotation | Original `OAI_egocentric` with overcooked_ai state | Our JAX implementation using JaxMARL state | LOW — same algorithm, different state source |
| Shaped rewards | overcooked_ai: +3 onion→pot, +0 dish pickup, +5 soup pickup | JaxMARL: +3 onion→pot, +3 plate pickup (conditional), +5 soup pickup | LOW — plate pickup reward differs (+0 vs +3) |
| Training framework | SB3 PPO (PyTorch) | Custom JAX PPO | LOW — same algorithm, different framework |
| Env auto-reset | SB3 VecEnv handles auto-reset with proper obs masking | JaxMARL auto-reset (simpler) | LOW — minor difference in terminal state handling |

### Parameters That Were Previously Wrong (now fixed)

| Parameter | Was | Now | multiHRI |
|-----------|-----|-----|----------|
| Network | 3×64 MLP (180K params) | 2×256 MLP (400K params) | 2×256 |
| n_steps | 128 | 500 | 500 |
| n_envs | 40 (8×5) | 210 (42×5) | 210 |
| Steps/update | 5,120 | 105,000 | 105,000 |
| Reward | sparse + shaped (Carroll) | Originaler ratio blending | Originaler |
| Pop total | 60M | 100M | 100M |
| Pop size | 16 | 8 | 8 |
| Checkpoints | 15 | 35 | 35 |
| Starting positions | Fixed, no shuffle | Fixed + slot shuffle | random p_idx |
| MEP epoch | 20 updates (~102K) | 1 update (~105K) | 1 learn() call (~100K) |

---

## Training Schedule

### Step 0: BC Training (~25 min, CPU)

```bash
for SEED in 0 1 2 3 4; do
    python train_bc.py --seed $SEED --bc-epochs 100 --num-bc-agents 3
done
```

- 2×64 MLP (Carroll architecture, unchanged)
- 3 agents × 3 layouts × 5 seeds = 45 BC models
- Output: `checkpoints/bc_seed{0..4}.pkl`

### Step 1: SP Population (5 seeds × 8 agents = 40 agents vmapped)

```bash
python train_sp_population.py --seeds 0,1,2,3,4 \
    --total-timesteps 100000000 --pop-size 8 --num-agents 2 --wandb
```

| Detail | Value |
|--------|-------|
| Total agents vmapped | 40 |
| Timesteps per agent | 100M |
| PPO updates per agent | 952 |
| Batch size | 500 × 210 × 2 = 210,000 |
| Checkpoints | 35 per agent = 280 total |
| Output | `checkpoints/sp_pop_seed{0..4}.pkl` |

**Time estimate**: 40 agents at ~0.6M steps/s throughput ≈ 40×100M/0.6M ≈ ~111 min

### Step 2: SP Ego (3 seeds vmapped)

```bash
python train_sp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --wandb
```

| Detail | Value |
|--------|-------|
| Agents vmapped | 3 |
| Timesteps | 100M |
| PPO updates | 952 |
| Output | `checkpoints/sp_n2_seed{0..2}.pkl` |

### Step 3: MEP-16 Pop + Ego

```bash
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 100000000 --ego-timesteps 100000000 \
    --pop-size 16 --epoch-updates 1 --num-agents 2 --num-unseen 1 --wandb
```

| Phase | Detail |
|-------|--------|
| Pop | 16 agents, 952 rounds, ~59 rounds/agent avg |
| Pop | Entropy bonus: `0.01 * (-log(P_pop(a)))` |
| Pop | Self-play pairing, transitions from both agents |
| Ego | Prioritized sampling from frozen pop (alpha=3.0) |
| Ego | 952 PPO updates |

### Step 4: MEP-8 Pop + Ego

Same as Step 3 but `--pop-size 8`. Each agent gets ~119 rounds avg.

### Step 5: BCP Ego

```bash
python train_bcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 \
    --bc-anneal-horizon 8000 --wandb
```

- 3 layouts only (cramped_room, asymm_advantages, coord_ring)
- bc_factor anneals 0→1 over 8,000 env steps
- Ego: 2×256 MLP, BC partner: 2×64 MLP

### Step 6: FCP Ego

```bash
python train_fcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 --wandb
```

- Pool: 280 checkpoints from SP pop (8 agents × 35 ckpts), categorized H/M/L
- Uniform random tier (33/33/33)

---

## Per-Method Budget = 100M timesteps each

| Method | Population | Ego | Total |
|--------|-----------|-----|-------|
| SP | — | 100M | 100M |
| MEP-16 | 100M (entropy pop) | 100M (vs pop) | 200M |
| MEP-8 | 100M (entropy pop) | 100M (vs pop) | 200M |
| BCP | ~0 (BC supervised) | 100M (vs BC) | ~100M |
| FCP | 100M (shared SP pop) | 100M (vs pool) | 200M |

Note: multiHRI's total per method is also ~200M for FCP (100M SP pop + 100M FCP ego).

---

## Evaluation

### Unseen Partners (from seeds 3,4)

| Type | Count | Source |
|------|-------|--------|
| SP | 8×2 = 16 agents | `sp_pop_seed{3,4}.pkl` |
| MEP-16 | 16×2 = 32 agents | `mep16_pop_seed{3,4}.pkl` |
| MEP-8 | 8×2 = 16 agents | `mep8_pop_seed{3,4}.pkl` |
| BC | 3×3×2 = 18 agents | `bc_seed{3,4}.pkl` |

### Evaluation Protocol

- 100 episodes per pair per layout
- 400 steps per episode
- Deterministic actions (argmax)
- Report sparse reward only (deliveries × 20)
- Mean ± SE across ego seeds × partners

### Result Tables

**Table 1**: Population & BC self-play (competence)
**Table 2**: Ego self-play (baseline coordination)
**Table 3**: Cross-play per skill level (main results)
**Table 4**: Overall summary

---

## Remaining Differences from multiHRI (cannot easily fix)

1. **SB3 CombinedExtractor**: SB3's MultiInputPolicy may apply a built-in CNN to the 3D egocentric observation `(7,7,27)` before the MLP. Our implementation uses a flat MLP directly on the flattened 1274-dim obs. If SB3 applies CNN, multiHRI's effective network is larger. **To investigate: check if SB3 NatureCNN works on 7×7 input.**

2. **27 vs 26 obs channels**: Our JaxMARL env produces 26 channels. multiHRI's overcooked_ai produces 27 (extra "goal" channel). The missing channel marks goal object locations, which are already covered by the delivery location channel (channel 15). **Impact: minimal.**

3. **Shaped reward difference**: JaxMARL gives +3 for conditional plate pickup. Original overcooked_ai gives +0 for dish pickup (disabled). This means our agents get slightly more shaped reward signal. **Impact: may help learning, slight positive bias.**

4. **Auto-reset behavior**: SB3's VecEnv properly handles terminal observations (returns the last obs before reset). JaxMARL returns the first obs of the new episode on terminal. **Impact: affects last-step value bootstrap, minor.**

5. **Training framework**: SB3 PPO vs our JAX PPO. Implementation differences in gradient computation, advantage normalization, etc. **Impact: should be equivalent but micro-differences exist.**
