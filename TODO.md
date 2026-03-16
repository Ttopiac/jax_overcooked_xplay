# TODO

### Evaluation has not supported N-XPlay

**Current state:** `evaluate.py` is hardcoded for 2 agents:
- `evaluate_episode()` takes `params_a, params_b` (exactly 2)
- Observation encoding and action computation are written for agent_0 and agent_1 only
- Environment created with `num_agents=2` hardcoded
- Reward tracked from `rewards["agent_0"]` only (fine — all agents share the same sparse reward)

**How to fix:** Generalize `evaluate.py` to support N agents with X unseen partners:
1. Add `--num-agents` and `--num-unseen` CLI args
2. Change `evaluate_episode()` signature to accept `params_ego` + `params_partners` list instead of `params_a, params_b`
3. Replace hardcoded 2-agent obs/action loop with dynamic loop over `range(num_agents)`:
   - First `num_ego` agents use `params_ego`
   - Last `num_unseen` agents use corresponding `params_partners[j]`
4. Create env with `num_agents=num_agents` instead of `num_agents=2`
5. Update `evaluate_self_play()` and `evaluate_cross_play()` accordingly

**No changes needed in:**
- `overcooked_env/overcooked.py` — already supports N agents
- `obs_encoding.py` — `encoder.encode(obs, state, agent_idx, num_envs)` already works for any agent_idx
- `nxplay.py` — can reuse `select_partners()` and `shuffle_agent_positions()` utilities

### Add raw image observation mode

**Goal:** Let agents observe the full grid as a rendered image (like what a human sees), instead of the hand-crafted 26-channel egocentric encoding. This enables an experiment comparing whether engineered features actually help vs just seeing raw state.

**Current status:**
- We removed the `full` mode and kept only `egocentric` (7x7x26 = 1274-dim, MLP-compatible)
- We noted `raw` as a future option in `obs_encoding.py`

**What "raw" mode means:**
- Full grid observation as-is or rendered RGB from `maze_map`, no hand-crafted channels, no agent-centering
- Observation shape: (H, W, C) varying per layout (e.g., cramped_room 4x5, counter_circuit 5x8)
- Requires a CNN since spatial structure matters
- Naturally handles any N agents — no ego/teammate channel separation needed

**Literature survey (no prior ZSC work uses raw pixels in Overcooked):**

All major ZSC papers (FCP, MEP, TrajeDi, COLE, Other-Play, BRDiv, OvercookedV2) use the 26-channel boolean encoding + small CNN (32-64-32, 3x3 kernels). Nobody has tested raw pixel observations for Overcooked ZSC — this would be novel.

The closest analogue is **Melting Pot** (DeepMind), which is the main multi-agent benchmark using actual pixel observations with IMPALA CNN encoders.

**Recommended CNN architecture: IMPALA CNN or Impoola**

| Architecture | Details | Pros/Cons |
|---|---|---|
| Carroll 3-conv (32-64-32, 3x3) | Current Overcooked standard | Too shallow for raw pixels |
| Nature DQN (32@8x8s4, 64@4x4s2, 64@3x3s1) | Classic Atari | Designed for 84x84, strides too aggressive for small grids |
| **IMPALA CNN** (3 ConvSequence blocks {16,32,32}, each with MaxPool + 2 ResBlocks) | De facto standard for gridworld pixels (Procgen, Griddly) | Strong but fixed input size |
| **Impoola** (IMPALA + Global Average Pooling instead of flatten) | 2025, +17% generalization on Procgen, -35% params | **Handles variable grid sizes natively** — best fit |

**Recommendation:** Use **Impoola** (IMPALA CNN with GAP replacing flatten). GAP makes the encoder input-size-independent, so no padding needed across layouts. This is the cleanest solution for variable layout sizes.

**Implementation plan:**

1. `obs_encoding.py`: Add `raw` mode returning the full grid observation (from `maze_map` or env rendering), shape (H, W, C). No padding needed if using GAP.
2. `networks.py`: Add `ActorCriticCNN` using Impoola architecture:
   - 3 ConvSequence blocks: Conv2d(3x3) → MaxPool(3x3, stride 2) → 2 ResBlocks each
   - Channels: {16, 32, 32} (or {32, 64, 64} for wider variant)
   - Replace flatten with Global Average Pooling
   - FC head: 256 → action_dim (actor) + 1 (critic)
3. All training scripts: Add `--obs-encoding raw` flag that selects CNN network + raw encoder
4. Hyperparameter adjustments for pixel-based training:
   - Expect 5-10x more training steps than feature-based
   - Consider DrQ-style random shift augmentation for sample efficiency
   - Larger batch sizes (256-512 vs current)
   - Normalize pixel values (/255)
   - Consider LSTM layer if partial observability is an issue (as OvercookedV2 found necessary)

**Key references:**
- IMPALA CNN: Espeholt et al., 2018 (arxiv.org/abs/1802.01561)
- Impoola (GAP for RL): arxiv.org/abs/2503.05546
- DrQ augmentation: arxiv.org/abs/2004.13649
- Melting Pot (pixel MARL benchmark): github.com/google-deepmind/meltingpot
- OvercookedV2 (1x1 conv + LSTM addition): arxiv.org/abs/2503.17821
