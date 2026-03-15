# Next Steps: Raw Image Observations & Harder ZSC

## Context

Our current setup uses a 26-channel egocentric encoding (7x7x26 = 1274-dim) fed to a 3-layer MLP. This is the same family of hand-crafted features used by all existing Overcooked ZSC papers (FCP, MEP, TrajeDi, COLE, Other-Play, BRDiv). No published work has tested raw image observations for ZSC in Overcooked.

Meanwhile, OvercookedV2 (Gessler et al., ICLR 2025) showed that standard Overcooked is not a meaningful ZSC benchmark — the SP-XP gap is mostly caused by poor state coverage, not genuine coordination challenges. State augmentation (training with diverse starting states from cross-play rollouts) nearly closes the gap. This means our current experiments on standard Overcooked may not be testing what we think they're testing.

This note outlines three directions: (1) raw image observation experiments on current Overcooked, (2) moving to harder ZSC scenarios where coordination actually matters, and (3) bridging ZSC to real robotics (dual Franka arms).

---

## Why Hand-Crafted Encoding Undermines ZSC Research

Beyond the state coverage problem identified by OvercookedV2, there is a deeper issue: **the 26-channel encoding itself makes ZSC results non-transferable to other domains.**

### The encoding is doing the hard work, not the method

The 26-channel boolean encoding tells the agent exactly what matters: pot states, onion locations, teammate positions and directions, counter contents, etc. This is hand-engineered domain knowledge baked into the observation. When a ZSC method "works" on this encoding, we can't tell whether:
- The **method** learned to coordinate, or
- The **encoding** made the perception problem trivial, and the agent just needed to see more states

If Method A beats Method B, the difference could be driven by how well each method exploits the encoding's structure, not by coordination ability.

### Non-transferability

Every new environment needs a completely new hand-crafted encoding. Overcooked's 26 channels don't apply to any other domain — not to Hanabi, not to robotics, not to any real deployment. A ZSC method validated only on hand-crafted features has no evidence of working when the agent must learn its own representation.

### The representation learning confound

State augmentation nearly closes the SP-XP gap (OvercookedV2, ICLR 2025). This may be *partly because* the 26-channel encoding gives agents such a complete, pre-digested view of the world that they just need to encounter more states to generalize. With raw pixels, representation learning itself could become a coordination bottleneck — independently trained agents might learn incompatible internal representations, creating a *genuinely harder* ZSC problem that state augmentation alone cannot fix.

### Implication

Raw image observations are not just "can we match performance without features" — they are essential for making ZSC research meaningful and transferable. A ZSC method that works from pixels is a result that generalizes to robotics, new games, and real deployment. A method that only works on hand-crafted channels is domain-locked.

---

## Part 1: Raw Image Observation Experiments

### Goal

Compare whether hand-crafted 26-channel egocentric encoding helps vs. raw grid observation. This is novel — nobody has done this for Overcooked ZSC.

### Observation Format

- **Current (egocentric):** 7x7x26 boolean channels, agent-centered and rotated so agent faces south. MLP-compatible (1274-dim flat vector).
- **Raw:** Full grid as spatial array, shape (H, W, C) varying per layout. No hand-crafted channels, no agent-centering. Requires CNN.

### Recommended CNN: Impoola (IMPALA + Global Average Pooling)

| Architecture | Source | Details | Why / Why Not |
|---|---|---|---|
| Carroll 3-conv (32-64-32, 3x3) | Carroll et al., 2020 | Standard Overcooked CNN for 26-ch encoding | Too shallow for raw pixels |
| Nature DQN CNN | Mnih et al., 2015 | 32@8x8s4, 64@4x4s2, 64@3x3s1 | Strides too aggressive for small grids (4x5 to 5x9) |
| IMPALA CNN | Espeholt et al., 2018 | 3 ConvSequence blocks {16,32,32}, each with MaxPool + 2 ResBlocks, then flatten -> FC(256) | De facto standard for gridworld pixels (Procgen, Griddly). Fixed input size. |
| **Impoola** | 2025 (arxiv 2503.05546) | IMPALA CNN but replace flatten with Global Average Pooling | **Best fit.** Handles variable grid sizes natively (no padding). +17% generalization, -35% params vs IMPALA on Procgen. |

**Why Impoola:** Our layouts range from 4x5 (cramped_room) to 5x9 (asymm_advantages). GAP removes spatial dimension dependence, so a single network handles all layouts without padding. This also reduces translation sensitivity, which is desirable when we're NOT agent-centering the observation.

### OvercookedV2's Architecture (for comparison)

OvercookedV2 found that the standard Carroll architecture did not learn under their harder scenarios. Their fix:
- **Three 1x1 convolutions** before the main conv layers (to transform sparse boolean inputs)
- **LSTM layer** after the CNN (for temporal reasoning under partial observability)
- Still uses structured channel encoding, not raw pixels

If we add partial observability later, we should consider adding an LSTM too.

### Training Adjustments for Raw Pixels

Based on literature (DrQ, CURL, DreamerV3, Melting Pot baselines):

| Parameter | Current (MLP + features) | Expected (CNN + raw pixels) |
|---|---|---|
| Training steps | ~1000 updates | 5-10x more (5000-10000 updates) |
| Batch size | Current | 2-4x larger |
| Input normalization | None (boolean) | Divide by 255 |
| Data augmentation | None | Random shift (DrQ-style) recommended |
| Recurrence | None | Consider LSTM if partial obs added |
| Encoder LR | Same as policy | Optionally lower (0.5-1x policy LR) |

### Key References

- Impoola: arxiv.org/abs/2503.05546
- IMPALA CNN: arxiv.org/abs/1802.01561
- DrQ (image augmentation for RL): arxiv.org/abs/2004.13649
- CURL (contrastive representations): arxiv.org/abs/2004.04136
- Melting Pot (pixel-based MARL benchmark): github.com/google-deepmind/meltingpot

### Experiment Design

**Hypothesis:** The 26-channel egocentric encoding captures sufficient information for standard Overcooked. Raw pixels will require more training but should reach similar performance. The real question is whether raw pixels generalize better to unseen layouts or unseen partners.

**Comparisons:**
1. SP with egocentric MLP (baseline, already done)
2. SP with raw pixels + Impoola CNN
3. Cross-play comparison: does raw pixel training produce more robust agents?
4. (If we add procedural layouts) Generalization to unseen layouts: features vs. pixels

---

## Part 2: Harder ZSC — Beyond Standard Overcooked

### Why Standard Overcooked is Insufficient

OvercookedV2 (ICLR 2025) demonstrated that:
- The SP-XP gap in standard Overcooked is primarily a **state coverage** problem, not a coordination problem
- **State augmentation** (sampling starting states from cross-play rollouts, then training in self-play) nearly eliminates the gap
- Most standard layouts have few high-return strategies with limited convention ambiguity
- Full observability means agents never need to communicate — they can independently compute optimal policies from the current state

**Implication:** Our current SP/MEP/FCP/BCP experiments may be measuring generalization quality rather than coordination ability.

### What Makes ZSC Genuinely Hard

From the literature survey, genuine ZSC difficulty requires:

1. **Information asymmetry** — agents have different private information and must share it to coordinate
2. **Convention ambiguity** — multiple equally good strategies exist, agents must independently converge on the same one
3. **Test-time adaptation** — no fixed policy works; agents must form protocols on-the-fly during the episode

Standard Overcooked has none of these.

### Candidate Directions (Ranked by Impact x Feasibility)

#### Direction 1: Integrate OvercookedV2 Features

**What:** Add partial observability, multiple ingredients, recipe indicators, and communication buttons to our JaxMARL-based environment.

**Why:** OvercookedV2 is already in JaxMARL (github.com/FLAIROx/JaxMARL). It's the most direct upgrade. None of the tested methods (SP, State-Augmented, Other-Play, FCP) solve it — XP scores go negative in some layouts.

**Three layout classes from OvercookedV2:**
- **Grounded Coordination:** One agent sees recipe indicator, must communicate via costly button press. SP agents learn to skip the button (arbitrary convention), which fails in ZSC.
- **Test-Time Protocol Formation:** Asymmetric info + delivery feedback but no explicit channel. Agents must develop conventions during the episode through trial-and-error. Hardest class — requires fundamentally new methods.
- **Demo Cook:** Implicit communication through ingredient choice (placing onion vs tomato signals the recipe). Requires interpreting partner's actions as informative.

**OvercookedV2 results (Table 2 from paper):**

| Layout | Method | SP | XP | Gap |
|---|---|---|---|---|
| Grounded Coord. Simple | Self-Play | 154 | -55 | 210 |
| | FCP | 64 | 46 | 22 |
| Test-Time Simple | Self-Play | 145 | -81 | 220 |
| | FCP | 35 | 6 | 25 |
| Demo Cook Simple | Self-Play | 172 | 36 | 138 |
| | FCP | 22 | 1 | 25 |

Note: FCP has lower gap but also much lower absolute scores. No method achieves good XP performance.

**Feasibility:** High — same JaxMARL framework. Would need to:
- Add view radius masking to observation encoding
- Add ingredient variety + recipe sampling to environment
- Add recipe indicator blocks and button mechanics
- Add LSTM to network architecture

**Key architectural change from OvercookedV2:** Their observation encoding uses 1x1 convolutions to process sparse inputs + LSTM for temporal reasoning. This is not egocentric or raw — it's a modified structured encoding with partial observability.

#### Direction 2: State Augmentation (as a baseline sanity check)

**What:** Implement the state augmentation algorithm from OvercookedV2 Section 5.1.

**Why:** If state augmentation closes the SP-XP gap in our experiments too, it confirms that our current results are measuring state coverage, not coordination. This is a necessary sanity check before claiming our methods improve ZSC.

**How:**
1. Train 10 independent SP agents
2. Roll out cross-play episodes for all pairings
3. Sample states from those rollouts into a buffer B
4. Re-train each agent in self-play, sampling starting states from B
5. Repeat for 10 iterations

**Feasibility:** Medium — requires modifying the environment reset to accept arbitrary starting states, and adding the cross-play rollout + buffer logic.

#### Direction 3: Procedural Layout Generation (CEC approach)

**What:** Build a JAX-based layout generator that randomizes walls, station positions, and map dimensions. Train across distributions of layouts.

**Why:** CEC (Cross-Environment Cooperation, 2025) showed that environment diversity beats partner diversity for ZSC. Agents learn "general norms" that transfer to both unseen layouts AND unseen partners. Up to 10^17 possible configurations.

**Key finding from CEC:** CEC outperforms FCP and other population methods. In human studies, CEC agents were rated most adaptable and least frustrating.

**Key finding from OGC (Overcooked Generalisation Challenge, TMLR 2025):** Most methods achieve near-zero returns on held-out procedurally generated layouts. Only Population PAIRED with SoftMoE-LSTM achieves 14.6% solved levels. This is a wide-open research problem.

**Feasibility:** Medium-high — our layout_grid_to_dict() already supports ASCII-defined layouts. Need a generator that produces valid solvable layouts.

**Synergy with raw pixels:** Procedural layouts vary in size. Raw pixels + Impoola (GAP) handle variable sizes naturally, while the fixed 7x7 egocentric encoding would need padding/resizing. This is where raw observation mode would really shine.

#### Direction 4: N-Agent Scaling (Already Partially Implemented)

**What:** Run experiments with N=3-5 agents on larger layouts.

**Why:** NAHT (NeurIPS 2024) showed coordination difficulty grows combinatorially with N. Single-agent ad hoc policies don't scale. Our N-XPlay framework already supports this.

**Feasibility:** High — code already supports N agents. Just need to run experiments and update evaluation.

#### Direction 5: Noisy ZSC (NZSC)

**What:** Break common knowledge — agents observe different noisy versions of the state. Reward corruption, speed mismatches, observation noise.

**Why:** Real-world motivated (different robot manufacturers with different sensors). Meta-NZSC training (training with diverse noise distributions) substantially outperforms standard approaches.

**Feasibility:** Medium — relatively easy to add observation noise or reward perturbation, but the meta-training framework is more involved.

---

## Recommended Roadmap

### Phase 1: Validate current results (are we actually measuring ZSC?)
1. Implement state augmentation baseline
2. If it closes the SP-XP gap on standard layouts -> confirms OvercookedV2's findings
3. This tells us whether our MEP/FCP improvements over SP are about coordination or just coverage

### Phase 2: Raw image experiments (novel contribution)
1. Implement raw observation mode + Impoola CNN
2. Run SP with raw pixels on standard layouts
3. Compare training efficiency and final performance vs. egocentric MLP
4. Compare cross-play robustness

### Phase 3: Move to genuinely hard ZSC
1. Integrate OvercookedV2 features (partial obs, recipes, asymmetric info)
2. Test raw pixels vs. structured encoding on OvercookedV2 layouts
3. Run N-agent experiments (N=3-5)
4. (Optional) Procedural layout generation for generalization experiments

### Phase 4: Novel methods for hard ZSC
1. Test-time adaptation mechanisms (no existing method solves this)
2. Implicit communication through actions (Demo Cook scenarios)
3. Meta-learning across noise distributions (NZSC)

### Phase 5: Dual-Franka ZSC in simulation
1. Build dual-Franka multi-agent environment in Isaac Lab
2. Start with simple tasks (independent reach, handover) to validate the multi-agent framing
3. Apply ZSC methods (SP populations, FCP, MEP, N-XPlay)
4. Compare raw point cloud vs. hand-crafted state features — same debate as Overcooked
5. Scale to harder tasks (shared lift, collaborative assembly, kitchen-style)

### Phase 6: Sim-to-real transfer
1. Domain randomization in Isaac Lab for sim-to-real
2. Transfer to real dual-Franka setup
3. Zero-shot coordination test: can one trained arm coordinate with an unseen real arm?
4. Human-robot ZSC: replace one arm with a human operator

---

## Part 3: From Gridworld ZSC to Real Robotics (Dual Franka Arms)

### The Gap

**No one has done ZSC on real robots.** The entire ZSC literature is in gridworlds (Overcooked) and card games (Hanabi). This is the biggest open opportunity.

Our lab has two Franka Panda arms. If our ZSC methods can transfer from simulation to real dual-arm coordination, that would be a major contribution — bridging the gap between ZSC theory and real-world deployment.

### Current State of Dual-Arm Coordination

Almost all dual-arm manipulation work uses **centralized control** (single policy controlling both arms), not decentralized multi-agent policies. Key works:

- **DA-VIL** (ICRA 2025): Dual-arm RL with variable impedance control. Two Franka-style arms. Centralized.
- **Stabilize to Act / BUDS** (CoRL 2023, Oral): Role-based bimanual manipulation — one arm stabilizes, one acts. 76.9% success with 20 demos. Interesting for heterogeneous ZSC.
- **Bi-VLA** (2024): Vision-Language-Action model for bimanual dexterous manipulation. Uses VLM pipeline, not RL.
- **Curriculum Design for Dual-Arm Assembly** (Machines, 2024): Two Franka Pandas doing peg-in-hole (1mm clearance). RL + impedance control. Sim-to-real demonstrated.

**Key observation:** Framing dual-arm manipulation as a *multi-agent* ZSC problem (each arm is an independent agent that must coordinate zero-shot) would be novel.

### Closest Work to Robot ZSC

- **Edinburgh SPADS CDT PhD project**: Open position explicitly proposing ZSC for real robots. Indicates this is recognized as an open problem but not yet solved.
- **NAHT** (NeurIPS 2024): N-agent ad hoc teamwork with POAM (Policy Optimization with Agent Modelling). Simulation only but the problem formulation applies to robots.
- **HOTSPOT** (PLOS ONE 2024): Ad hoc teamwork in mixed human-robot teams — the real-world analogue of ZSC.
- **Mixed-Initiative Human-Robot Teaming** (2024): Online Bayesian adaptation for human-robot coordination without prior human data.

### Simulation Environments for Dual-Arm ZSC

#### Best candidates for building a dual-Franka ZSC benchmark:

**1. Bi-DexHands (NeurIPS 2022) — Most mature multi-agent bimanual benchmark**
- Two Shadow Dexterous Hands in Isaac Gym
- Tasks: handover, lift, throw, place, catch, stack blocks, open bottles, lift pots
- >2000 objects (YCB/SAPIEN). 40,000+ FPS on single RTX 3090
- Supports MARL natively (HAPPO, MAPPO, MADDPG, etc.)
- Uses Shadow Hands not Frankas, but the framework and tasks are directly relevant
- GitHub: github.com/PKU-MARL/DexterousHands

**2. Isaac Lab — Best for Franka sim-to-real**
- NVIDIA's successor to IsaacGym. Supports multi-agent RL natively
- Franka manipulation environments included (Lift-Cube, Cabinet)
- 150,000+ FPS for Franka tasks
- MARL extensions available (github.com/TIERS/multi-agent-rl-omni)
- Best path for sim-to-real with domain randomization

**3. Robosuite Two-Arm Environments**
- Two-Arm Peg-In-Hole, Two-Arm Handover (hammer pickup + handover)
- MuJoCo-based, well-documented API
- Safe MARobosuite extension adds safety constraints for multi-agent
- GitHub: github.com/chauncygu/Safe-Multi-Agent-Robosuite

**4. RoboCasa (RSS 2024) — Kitchen simulation bridge**
- 100+ kitchen manipulation tasks, 2500+ 3D objects
- Supports Franka Panda and mobile manipulators
- Built on Robosuite/MuJoCo
- RoboCasa365 (Feb 2026): 365 tasks, 2200+ hours of demo data
- Most promising bridge between Overcooked-style kitchen tasks and real robot deployment

**5. RoboTwin (CVPR 2025 Highlight)**
- Dual-arm robot benchmark with generative digital twins
- 50 bimanual tasks across 5 robot embodiments, 731 objects

**6. DualTHOR (2025)**
- Dual-arm humanoid simulation on AI2-THOR
- Finding: current VLMs struggle badly with dual-arm coordination

### Observation Modality for Robot ZSC

**Point Cloud Matters** (NeurIPS 2024 D&B, Zhu et al.) tested across 125 tasks in OBSBench:
- **Point cloud methods frequently outperform RGB and RGB-D** in contact-rich manipulation
- Better generalization across geometric/visual conditions
- Better sample efficiency than image-based methods

**Recommended observation stack for dual-Franka ZSC:**
- **Point cloud** for environment and partner perception (geometric, transfers well sim-to-real)
- **Proprioception** (joint angles, velocities, forces) for own-state awareness
- **Optional RGB** for richer context if needed

This connects to the raw-vs-features debate: in robotics, raw point clouds are the analogue of raw pixels in gridworlds — you learn your own representation instead of hand-crafting features.

### Vision-Language-Action Models and Multi-Robot Coordination

- **Helix** (Figure AI, Feb 2025): First VLA operating two robots simultaneously. Two Figure 02 humanoids coordinate on shared tasks via natural language. Dual-system: System 2 (7-9 Hz, scene understanding) + System 1 (200 Hz, reactive control). Trained on 500 hours of teleoperation. Coordination via language prompts, not learned conventions.
- **Octo** (2024): 93M-param transformer diffusion policy, trained on 800K episodes (Open X-Embodiment). Single-agent only.
- **OpenVLA** (Stanford, 2024): 7B-param VLA, outperforms RT-2 and Octo. Single-agent only.
- **Position paper** (OpenReview 2025): "Robotic Foundation Models Should Evolve Toward an Interactive Multi-Agent Perspective" — explicitly argues current models' single-agent assumption limits coordination.

**Key insight:** Helix shows multi-robot VLA coordination is possible but uses language as the coordination mechanism. ZSC asks: can robots coordinate *without* a shared language, through learned conventions? This is a harder and more fundamental problem.

### Sim-to-Real Transfer for Multi-Agent

- **AutoMate** (RSS 2024, NVIDIA): Zero-shot sim-to-real with 84.5% success on 20 assemblies. Single-arm but demonstrates the pipeline.
- **TRANSIC** (CoRL 2024): Human-in-the-loop sim-to-real via online corrections. Residual policies bridge unmodeled gaps. Could be combined with ZSC for real deployment.
- **Duckie-MAAD** (2024): MAPPO + domain randomization for multi-agent driving, transferred to real miniature cars. One of the few true multi-agent sim-to-real demonstrations.

### Proposed Path: From Overcooked ZSC to Dual-Franka ZSC

#### Step 1: Build a dual-Franka ZSC benchmark in simulation

Create a multi-agent environment in Isaac Lab with two Franka Pandas in a shared workspace. Tasks with increasing coordination difficulty:

| Task | Coordination Type | ZSC Difficulty |
|---|---|---|
| Independent reach (separate targets) | None (baseline) | Easy |
| Handover (pass object between arms) | Sequential, role assignment | Medium |
| Shared lift (both arms lift one heavy object) | Tight coupling, force coordination | Hard |
| Collaborative assembly (peg-in-hole, one holds + one inserts) | Asymmetric roles, timing | Hard |
| Kitchen-style (fetch ingredients, prepare, serve) | Overcooked analogue in continuous control | Hardest |

Each arm is an independent agent with:
- Observation: proprioception + point cloud of workspace (or RGB from wrist camera)
- Action: joint torques or end-effector pose (continuous)
- Reward: shared task completion reward

#### Step 2: Apply ZSC methods from Overcooked

- Train populations via self-play (SP), then evaluate cross-play
- Apply FCP/MEP population diversity methods
- Test N-XPlay with ego + unseen partner arms
- Compare: does raw point cloud observation produce better ZSC than hand-crafted state features?

#### Step 3: Sim-to-real transfer

- Domain randomization in Isaac Lab (friction, mass, visual textures)
- Transfer trained policies to real dual-Franka setup
- Evaluate: can a Franka arm trained in simulation coordinate zero-shot with a real arm it has never been paired with?
- Optional: TRANSIC-style human corrections to bridge residual gap

#### Step 4: Human-robot ZSC

- Replace one Franka arm with a human operator (teleoperation or direct manipulation)
- Evaluate: can the trained arm coordinate with an unseen human partner?
- This is the ultimate test of ZSC — and directly relevant to real-world deployment

### Key Gap This Would Fill

There is **no existing benchmark** that combines:
1. Continuous action spaces
2. Physics-based simulation
3. Manipulation tasks
4. ZSC evaluation protocol (train ego, evaluate with unseen partners)

Building this would be a significant contribution to both the ZSC and robotics communities.

### Key References for Part 3

- Bi-DexHands: arxiv.org/abs/2206.08686
- Isaac Lab: github.com/isaac-sim/IsaacLab
- Robosuite Two-Arm: robosuite.ai/docs/modules/environments.html
- RoboCasa: robocasa.ai
- RoboTwin: arxiv.org/abs/2409.02920
- Point Cloud Matters: arxiv.org/abs/2402.02500
- Helix (Figure AI): figure.ai/news/helix
- AutoMate: research.nvidia.com/publication/2024-07_automate
- TRANSIC: arxiv.org/abs/2405.10315
- DA-VIL: arxiv.org/abs/2410.19712
- Stabilize to Act: arxiv.org/abs/2309.01087
- NAHT: NeurIPS 2024
- franka_bimanual_controllers: github.com/franzesegiovanni/franka_bimanual_controllers
- Multi-Agent FM Position Paper: openreview.net (2025)

---

## Key Papers (All Parts)

| Paper | Venue | Key Idea | Relevance |
|---|---|---|---|
| OvercookedV2 (Gessler et al.) | ICLR 2025 | Standard Overcooked too easy; partial obs + asymmetric info needed | Primary motivation for moving beyond current setup |
| CEC (Cross-Env Cooperation) | 2025 | Environment diversity > partner diversity; procedural layouts | Alternative to population methods |
| OGC (Overcooked Generalisation Challenge) | TMLR 2025 | UED for Overcooked; most methods fail on unseen layouts | Open problem for generalization |
| Impoola | 2025 | GAP replaces flatten in IMPALA CNN; variable input sizes | CNN architecture for raw pixels |
| NAHT | NeurIPS 2024 | N-agent ad hoc teamwork; single-agent policies don't scale | Motivation for N-XPlay experiments |
| Noisy ZSC | 2024 | Break common knowledge with observation/reward noise | Harder ZSC without env changes |
| ZSC-Eval | NeurIPS 2024 | Evaluation toolkit with BR-Div and BR-Prox metrics | Better evaluation methodology |
| ScaPT | 2025 | Hierarchical meta-agent for large populations | Scalable population training |
| MAZE | 2024 | Coevolution for heterogeneous ZSC | Separate populations per role |
| Melting Pot | NeurIPS 2023 | Pixel-based multi-agent benchmark, IMPALA CNN | Closest analogue for raw pixel MARL |
| Bi-DexHands | NeurIPS 2022 | Two Shadow Hands in Isaac Gym, MARL bimanual manipulation | Most mature multi-agent bimanual benchmark |
| Point Cloud Matters (Zhu et al.) | NeurIPS 2024 D&B | Point clouds outperform RGB in contact-rich manipulation | Observation modality for robot ZSC |
| AutoMate (NVIDIA) | RSS 2024 | Zero-shot sim-to-real, 84.5% success across 20 assemblies | Sim-to-real pipeline for manipulation |
| TRANSIC | CoRL 2024 | Human-in-the-loop sim-to-real via online corrections | Bridging residual sim-to-real gap |
| Stabilize to Act / BUDS | CoRL 2023 (Oral) | Role-based bimanual: one arm stabilizes, one acts | Heterogeneous coordination paradigm |
| Helix (Figure AI) | 2025 | First VLA controlling two robots simultaneously | Multi-robot coordination via language |
| RoboCasa | RSS 2024 | 100+ kitchen tasks, Franka support, MuJoCo | Bridge from Overcooked to real kitchen robotics |
| RoboTwin | CVPR 2025 Highlight | 50 bimanual tasks, 5 embodiments, generative digital twins | Dual-arm benchmark |
| DA-VIL | ICRA 2025 | Dual-arm RL + variable impedance control | Franka-style dual-arm RL |
| FM Multi-Agent Position Paper | OpenReview 2025 | Foundation models need multi-agent perspective | Argues current single-agent VLAs are limited |
| CEC / InfiniteKitchen | ICML 2025 | Procedural Overcooked generator, env diversity > partner diversity | Most advanced ZSC method, human-validated |
