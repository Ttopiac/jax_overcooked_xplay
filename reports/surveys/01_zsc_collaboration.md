# Survey: Zero-Shot Coordination, Human-AI/Robot Collaboration, and Decentralized Adaptation

## 1. Zero-Shot Coordination (ZSC)

### 1.1 Problem Definition
Train an agent that can coordinate with an **unseen partner** at test time — no prior interaction, no shared training history. The agent must independently converge on compatible behavior.

### 1.2 Population-Based Methods (current approach, expensive)

**Fictitious Co-Play (FCP)** — Strouse et al., 2021 (arXiv:2110.08176)
- Train ego as best response to a diverse population of self-play agents + past checkpoints
- Requires training full SP population first → expensive
- Our Run 3 baseline

**Maximum Entropy Population (MEP)** — Zhao et al., 2022 (arXiv:2112.11701)
- Train population with entropy bonus for diversity, then train ego against it
- Even more expensive: entropy bonus requires CNN forward on ALL pop members every step
- Our Run 3 baseline

**Hierarchical Population Training** — Loo et al., 2023 (arXiv:2305.16708)
- Dynamically switch between policies based on partner adaptation
- Still requires population training

### 1.3 Population-Free Methods (the future direction)

**Other-Play (OP)** — Hu et al., 2020 (arXiv:2003.02979)
- Play against symmetry-transformed versions of yourself
- **No population needed** — single agent training
- Requires environment symmetries to exploit
- Extended by: Equivariant Networks for ZSC (arXiv:2210.12124), K-level Reasoning (arXiv:2207.07166)

**PECAN: Policy Ensemble for Context-Aware ZSC** — Lou et al., 2023 (arXiv:2301.06387)
- Policy ensemble with diversity, adapts to novel human coordination
- Lighter than full population training

**Cross-environment Cooperation (CEC)** — Jha et al., 2025 (arXiv:2504.12714)
- Learn cooperative skills across multiple environments
- Zero-shot transfer to new partners without partner-specific training

**In-Context Coordination (CooT)** — arXiv:2506.23549
- Uses recent interaction history to rapidly adapt to unseen partners via transformers
- **No population** — learns to adapt from context, like in-context learning in LLMs

**Learning to Cooperate with Unseen Agents via Meta-RL** — Charakorn et al., 2021 (arXiv:2111.03431)
- Meta-learning approach: learn to quickly adapt to any partner
- Single training run produces universally adaptive agent

### 1.4 Theory of Mind Approaches

**ToM-Guided Strategy Adaptation** — Ni et al., 2025 (arXiv:2602.12458)
- Infer teammate intentions via Theory of Mind, select best-response strategy
- No population — builds partner model online

**GAMMA: Generative Agents for ZSC** — Liang et al., 2024 (arXiv:2411.13934)
- Use generative modeling to produce diverse synthetic partner behaviors
- Train ego against generated partners — cheaper than training real agent population

### 1.5 Evaluation and Benchmarks

**ZSC-Eval** — Wang et al., 2023 (arXiv:2310.05208)
- Evaluation toolkit and benchmark for multi-agent ZSC

**OvercookedV2** — Gessler et al., 2025 (arXiv:2503.17821)
- Enhanced benchmark with asymmetric information and stochasticity

**Overcooked Generalisation Challenge** — Ruhdorfer et al., 2024 (arXiv:2406.17949)
- Benchmark for cooperation with unknown partners in unknown environments

---

## 2. Human-AI Collaboration

### 2.1 Adapting to Human Partners

**Coordination via Strategy Matching** — Zhao et al., 2022 (arXiv:2210.15099)
- Observe human-human teams → learn Mixture-of-Experts that matches task strategies
- Adapts to unseen human partner styles

**Tackling Cooperative Incompatibility** — Li et al., 2023 (arXiv:2306.03034)
- Graph-theory approach to address coordination failures with novel humans

**Conditional Imitation Learning** — Shih et al., 2022 (arXiv:2201.01448)
- Infer and adapt to partner strategy without reward or dynamics knowledge

**Role Play: Adaptive Role-Specific Strategies** — Long et al., 2024 (arXiv:2411.01166)
- Role embeddings improve ZSC with unseen agents

### 2.2 Understanding Human Partners

**Partner Modelling Emerges in Recurrent Agents** — Mon-Williams et al., 2025 (arXiv:2505.17323)
- Internal representations of partners emerge when beneficial for coordination

**Eyes on the Game** — Hulle et al., 2024 (arXiv:2407.03298)
- Implicit human signals (gaze + gameplay) reveal proficiency, trust, intent

**Gaze-informed Trust Signatures** — Ries et al., 2024 (arXiv:2409.19139)
- Eye-tracking metrics for team trust and performance

### 2.3 Evaluating Robustness

**Evaluating Robustness of Collaborative Agents** — Knott et al., 2021 (arXiv:2101.05507)
- Unit test methodology for edge case robustness

**Unsupervised Partner Design** — Ruhdorfer et al., 2025 (arXiv:2508.06336)
- Generate diverse partners without supervision for robust ad-hoc teamwork

---

## 3. Human-Robot Physical Collaboration

### 3.1 Adaptation in Physical Tasks

**Fixed-time Adaptive Neural Control for Physical HRC** — Sun et al., 2023 (arXiv:2303.02456)
- Compliance control during physical contact with workspace constraints

**Optimal Synergistic Human-Robot Coordination in Manufacturing** — Sandrini et al., 2025 (arXiv:2503.07238)
- Task allocation using learned synergies from Bayesian estimation

**Real-time Music Accompaniment** — Wang et al., 2024 (arXiv:2409.11952)
- Behavior-adaptive controller for temporal synchronization with human musicians

**Dynamic Human Task Proficiencies** — Liu et al., 2020 (arXiv:2007.01921)
- Robots explore relative strengths and learning abilities of human teammates

### 3.2 Sim-to-Real for Collaboration

**MonTA: Real-time Adaptation of Embodied Agent** — Liu et al., 2024 (arXiv:2412.00435)
- Hierarchical framework for monitoring and adapting to human behavior changes

**FABRIC Framework** — arXiv:2104.01976
- Design framework addressing human behavioral diversity and real-time adaptation

---

## 4. How Humans Adapt Collaboration Dynamics

### 4.1 Key Mechanisms (from cognitive science / HRI papers)

**Dual Process Theory for Real-time Collaboration** — arXiv:2507.14088
- Fast (System 1) and slow (System 2) reasoning during collaboration
- Humans use quick heuristics for routine coordination, deliberate reasoning for novel situations

**Goal Inference from Motion** — arXiv:1802.01780
- Humans rapidly infer partner goals from observed actions
- Behavioral experiments: humans prefer partners that adapt to their actions

**Rational Communication** — arXiv:2510.23340
- Humans project partner awareness across future timesteps
- Signaling strategies adapted to what the partner can perceive

### 4.2 Key Human Adaptation Patterns
Based on the papers above, humans adapt to new collaborators through:
1. **Rapid intent inference** — observing a few actions to predict partner goals
2. **Strategy matching** — recognizing familiar collaboration patterns and adapting
3. **Mutual adaptation** — both partners adjust simultaneously until convergence
4. **Communication** — verbal and non-verbal signals to reduce uncertainty
5. **Mental modeling** — building and updating a model of the partner's capabilities

---

## 5. Sperm Collective Dynamics: Decentralized Coordination Among Billions

### 5.1 The Phenomenon
~300 million sperm navigate the reproductive tract, with only ~1 reaching the egg. No central controller. Each sperm makes independent decisions based on local sensing.

### 5.2 Navigation Mechanisms

**Three Guidance Systems** — Perez-Cerezales, Boryshpolets, Eisenbach, 2015 (PubMed)
- **Chemotaxis**: Following chemical gradients from the egg/oviductal fluid
- **Thermotaxis**: Swimming toward warmer regions (the egg is ~2°C warmer)
- **Rheotaxis**: Swimming against fluid flow in the reproductive tract

Each mechanism operates at different distances:
- Rheotaxis: long-range (centimeters, against fluid flow)
- Thermotaxis: mid-range (millimeters, temperature gradient)
- Chemotaxis: short-range (micrometers, near the egg)

### 5.3 Cooperation and Competition

**The Dynamics of Sperm Cooperation in a Competitive Environment** — Fisher, Giomi, Hoekstra, Mahadevan, 2014 (Proceedings of the Royal Society B)

Key findings:
- Sperm form **cooperative aggregates** of 6-7 cells using hook-shaped head adhesion
- Groups move with **more persistent linearity** but not faster than individuals
- Optimal group size: **6-7 sperm** — larger groups suffer geometric interference (members swim against each other)
- Sperm from **promiscuous species** (intense competition) form optimal-sized groups more often than monogamous species
- **Sexual selection pressure** drives evolution of cooperative behavior

### 5.4 Relevance to Multi-Agent Coordination

The sperm system demonstrates several principles applicable to AI:

1. **Local sensing, global coordination**: Each agent uses only local signals (chemical gradient, temperature, flow) yet the population collectively solves a global navigation problem. No agent has a global view.

2. **Emergent optimal group size**: Small cooperating groups (6-7) outperform both individuals and large groups. There's a sweet spot between independence and coordination — paralleling our observation that population size matters in ZSC training.

3. **Multi-modal sensing at different scales**: Different sensing mechanisms activate at different distances (rheotaxis → thermotaxis → chemotaxis). This is analogous to hierarchical RL policies that use different strategies at different time scales.

4. **Competition drives cooperation**: Intense competitive pressure (promiscuous mating) evolves MORE cooperative behavior, not less. In ZSC, competitive diversity in the training population produces more robust coordinators.

5. **No training, pure evolution**: Sperm don't learn or adapt individually. Their coordination strategies are encoded genetically through millions of years of evolution. This is the ultimate "zero-shot" — the mechanism must work on the first and only attempt, with no prior interaction with this specific partner or environment.

6. **Massive redundancy as a strategy**: 300 million agents for 1 success. The system doesn't optimize individual performance — it optimizes population-level success probability through diversity and redundancy.

---

## 6. Implications for Our Project

### The Population Training Bottleneck
Current ZSC methods (FCP, MEP) require expensive population training. From the survey, promising alternatives include:

1. **Other-Play** (population-free, exploits symmetries) — directly applicable to Overcooked which has positional symmetries
2. **In-context coordination** (CooT) — agent adapts from interaction history, no population
3. **Generative partner modeling** (GAMMA) — generate diverse partners cheaply instead of training them
4. **Meta-learning** — learn to adapt to ANY partner in a few steps

### Inspiration from Sperm Biology
The sperm system suggests:
- **Small cooperative groups** (6-7) outperform large populations — maybe we don't need 16-agent populations, 6-8 might be optimal
- **Multi-scale sensing** — agents could use different observation modes at different decision scales (pixel for spatial reasoning, abstract features for planning)
- **The coordination mechanism should be zero-shot by design** — hardcoded inductive biases rather than learned through expensive population training

### Visual Observations for Robotics Transfer
From the survey, NO existing ZSC paper uses pixel observations for Overcooked or similar tasks. All use engineered features. We are pioneering this direction, which is necessary for:
- Sim-to-real transfer (robots have cameras, not feature extractors)
- Human-interpretable agent behavior (humans and agents see the same thing)
- Generalization to unseen environments (pixel-based policies transfer to new kitchens)
