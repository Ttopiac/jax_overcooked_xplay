# Survey: Visual RL Architectures and Robot Observation Inputs

## 1. Visual Encoder Architectures for Pixel-Based RL

### 1.1 The Standard: Nature DQN CNN (2013-2015)
- **Architecture**: 3 conv layers (32@8×8/stride4, 64@4×4/stride2, 64@3×3/stride1) → FC(512)
- **Input**: 84×84 grayscale (4 frame stack)
- **Used in**: Atari (Mnih et al., 2015), many subsequent RL papers
- **Status**: Baseline architecture. Procgen paper showed it "almost completely fails" on diverse environments.

### 1.2 IMPALA CNN (2018)
- **Shallow**: 2 conv layers (16@8×8/stride4, 32@4×4/stride2) → LSTM(256) → FC(256). **1.2M params.**
- **Deep (ResNet)**: 3 blocks × (Conv + MaxPool + 2 ResBlocks) at channels [16, 32, 32]. **15 conv layers, 1.6M params.**
- **Input**: 96×72 (DMLab), 84×84 (Atari)
- **Used in**: IMPALA (Espeholt et al., 2018), Procgen (Cobbe et al., 2020)
- **Key finding**: Deep model needed for diverse environments (Procgen). Shallow sufficient for single-task.
- **Paper**: arXiv:1802.01561

### 1.3 DreamerV3 World Model (2023)
- **Encoder**: CNN that maps images → discrete latent representations z_t
- **Key insight**: CNN runs ONLY on replay buffer samples. Actor/critic train on imagined latent trajectories (no CNN needed). This avoids the CNN-per-step bottleneck.
- **Hardware**: Single A100 GPU
- **Used in**: Atari (200M), Procgen (50M), DMLab (100M), Minecraft (100M)
- **Paper**: arXiv:2301.04104

### 1.4 DrQ / DrQ-v2: Data Augmentation for Visual RL (2020-2021)
- **Key idea**: Simple image augmentation (random shift) makes model-free RL work from pixels without world models or contrastive learning.
- **Encoder**: Standard CNN (similar to SAC encoder, ~4 conv layers)
- **Input**: 84×84 RGB
- **Achievement**: DrQ-v2 solved humanoid locomotion from pixels on single GPU in 8 hours.
- **Papers**: arXiv:2004.13649 (DrQ), arXiv:2107.09645 (DrQ-v2)

### 1.5 CURL: Contrastive Learning for Visual RL (2020)
- **Key idea**: Contrastive learning extracts features from pixels, nearly matching state-based efficiency.
- **Encoder**: CNN encoder trained with contrastive objective
- **Input**: 84×84 or 100×100 RGB
- **Paper**: arXiv:2004.04136

### 1.6 RAD: Reinforcement Learning with Augmented Data (2020)
- **Key idea**: Data augmentation alone (crop, translate, color jitter) dramatically improves pixel-based RL.
- **Used with**: Any existing RL algorithm + standard CNN encoder
- **Paper**: arXiv:2004.14990

### 1.7 Masked World Models / MVP: Pre-trained Visual Encoders (2022)
- **Key idea**: Pre-train visual encoder (CNN or ViT) via masked image modeling on in-the-wild video, then freeze it for RL.
- **Encoder**: Compared CNNs vs Vision Transformers (ViT)
- **Key finding**: Pre-trained encoders outperform supervised encoders by up to 80% on robot manipulation
- **Papers**: arXiv:2206.14244 (MWM), arXiv:2203.06173 (MVP)

### 1.8 R3M: Pre-trained on Human Video (2022)
- **Key idea**: Pre-train visual representation on Ego4D human video dataset using time-contrastive + video-language alignment
- **Result**: +20% success over training from scratch, outperforms CLIP and MoCo by 10%
- **Used for**: Real Franka robot manipulation from 20 demonstrations
- **Paper**: arXiv:2203.12601

### 1.9 RT-2: Vision-Language-Action Models (2023)
- **Key idea**: Fine-tune large vision-language models (PaLI-X, PaLM-E) to output robot actions as text tokens
- **Input**: Raw camera images (same as the VLM's visual input)
- **No separate CNN** — uses the VLM's built-in visual encoder (likely ViT-based)
- **Paper**: arXiv:2307.15818

---

## 2. What Do Real Robots Actually Use as Input?

### 2.1 Raw Camera Images (End-to-End Visual RL)

These systems take raw RGB or RGB-D images directly:

- **SLIM** (arXiv:2501.09905): "single wrist-mounted RGB camera" → RL policy for quadruped manipulation. Sim-to-real transfer.
- **Sim-to-real vehicle control** (arXiv:2012.07461): "forward-facing monocular camera" → end-to-end driving.
- **Visual pushing/grasping** (arXiv:1803.09956): Overhead RGB-D → pixel-to-action mapping for manipulation.
- **Vision-based robotic control** (arXiv:1511.03791): "raw-pixel images" → robotic manipulator control.
- **RT-2** (arXiv:2307.15818): Raw camera → VLM → action tokens for real robot.
- **R3M** (arXiv:2203.12601): Camera images → pre-trained frozen encoder → policy. Real Franka arm.

### 2.2 Processed/Engineered Features (Common in Practice)

Many robot systems still use processed inputs:

- **Joint angles + end-effector position** (proprioception): Most common in manipulation RL
- **Object poses from perception pipeline**: Detected via separate vision system, fed as state vector
- **Point clouds**: From depth cameras, processed into structured 3D representations
- **Segmentation masks**: Pre-processed from raw images

### 2.3 The Trend: Moving Toward Raw Images

The field is clearly moving toward raw visual input:

1. **2015-2018**: Most robot RL used state features. Pixel RL only in simulation (Atari, MuJoCo).
2. **2019-2021**: DrQ, CURL, RAD made pixel RL sample-efficient. First sim-to-real with visual RL.
3. **2022-2023**: Pre-trained visual encoders (R3M, MVP) enable real robot learning from few demos.
4. **2024-2025**: VLMs (RT-2) take raw images and output actions. The encoder IS the foundation model.

### 2.4 Key Insight: The CNN is Rarely Trained End-to-End from Scratch in Robotics

Most successful real-robot visual RL systems use one of:
1. **Pre-trained frozen encoder** (R3M, MVP, CLIP) + learned policy head
2. **Foundation model** (RT-2, Octo) that already understands images
3. **Sim-trained encoder** transferred to real (with domain randomization)

Training a CNN from scratch on raw pixels with RL reward signal alone (what we're doing) is the hardest path. It works in simulation (Atari, DMControl) but is rare in real robotics due to sample inefficiency.

---

## 3. Implications for Our Project

### Architecture Choice
For sim-to-real transfer to real kitchen robots:
- **IMPALA shallow CNN trained from scratch** (our current approach) is viable for simulation but will likely need a pre-trained encoder for real robot transfer
- The IMPALA shallow architecture (Conv 8×8/stride4 → Conv 4×4/stride2) is actually the same pattern used in many successful visual RL systems
- For future real robot work: consider pre-training the encoder on kitchen video (R3M-style) then freezing it

### Why Our FCP Pixel CNN May Have Failed
Based on this survey, likely causes:
1. **No data augmentation**: DrQ/RAD showed augmentation is critical for pixel RL. We use none.
2. **End-to-end from scratch**: Most successful pixel RL systems use pre-trained or augmented encoders
3. **The dual-encoder training setup**: Ego sees pixels, partners see features — the ego may learn to exploit the partner's behavior rather than learning visual coordination

### Recommended Next Steps
1. **Add data augmentation** (random crop/shift) — this is the single biggest improvement for pixel RL (DrQ paper)
2. **Try the DreamerV3 approach**: Train a world model, learn policy in latent space. Avoids CNN-per-step cost.
3. **Pre-train encoder** on Overcooked gameplay video, then freeze for RL
4. For real robot: use R3M or similar pre-trained encoder on kitchen video
