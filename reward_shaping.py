"""Reward shaping utilities for Overcooked.

Three modes:
  - "sparse":     Sparse delivery reward only.
  - "shaped":     Carroll-style: sparse + factor * shaped.
  - "originaler": multiHRI-style: group_sparse * ratio + individual_shaped * (1-ratio).
                  ratio ramps from 0 → final_ratio over warmup_steps.

Usage:
    shaper = RewardShaper(mode="originaler")
    reward = shaper.compute(rewards, infos, agent_key, step_count)
"""

import jax.numpy as jnp


class RewardShaper:
    """Configurable reward computation.

    Args:
        mode: "sparse", "shaped", or "originaler".
        reward_shaping_factor: For "shaped" mode — multiplier on dense reward.
        reward_shaping_horizon: For "shaped" mode — annealing horizon.
        final_ratio: For "originaler" mode — target sparse ratio (default 0.5).
        warmup_steps: For "originaler" mode — steps to reach final_ratio (default 1e7).
    """

    def __init__(
        self,
        mode="originaler",
        reward_shaping_factor=1.0,
        reward_shaping_horizon=float("inf"),
        final_ratio=0.5,
        warmup_steps=10_000_000,
    ):
        assert mode in ("sparse", "shaped", "originaler"), f"Unknown reward mode: {mode}"
        self.mode = mode
        self.initial_factor = reward_shaping_factor
        self.horizon = reward_shaping_horizon
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def compute(self, rewards, infos, agent_key, step_count=0):
        """Compute reward for one agent across a batch of environments.

        Args:
            rewards: dict from env.step, {"agent_0": (N,), ...}.
            infos: dict from env.step, with infos["shaped_reward"][agent_key].
            agent_key: "agent_0" etc.
            step_count: cumulative environment step.

        Returns:
            (N,) float32 reward.
        """
        sparse_r = rewards[agent_key]

        if self.mode == "sparse":
            return sparse_r

        shaped_r = infos["shaped_reward"][agent_key]

        if self.mode == "shaped":
            # Carroll: sparse + factor * shaped, factor anneals to 0
            if self.horizon == float("inf") or self.horizon == 0:
                factor = self.initial_factor
            else:
                factor = jnp.maximum(1.0 - step_count / self.horizon, 0.0) * self.initial_factor
            return sparse_r + factor * shaped_r

        # originaler: group_sparse * ratio + individual_shaped * (1-ratio)
        # ratio ramps from 0 → final_ratio over warmup_steps
        group_sparse = sum(rewards[k] for k in rewards if k != "__all__")
        ratio = jnp.minimum(step_count / self.warmup_steps, self.final_ratio)
        return group_sparse * ratio + shaped_r * (1.0 - ratio)
