"""Reward shaping utilities for Overcooked.

Provides configurable reward computation matching Carroll et al. (2019):
  - "sparse":  Sparse delivery reward only (20 per soup).
  - "shaped":  Dynamic blend of sparse + shaped rewards.
               reward = magnifier * (ratio * sparse + (1 - ratio) * shaped)
               ratio ramps from 0 → final_ratio over warmup_steps.

Usage:
    shaper = RewardShaper(mode="shaped")       # or "sparse"
    reward = shaper.compute(rewards, infos, agent_key, step_count)
"""

import jax.numpy as jnp


class RewardShaper:
    """Configurable reward computation.

    Args:
        mode: "sparse" or "shaped".
        magnifier: Reward scaling factor (default 3.0, matching original).
        final_ratio: Target sparse ratio at end of warmup (default 0.5).
        warmup_steps: Env steps over which ratio ramps from 0 to final_ratio
                      (default 10_000_000, matching original).
    """

    def __init__(
        self,
        mode="shaped",
        magnifier=3.0,
        final_ratio=0.5,
        warmup_steps=10_000_000,
    ):
        assert mode in ("sparse", "shaped"), f"Unknown reward mode: {mode}"
        self.mode = mode
        self.magnifier = magnifier
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def compute(self, rewards, infos, agent_key, step_count=0):
        """Compute reward for one agent across a batch of environments.

        Args:
            rewards: dict from env.step, e.g. {"agent_0": (num_envs,), ...}.
            infos: dict from env.step, with infos["shaped_reward"][agent_key].
            agent_key: "agent_0" or "agent_1".
            step_count: current cumulative environment step (for dynamic ratio).

        Returns:
            (num_envs,) float32 reward.
        """
        sparse_r = rewards[agent_key]

        if self.mode == "sparse":
            return sparse_r

        shaped_r = infos["shaped_reward"][agent_key]
        ratio = jnp.minimum(step_count / self.warmup_steps, self.final_ratio)
        combined = ratio * sparse_r + (1.0 - ratio) * shaped_r
        return self.magnifier * combined
