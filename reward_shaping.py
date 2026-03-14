"""Reward shaping utilities for Overcooked.

Matches the multiHRI Learner reward computation:
  - "sparse":     Sparse delivery reward only (group sum).
  - "originaler": group_sparse * ratio + individual_shaped * (1 - ratio).
                  NO magnifier. Default learner type in original code.
  - "solo":       magnifier * (group_sparse * ratio + individual_shaped * (1 - ratio)).
                  Same as originaler but with magnifier (like SoloWorker).

Ratio ramps from 0 → final_ratio over warmup_steps (default 0 → 0.5 over 10M).

Usage:
    shaper = RewardShaper(mode="originaler")
    reward = shaper.compute(rewards, infos, agent_key, step_count)
"""

import jax.numpy as jnp


class RewardShaper:
    """Configurable reward computation matching multiHRI Learner classes.

    Args:
        mode: "sparse", "originaler", or "solo".
        magnifier: Reward scaling (only used by "solo" mode, default 3.0).
        final_ratio: Target sparse ratio at end of warmup (default 0.5).
        warmup_steps: Env steps over which ratio ramps 0 → final_ratio
                      (default 10_000_000).
    """

    def __init__(
        self,
        mode="originaler",
        magnifier=3.0,
        final_ratio=0.5,
        warmup_steps=10_000_000,
    ):
        assert mode in ("sparse", "originaler", "solo"), f"Unknown reward mode: {mode}"
        self.mode = mode
        self.magnifier = magnifier
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def compute(self, rewards, infos, agent_key, step_count=0):
        """Compute reward for one agent across a batch of environments.

        Matches multiHRI's Originaler learner:
          personal_reward = group_sparse * ratio + individual_shaped * (1 - ratio)

        Args:
            rewards: dict from env.step, {"agent_0": (N,), "agent_1": (N,)}.
            infos: dict from env.step, with infos["shaped_reward"][agent_key].
            agent_key: "agent_0" or "agent_1".
            step_count: cumulative environment step (for dynamic ratio).

        Returns:
            (N,) float32 reward.
        """
        # Group sparse = sum of both agents' sparse rewards
        group_sparse_r = rewards["agent_0"] + rewards["agent_1"]

        if self.mode == "sparse":
            return group_sparse_r

        # Individual shaped reward for this agent
        individual_shaped_r = infos["shaped_reward"][agent_key]

        ratio = jnp.minimum(step_count / self.warmup_steps, self.final_ratio)
        personal_reward = group_sparse_r * ratio + individual_shaped_r * (1.0 - ratio)

        if self.mode == "originaler":
            return personal_reward
        else:  # "solo"
            return self.magnifier * personal_reward
