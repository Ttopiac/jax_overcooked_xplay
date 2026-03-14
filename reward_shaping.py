"""Reward shaping utilities for Overcooked.

Two modes:
  - "sparse":  Sparse delivery reward only (from JaxMARL env).
  - "shaped":  sparse + shaped (Carroll et al. 2019 style).

Usage:
    shaper = RewardShaper(mode="shaped")
    reward = shaper.compute(rewards, infos, agent_key)
"""

import jax.numpy as jnp


class RewardShaper:
    """Configurable reward computation.

    Args:
        mode: "sparse" or "shaped".
              "sparse" = JaxMARL default (delivery reward only).
              "shaped" = Carroll et al. (sparse + shaped, simple sum).
    """

    def __init__(self, mode="shaped"):
        assert mode in ("sparse", "shaped"), f"Unknown reward mode: {mode}"
        self.mode = mode

    def compute(self, rewards, infos, agent_key, step_count=0):
        """Compute reward for one agent across a batch of environments.

        Args:
            rewards: dict from env.step, {"agent_0": (N,), "agent_1": (N,)}.
            infos: dict from env.step, with infos["shaped_reward"][agent_key].
            agent_key: "agent_0" or "agent_1".
            step_count: unused, kept for API compatibility.

        Returns:
            (N,) float32 reward.
        """
        sparse_r = rewards[agent_key]

        if self.mode == "sparse":
            return sparse_r

        shaped_r = infos["shaped_reward"][agent_key]
        return sparse_r + shaped_r
