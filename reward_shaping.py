"""Reward shaping utilities for Overcooked.

Matches Carroll et al. (2019) reward computation:
  reward = sparse + reward_shaping_factor * dense

Where reward_shaping_factor linearly anneals from initial value to 0
over reward_shaping_horizon timesteps.

Two modes:
  - "sparse":  Sparse delivery reward only.
  - "shaped":  Carroll-style: sparse + factor * shaped.

Note: Carroll uses potential-based dense rewards (phi) by default, but
JaxMARL only provides event-based shaped rewards (onion in pot +3, etc.).
We use JaxMARL's shaped rewards as the dense signal.

Usage:
    shaper = RewardShaper(mode="shaped")
    reward = shaper.compute(rewards, infos, agent_key, step_count)
"""

import jax.numpy as jnp


class RewardShaper:
    """Configurable reward computation matching Carroll et al. (2019).

    Carroll's formula:
        reward = sparse + reward_shaping_factor * dense
        factor anneals: initial_factor -> 0 over horizon steps

    Args:
        mode: "sparse" or "shaped".
        reward_shaping_factor: Initial multiplier on dense/shaped reward (default 1.0).
        reward_shaping_horizon: Timesteps over which factor anneals to 0.
            inf = no annealing (factor stays constant). Default inf.
    """

    def __init__(
        self,
        mode="shaped",
        reward_shaping_factor=1.0,
        reward_shaping_horizon=float("inf"),
    ):
        assert mode in ("sparse", "shaped"), f"Unknown reward mode: {mode}"
        self.mode = mode
        self.initial_factor = reward_shaping_factor
        self.horizon = reward_shaping_horizon

    def _get_factor(self, step_count):
        """Linear annealing: initial_factor -> 0 over horizon steps."""
        if self.horizon == float("inf") or self.horizon == 0:
            return self.initial_factor
        fraction = jnp.maximum(1.0 - step_count / self.horizon, 0.0)
        return fraction * self.initial_factor

    def compute(self, rewards, infos, agent_key, step_count=0):
        """Compute reward for one agent across a batch of environments.

        Carroll's formula: sparse + factor * dense
        Both agents receive the same sparse reward (team reward).
        Each agent gets their own dense/shaped reward.

        Args:
            rewards: dict from env.step, {"agent_0": (N,), "agent_1": (N,)}.
            infos: dict from env.step, with infos["shaped_reward"][agent_key].
            agent_key: "agent_0" or "agent_1".
            step_count: cumulative environment step (for factor annealing).

        Returns:
            (N,) float32 reward.
        """
        sparse_r = rewards[agent_key]

        if self.mode == "sparse":
            return sparse_r

        dense_r = infos["shaped_reward"][agent_key]
        factor = self._get_factor(step_count)
        return sparse_r + factor * dense_r
