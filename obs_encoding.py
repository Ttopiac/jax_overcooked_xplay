"""Observation encoding utilities for Overcooked.

Provides two encoding modes:
  - "full":       Raw grid obs, flattened and padded to max layout size.
  - "egocentric": 7x7 window centered on agent, rotated so agent faces SOUTH.
                  Fixed (26, 7, 7) = 1274-dim across ALL layouts.

Usage:
    encoder = ObsEncoder(mode="egocentric")        # or "full"
    flat_obs = encoder.encode(obs, state, agent_idx, num_envs)
"""

import jax
import jax.numpy as jnp
from functools import partial


# Direction index mapping (from JaxMARL):
#   0 = NORTH (0, -1)
#   1 = SOUTH (0,  1)
#   2 = EAST  (1,  0)
#   3 = WEST  (-1, 0)
# Original multiHRI rotates so agent always faces SOUTH.
# Rotation amounts (k for jnp.rot90 on axes (1,2)):
#   SOUTH (1): k=0 (no rotation)
#   NORTH (0): k=2 (180 degrees)
#   EAST  (2): k=3 (270 degrees, equiv to -90)
#   WEST  (3): k=1 (90 degrees)
_ROT_K = jnp.array([2, 0, 3, 1], dtype=jnp.int32)  # indexed by dir_idx

EGO_SIZE = 7
EGO_CHANNELS = 26  # JaxMARL's channel count
EGO_FLAT_DIM = EGO_CHANNELS * EGO_SIZE * EGO_SIZE  # 1274


def _egocentric_single(obs, agent_pos, agent_dir_idx):
    """Convert one (H, W, C) observation to egocentric (C, EGO_SIZE, EGO_SIZE).

    Args:
        obs: (H, W, C) full grid observation (uint8).
        agent_pos: (2,) agent position (row, col).
        agent_dir_idx: scalar, direction index (0-3).

    Returns:
        (C, EGO_SIZE, EGO_SIZE) float32 egocentric observation.
    """
    C = obs.shape[2]
    pad = EGO_SIZE // 2  # 3

    # Transpose to (C, H, W)
    obs_chw = obs.transpose(2, 0, 1).astype(jnp.float32)

    # Pad spatial dims
    padded = jnp.pad(obs_chw, ((0, 0), (pad, pad), (pad, pad)))

    # Extract 7x7 window centered on agent (agent_pos is already in grid coords)
    row = jnp.int32(agent_pos[0])
    col = jnp.int32(agent_pos[1])
    window = jax.lax.dynamic_slice(padded, (jnp.int32(0), row, col), (C, EGO_SIZE, EGO_SIZE))

    # Rotate so agent always faces SOUTH
    k = _ROT_K[agent_dir_idx]
    # jnp.rot90 doesn't accept traced k, so use lax.switch
    window = jax.lax.switch(
        k,
        [
            lambda w: w,                             # k=0: no rotation
            lambda w: jnp.rot90(w, k=1, axes=(1, 2)),  # k=1: 90 degrees
            lambda w: jnp.rot90(w, k=2, axes=(1, 2)),  # k=2: 180 degrees
            lambda w: jnp.rot90(w, k=3, axes=(1, 2)),  # k=3: 270 degrees
        ],
        window,
    )

    return window


def egocentric_encode(obs, state, agent_idx, num_envs):
    """Encode observations as egocentric for a batch of environments.

    Args:
        obs: dict with agent keys, each value shape (num_envs, H, W, C).
        state: JaxMARL env state (vmapped, so fields have leading num_envs dim).
        agent_idx: 0 or 1, which agent to encode for.
        num_envs: number of parallel environments.

    Returns:
        (num_envs, EGO_FLAT_DIM) float32 flattened egocentric observations.
    """
    agent_key = f"agent_{agent_idx}"
    raw_obs = obs[agent_key]  # (num_envs, H, W, C)
    positions = state.agent_pos[:, agent_idx, :]   # (num_envs, 2)
    dir_idxs = state.agent_dir_idx[:, agent_idx]   # (num_envs,)

    # vmap over envs
    ego_obs = jax.vmap(_egocentric_single)(raw_obs, positions, dir_idxs)
    # ego_obs: (num_envs, C, EGO_SIZE, EGO_SIZE)

    return ego_obs.reshape(num_envs, -1)  # (num_envs, EGO_FLAT_DIM)


def full_grid_encode(obs, state, agent_idx, num_envs, max_obs_dim):
    """Encode observations as full grid, flattened and padded.

    Args:
        obs: dict with agent keys.
        state: unused for full grid encoding.
        agent_idx: 0 or 1.
        num_envs: number of parallel environments.
        max_obs_dim: target flat dimension (pad to this size).

    Returns:
        (num_envs, max_obs_dim) float32 flattened observations.
    """
    agent_key = f"agent_{agent_idx}"
    raw_obs = obs[agent_key]  # (num_envs, H, W, C)
    flat = raw_obs.reshape(num_envs, -1).astype(jnp.float32)
    pad_width = max_obs_dim - flat.shape[1]
    if pad_width > 0:
        flat = jnp.pad(flat, ((0, 0), (0, pad_width)))
    return flat


class ObsEncoder:
    """Configurable observation encoder.

    Args:
        mode: "egocentric" or "full".
        max_obs_dim: only used for "full" mode (pad target).
    """

    def __init__(self, mode="egocentric", max_obs_dim=None):
        assert mode in ("egocentric", "full"), f"Unknown encoding mode: {mode}"
        self.mode = mode
        self.max_obs_dim = max_obs_dim

        if mode == "egocentric":
            self.obs_flat_dim = EGO_FLAT_DIM
        else:
            assert max_obs_dim is not None, "max_obs_dim required for full mode"
            self.obs_flat_dim = max_obs_dim

    def encode(self, obs, state, agent_idx, num_envs):
        """Encode observations.

        Args:
            obs: dict with agent keys.
            state: JaxMARL env state.
            agent_idx: 0 or 1.
            num_envs: number of parallel environments.

        Returns:
            (num_envs, self.obs_flat_dim) float32.
        """
        if self.mode == "egocentric":
            return egocentric_encode(obs, state, agent_idx, num_envs)
        else:
            return full_grid_encode(obs, state, agent_idx, num_envs, self.max_obs_dim)
