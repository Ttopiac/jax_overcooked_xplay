"""Observation encoding utilities for Overcooked.

Encoding modes:
  - "egocentric": 7x7 window centered on agent, rotated so agent faces SOUTH.
                  Fixed (26, 7, 7) = 1274-dim across ALL layouts and any N agents.
                  All N-1 teammates are overlaid in shared channels.

Future:
  - "raw": Rendered grid image (needs CNN, agent-agnostic, scales to any N).

Usage:
    encoder = ObsEncoder(mode="egocentric")
    flat_obs = encoder.encode(obs, state, agent_idx, num_envs)
"""

import jax
import jax.numpy as jnp


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

# Full grid observation: pad all layouts to max size (9w x 5h)
FULL_MAX_W = 9
FULL_MAX_H = 5
FULL_CHANNELS = 26
FULL_OBS_SHAPE = (FULL_CHANNELS, FULL_MAX_H, FULL_MAX_W)  # (26, 5, 9) CHW format


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


def full_grid_encode(obs, state, agent_idx, num_envs):
    """Encode observations as full padded grid images (CHW format).

    Returns the full grid observation for the specified agent, padded to
    (FULL_CHANNELS, FULL_MAX_H, FULL_MAX_W) for consistent shape across layouts.

    Args:
        obs: dict with agent keys, each value shape (num_envs, W, H, C).
        state: JaxMARL env state.
        agent_idx: int, which agent to encode for.
        num_envs: number of parallel environments.

    Returns:
        (num_envs, FULL_CHANNELS, FULL_MAX_H, FULL_MAX_W) float32.
    """
    agent_key = f"agent_{agent_idx}"
    raw_obs = obs[agent_key]  # (num_envs, H, W, C)

    # Transpose to CHW: (num_envs, C, H, W)
    obs_chw = raw_obs.transpose(0, 3, 1, 2).astype(jnp.float32)  # (num_envs, C, H, W)

    # Pad to max size: (C, H, W) -> (C, FULL_MAX_H, FULL_MAX_W)
    c, h, w = obs_chw.shape[1], obs_chw.shape[2], obs_chw.shape[3]
    pad_h = FULL_MAX_H - h
    pad_w = FULL_MAX_W - w
    if pad_h > 0 or pad_w > 0:
        obs_chw = jnp.pad(obs_chw, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

    return obs_chw  # (num_envs, FULL_CHANNELS, FULL_MAX_H, FULL_MAX_W)


class ObsEncoder:
    """Configurable observation encoder.

    Args:
        mode: "egocentric" or "full".
            - egocentric: 7x7 crop centered on agent, rotated. Output: (num_envs, 1274).
            - full: full padded grid image. Output: (num_envs, 26, 5, 9).
    """

    def __init__(self, mode="egocentric"):
        assert mode in ("egocentric", "full"), f"Unknown encoding mode: {mode}"
        self.mode = mode
        if mode == "egocentric":
            self.obs_flat_dim = EGO_FLAT_DIM
            self.obs_shape = None  # flat vector
        else:
            self.obs_flat_dim = FULL_CHANNELS * FULL_MAX_H * FULL_MAX_W  # 1170
            self.obs_shape = FULL_OBS_SHAPE  # (26, 5, 9) for CNN

    def encode(self, obs, state, agent_idx, num_envs):
        """Encode observations.

        Args:
            obs: dict with agent keys.
            state: JaxMARL env state.
            agent_idx: int, which agent to encode for.
            num_envs: number of parallel environments.

        Returns:
            If mode="egocentric": (num_envs, 1274) float32 flat vector.
            If mode="full": (num_envs, 26, 5, 9) float32 CHW image.
        """
        if self.mode == "egocentric":
            return egocentric_encode(obs, state, agent_idx, num_envs)
        else:
            return full_grid_encode(obs, state, agent_idx, num_envs)
