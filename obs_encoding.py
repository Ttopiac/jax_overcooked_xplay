"""Observation encoding utilities for Overcooked.

Encoding modes:
  - "egocentric": 7x7 window centered on agent, rotated so agent faces SOUTH.
                  Fixed (26, 7, 7) = 1274-dim across ALL layouts and any N agents.
                  All N-1 teammates are overlaid in shared channels.
  - "full": Full padded grid, (26, 5, 9) CHW feature channels.
  - "pixel": Pre-rendered sprite images matching the human visualizer.
             (3, 160, 288) CHW float32 in [0, 1]. Uses IMPALA CNN.

Usage:
    encoder = ObsEncoder(mode="egocentric")
    flat_obs = encoder.encode(obs, state, agent_idx, num_envs)
"""

import math

import jax
import jax.numpy as jnp
import numpy as np


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

# --- Pixel observation constants ---
# Tile resolution: 16×16 sprites rendered from the same visualizer as humans see.
# Stored as palette-indexed uint8 (1 byte/pixel) instead of float32 RGB (12 bytes/pixel)
# for 12× VRAM savings, enabling batch_size=8 with 42 envs/layout.
TILE_SIZE = 16
PIXEL_MAX_H = FULL_MAX_H * TILE_SIZE  # 5 * 16 = 80
PIXEL_MAX_W = FULL_MAX_W * TILE_SIZE  # 9 * 16 = 144
# Palette-indexed obs: (1, H, W) uint8 indices. CNN decodes to RGB via palette lookup.
PIXEL_OBS_SHAPE = (1, PIXEL_MAX_H, PIXEL_MAX_W)

# Atlas layout:
#   [0]: floor (empty/unseen)
#   [1]: wall
#   [2]: onion_pile
#   [3]: plate_pile
#   [4]: goal
#   [5]: onion (on counter)
#   [6]: plate (on counter)
#   [7]: dish (completed soup on counter)
#   [8-31]: pot states 0-23  (index = 8 + pot_status)
#   [32-63]: agent variants  (index = 32 + color*16 + dir*4 + inv_bucket)
#     color: 0=agent_0(red), 1=agent_1(green)
#     dir: 0-3 (N/S/E/W)
#     inv_bucket: 0=empty, 1=onion, 2=plate, 3=dish
_POT_BASE = 8
_AGENT_BASE = 32
_NUM_ATLAS_TILES = 64

# Object type -> base tile index (for non-pot, non-agent objects)
_OBJ_TILE_BASE_NP = np.zeros(11, dtype=np.int32)
_OBJ_TILE_BASE_NP[0] = 0   # unseen -> floor
_OBJ_TILE_BASE_NP[1] = 0   # empty -> floor
_OBJ_TILE_BASE_NP[2] = 1   # wall
_OBJ_TILE_BASE_NP[3] = 5   # onion
_OBJ_TILE_BASE_NP[4] = 2   # onion_pile
_OBJ_TILE_BASE_NP[5] = 6   # plate
_OBJ_TILE_BASE_NP[6] = 3   # plate_pile
_OBJ_TILE_BASE_NP[7] = 4   # goal
_OBJ_TILE_BASE_NP[8] = 8   # pot (base, add status)
_OBJ_TILE_BASE_NP[9] = 7   # dish
_OBJ_TILE_BASE_NP[10] = 32  # agent (base, add offset)

# Inventory item -> bucket index (0-3)
_INV_BUCKET_NP = np.zeros(11, dtype=np.int32)
_INV_BUCKET_NP[1] = 0  # empty
_INV_BUCKET_NP[3] = 1  # onion
_INV_BUCKET_NP[5] = 2  # plate
_INV_BUCKET_NP[9] = 3  # dish

_MAZE_MAP_PADDING = 4  # agent_view_size(5) - 1


def _build_sprite_atlas(tile_size=TILE_SIZE):
    """Pre-render all tile sprites using the JaxMARL visualizer rendering primitives.

    Returns:
        numpy array of shape (_NUM_ATLAS_TILES, tile_size, tile_size, 3) uint8
    """
    from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
    from jaxmarl.viz import grid_rendering as rendering
    from overcooked_env.common import OBJECT_TO_INDEX, COLOR_TO_INDEX, COLORS

    subdivs = 3
    atlas = np.zeros((_NUM_ATLAS_TILES, tile_size, tile_size, 3), dtype=np.uint8)

    def _base_img():
        img = np.zeros((tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)
        rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
        return img

    def _render_and_downsample(obj_array):
        img = _base_img()
        OvercookedVisualizer._render_obj(obj_array, img)
        return rendering.downsample(img, subdivs).astype(np.uint8)

    def _render_agent(color_idx, dir_idx, inv_item):
        img = _base_img()
        obj = np.array([OBJECT_TO_INDEX['agent'], color_idx, dir_idx])
        OvercookedVisualizer._render_obj(obj, img)
        if inv_item != OBJECT_TO_INDEX['empty']:
            inv_obj = np.array([inv_item, -1, 0])
            OvercookedVisualizer._render_inv(inv_obj, img)
        return rendering.downsample(img, subdivs).astype(np.uint8)

    # [0]: floor
    atlas[0] = rendering.downsample(_base_img(), subdivs).astype(np.uint8)
    # [1]: wall
    atlas[1] = _render_and_downsample(np.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0]))
    # [2]: onion_pile
    atlas[2] = _render_and_downsample(np.array([OBJECT_TO_INDEX['onion_pile'], COLOR_TO_INDEX['yellow'], 0]))
    # [3]: plate_pile
    atlas[3] = _render_and_downsample(np.array([OBJECT_TO_INDEX['plate_pile'], COLOR_TO_INDEX['white'], 0]))
    # [4]: goal
    atlas[4] = _render_and_downsample(np.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0]))
    # [5]: onion
    atlas[5] = _render_and_downsample(np.array([OBJECT_TO_INDEX['onion'], COLOR_TO_INDEX['yellow'], 0]))
    # [6]: plate
    atlas[6] = _render_and_downsample(np.array([OBJECT_TO_INDEX['plate'], COLOR_TO_INDEX['white'], 0]))
    # [7]: dish
    atlas[7] = _render_and_downsample(np.array([OBJECT_TO_INDEX['dish'], COLOR_TO_INDEX['white'], 0]))
    # [8-31]: pot states (status 0-23)
    for status in range(24):
        atlas[_POT_BASE + status] = _render_and_downsample(
            np.array([OBJECT_TO_INDEX['pot'], COLOR_TO_INDEX['black'], status]))
    # [32-63]: agent variants
    inv_items = [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                 OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']]
    for color in range(2):
        for d in range(4):
            for inv_b, inv_item in enumerate(inv_items):
                atlas[_AGENT_BASE + color * 16 + d * 4 + inv_b] = _render_agent(color, d, inv_item)

    return atlas


def _build_palette_atlas(tile_size=TILE_SIZE):
    """Build a color palette and palette-indexed sprite atlas.

    Instead of storing RGB per pixel (3 bytes), we discover the unique colors
    across all sprites and store a single palette index per pixel (1 byte).
    This gives 3× memory savings over uint8 RGB and 12× over float32 RGB,
    enabling batch_size=8 with 42 envs/layout on a 24GB GPU.

    Returns:
        palette_np: (num_colors, 3) uint8 — the color lookup table
        indexed_atlas: (_NUM_ATLAS_TILES, tile_size, tile_size) uint8 — palette indices
    """
    atlas_rgb = _build_sprite_atlas(tile_size)  # (_NUM_ATLAS_TILES, T, T, 3) uint8

    # Discover unique colors across all tiles
    all_pixels = atlas_rgb.reshape(-1, 3)
    unique_colors, inverse = np.unique(all_pixels, axis=0, return_inverse=True)
    num_colors = len(unique_colors)
    assert num_colors <= 256, f"Too many unique colors ({num_colors}) for uint8 palette"

    # Build palette-indexed atlas
    indexed_atlas = inverse.reshape(atlas_rgb.shape[:-1]).astype(np.uint8)

    return unique_colors.astype(np.uint8), indexed_atlas


class SpriteAtlas:
    """Pre-rendered sprite atlas with palette indexing for memory-efficient storage.

    Palette indexing stores 1 byte per pixel (palette index) instead of 3 bytes
    (RGB uint8) or 12 bytes (RGB float32). The CNN decodes palette indices to
    float32 RGB via a single gather operation at forward-pass time.

    Memory comparison for 16×16 tiles, 42 envs/layout, 500 steps, batch_size=8:
        float32 RGB: 58 GB trajectory — does not fit 24GB GPU
        uint8 RGB:   19 GB trajectory — tight
        palette:      6 GB trajectory — comfortable fit

    Built once at init using OvercookedVisualizer rendering primitives.
    """

    def __init__(self, tile_size=TILE_SIZE):
        palette_np, indexed_atlas_np = _build_palette_atlas(tile_size)
        # Palette: (num_colors, 3) float32 normalized — used for decode at forward pass
        self.palette = jnp.array(palette_np, dtype=jnp.float32) / 255.0
        # Indexed atlas: (_NUM_ATLAS_TILES, T, T) uint8 palette indices
        self.indexed_atlas = jnp.array(indexed_atlas_np, dtype=jnp.uint8)
        self.num_colors = len(palette_np)
        self.tile_size = tile_size
        self.obj_tile_base = jnp.array(_OBJ_TILE_BASE_NP)
        self.inv_to_bucket = jnp.array(_INV_BUCKET_NP)


def pixel_encode(obs, state, agent_idx, num_envs, atlas):
    """Encode game state as palette-indexed pixel images.

    Returns uint8 palette indices (1 byte/pixel) instead of float32 RGB
    (12 bytes/pixel). The CNN decodes to RGB via palette lookup at forward time.

    Reads directly from state.maze_map (not the JaxMARL obs channels).
    All runtime operations are pure JAX: indexing, reshape, pad.

    Args:
        obs: unused (pixel mode reads from state.maze_map).
        state: env state (vmapped), has maze_map and agent_inv.
        agent_idx: unused (pixel view is the same for both agents).
        num_envs: batch size.
        atlas: SpriteAtlas instance.

    Returns:
        (num_envs, 1, PIXEL_MAX_H, PIXEL_MAX_W) uint8 palette indices.
    """
    p = _MAZE_MAP_PADDING
    maze_map = state.maze_map[:, p:-p, p:-p, :]  # (num_envs, H, W, 3)
    H, W = maze_map.shape[1], maze_map.shape[2]

    obj_types = maze_map[:, :, :, 0].astype(jnp.int32)
    color_idxs = maze_map[:, :, :, 1].astype(jnp.int32)
    statuses = maze_map[:, :, :, 2].astype(jnp.int32)

    # Agent inventory overlay: map each agent cell to its inventory item
    num_agents = state.agent_inv.shape[1]
    agent_inv_grid = jnp.zeros_like(obj_types)
    for k in range(num_agents):
        is_agent_k = (obj_types == 10) & (color_idxs == k)
        inv_k = state.agent_inv[:, k].astype(jnp.int32)
        agent_inv_grid = jnp.where(is_agent_k, inv_k[:, None, None], agent_inv_grid)

    # Tile index computation
    base_tiles = atlas.obj_tile_base[obj_types]
    pot_tiles = _POT_BASE + statuses
    inv_buckets = atlas.inv_to_bucket[agent_inv_grid]
    agent_tiles = _AGENT_BASE + color_idxs * 16 + statuses * 4 + inv_buckets

    is_pot = (obj_types == 8)
    is_agent = (obj_types == 10)
    tile_indices = jnp.where(is_agent, agent_tiles, jnp.where(is_pot, pot_tiles, base_tiles))

    # Gather palette-indexed tiles: (num_envs, H, W) -> (num_envs, H, W, T, T)
    T = atlas.tile_size
    tiles = atlas.indexed_atlas[tile_indices]  # uint8 palette indices

    # Merge tile pixels into full image
    # (num_envs, H, W, T, T) -> (num_envs, H, T, W, T) -> (num_envs, H*T, W*T)
    img = tiles.transpose(0, 1, 3, 2, 4).reshape(num_envs, H * T, W * T)

    # Pad smaller layouts to max pixel size
    pad_h = PIXEL_MAX_H - H * T
    pad_w = PIXEL_MAX_W - W * T
    if pad_h > 0 or pad_w > 0:
        img = jnp.pad(img, ((0, 0), (0, pad_h), (0, pad_w)))

    # Add channel dim: (num_envs, 1, H, W) — single-channel palette indices
    return img[:, None, :, :]


def palette_decode(palette, obs_indices):
    """Decode palette-indexed obs to float32 RGB for CNN input.

    Called inside the network forward pass. Converts compact uint8 palette
    indices (1 byte/pixel) to float32 RGB (12 bytes/pixel) just-in-time,
    keeping the trajectory storage 12× smaller.

    Args:
        palette: (num_colors, 3) float32 in [0, 1] — color lookup table.
        obs_indices: (..., 1, H, W) uint8 palette indices.

    Returns:
        (..., 3, H, W) float32 RGB image in [0, 1], ready for CNN.
    """
    # Remove channel dim: (..., 1, H, W) -> (..., H, W)
    idx = obs_indices[..., 0, :, :]
    # Gather RGB: (..., H, W) -> (..., H, W, 3)
    rgb = palette[idx.astype(jnp.int32)]
    # Transpose to CHW: (..., H, W, 3) -> (..., 3, H, W)
    return jnp.moveaxis(rgb, -1, -3)


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
        mode: "egocentric", "full", or "pixel".
            - egocentric: 7x7 crop centered on agent, rotated. Output: (num_envs, 1274).
            - full: full padded grid image. Output: (num_envs, 26, 5, 9).
            - pixel: palette-indexed 16×16 sprite images matching the human visualizer.
                     Output: (num_envs, 1, 80, 144) uint8.
                     CNN decodes to RGB via palette lookup at forward-pass time.
    """

    def __init__(self, mode="egocentric"):
        assert mode in ("egocentric", "full", "pixel"), f"Unknown encoding mode: {mode}"
        self.mode = mode
        if mode == "egocentric":
            self.obs_flat_dim = EGO_FLAT_DIM
            self.obs_shape = None  # flat vector
        elif mode == "full":
            self.obs_flat_dim = FULL_CHANNELS * FULL_MAX_H * FULL_MAX_W  # 1170
            self.obs_shape = FULL_OBS_SHAPE  # (26, 5, 9) for CNN
        else:  # pixel
            self.atlas = SpriteAtlas()
            self.obs_flat_dim = PIXEL_MAX_H * PIXEL_MAX_W  # 11520
            self.obs_shape = PIXEL_OBS_SHAPE  # (1, 80, 144) palette-indexed

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
            If mode="pixel": (num_envs, 1, 80, 144) uint8 palette indices.
        """
        if self.mode == "egocentric":
            return egocentric_encode(obs, state, agent_idx, num_envs)
        elif self.mode == "full":
            return full_grid_encode(obs, state, agent_idx, num_envs)
        else:
            return pixel_encode(obs, state, agent_idx, num_envs, self.atlas)
