"""Actor-Critic networks for JAX-based Overcooked training.

Three variants:
  - ActorCritic: MLP-based, takes flat observation vectors.
  - CNNActorCritic: Simple CNN for small grid observations (e.g. 26x5x9).
  - IMPALACNNActorCritic: 4-block IMPALA ResNet for pixel observations (3x160x288).
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
from typing import Sequence
import distrax


class ActorCritic(nn.Module):
    """Actor-Critic with separate MLP heads for policy and value."""
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        # Actor head
        actor = x
        for dim in self.hidden_dims:
            actor = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)))(actor)
            actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
        pi = distrax.Categorical(logits=logits)

        # Critic head
        critic = x
        for dim in self.hidden_dims:
            critic = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)))(critic)
            critic = nn.relu(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(critic)

        return pi, jnp.squeeze(value, axis=-1)


class CNNActorCritic(nn.Module):
    """Actor-Critic with shared CNN encoder and separate MLP heads.

    Simple CNN suitable for small grid-world observations (e.g. 26x5x9).
    Input: (C, H, W) image.

    Architecture:
      Conv(32, 3x3) -> ReLU -> Conv(64, 3x3) -> ReLU -> Conv(64, 3x3) -> ReLU
      -> Flatten -> Dense(256) -> ReLU -> Actor/Critic heads
    """
    action_dim: int
    cnn_channels: Sequence[int] = (32, 64, 64)
    mlp_dim: int = 256

    @nn.compact
    def __call__(self, x):
        # x shape: (..., C, H, W) — transpose to (..., H, W, C) for flax Conv
        x = jnp.moveaxis(x, -3, -1)  # (..., H, W, C)

        # Shared CNN encoder
        for ch in self.cnn_channels:
            x = nn.Conv(ch, kernel_size=(3, 3), padding="SAME",
                        kernel_init=orthogonal(jnp.sqrt(2)))(x)
            x = nn.relu(x)

        # Flatten spatial dims
        batch_shape = x.shape[:-3]
        x = x.reshape(*batch_shape, -1)

        # Shared dense
        x = nn.Dense(self.mlp_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)

        # Actor head
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)

        # Critic head
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)

        return pi, jnp.squeeze(value, axis=-1)


class IMPALACNNActorCritic(nn.Module):
    """IMPALA shallow CNN — 2 conv layers, same as IMPALA paper's shallow model.

    Designed for pixel observations with palette-indexed storage.
    Accepts (1, H, W) uint8 palette indices, decodes to RGB, then:
      Conv(16, 8×8, stride=4) → ReLU → Conv(32, 4×4, stride=2) → ReLU
      → Flatten → Dense(256) → ReLU → Actor/Critic heads

    For 8×8 tiles (40×72 images):
      Palette decode: (1, 40, 72) uint8 → (3, 40, 72) float32
      Conv(16, 8×8, stride=4): (3, 40, 72) → (16, 10, 18)
      Conv(32, 4×4, stride=2): (16, 10, 18) → (32, 5, 9)
      Flatten: 32 × 5 × 9 = 1440 → Dense(256)
      ~375K params, 2 conv layers

    Architecture from: Espeholt et al., "IMPALA: Scalable Distributed Deep-RL
    with Importance Weighted Actor-Learner Architectures", ICML 2018 (Figure 3,
    shallow model). Handles free-position objects (no tile grid assumption),
    suitable for future sim-to-real transfer with camera observations.
    """
    action_dim: int
    mlp_dim: int = 256
    palette: jnp.ndarray = None  # (num_colors, 3) float32, set at construction

    @nn.compact
    def __call__(self, x):
        # Palette decode: (…, 1, H, W) uint8 -> (…, 3, H, W) float32
        if self.palette is not None:
            from obs_encoding import palette_decode
            x = palette_decode(self.palette, x)

        # x: (..., C, H, W) -> (..., H, W, C) for flax Conv
        x = jnp.moveaxis(x, -3, -1)
        batch_shape = x.shape[:-3]

        # IMPALA shallow: 2 conv layers with large kernels/strides
        x = nn.Conv(16, kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                     kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2), padding="VALID",
                     kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)

        # Flatten spatial dims
        x = x.reshape(*batch_shape, -1)

        # Dense
        x = nn.Dense(self.mlp_dim, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)

        # Actor head
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        pi = distrax.Categorical(logits=logits)

        # Critic head
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)

        return pi, jnp.squeeze(value, axis=-1)
