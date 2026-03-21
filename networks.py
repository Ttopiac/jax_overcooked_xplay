"""Actor-Critic networks for JAX-based Overcooked training.

Two variants:
  - ActorCritic: MLP-based, takes flat observation vectors.
  - CNNActorCritic: CNN-based, takes CHW image observations (e.g. 26x5x9 grid).
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

    IMPALA-style small CNN suitable for grid-world observations.
    Input: (C, H, W) image (e.g. 26x5x9 for full grid, or batched).

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
