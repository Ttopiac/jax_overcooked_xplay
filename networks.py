"""Actor-Critic network for JAX-based Overcooked training."""

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
