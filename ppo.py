"""Shared PPO utilities: transitions, GAE, and update functions."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray


def compute_gae(traj_batch: Transition, last_val, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    Args:
        traj_batch: Transition with fields of shape (T, B, ...).
        last_val: Bootstrapped value, shape (B,).
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        advantages: shape (T, B)
        returns: shape (T, B)
    """
    def _get_advantages(carry, transition):
        gae, next_value = carry
        delta = (
            transition.reward
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
    )
    returns = advantages + traj_batch.value
    return advantages, returns


def ppo_update(train_state, traj_batch, advantages, returns, config, rng):
    """Run multiple PPO epochs over the trajectory batch.

    Args:
        train_state: Flax TrainState.
        traj_batch: Transition with fields of shape (T*B, ...) (already flattened).
        advantages: shape (T*B,).
        returns: shape (T*B,).
        config: dict with clip_eps, vf_coef, ent_coef, num_epochs, minibatch_size.
        rng: JAX random key.

    Returns:
        Updated train_state, loss_info dict.
    """
    batch_size = traj_batch.obs.shape[0]
    num_minibatches = batch_size // config["minibatch_size"]

    def _update_epoch(carry, epoch_rng):
        train_state = carry

        # Shuffle
        perm = jax.random.permutation(epoch_rng, batch_size)
        batch = jax.tree.map(lambda x: x[perm], traj_batch)
        shuffled_adv = advantages[perm]
        shuffled_ret = returns[perm]

        # Reshape into minibatches
        usable = num_minibatches * config["minibatch_size"]
        mb_batch = jax.tree.map(
            lambda x: x[:usable].reshape(num_minibatches, config["minibatch_size"], *x.shape[1:]),
            batch,
        )
        mb_adv = shuffled_adv[:usable].reshape(num_minibatches, config["minibatch_size"])
        mb_ret = shuffled_ret[:usable].reshape(num_minibatches, config["minibatch_size"])

        def _update_minibatch(train_state, data):
            mb, mb_advantages, mb_returns = data

            def _loss_fn(params):
                pi, value = train_state.apply_fn(params, mb.obs)
                log_prob = pi.log_prob(mb.action)
                entropy = pi.entropy().mean()

                # Normalize advantages
                adv = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PPO clipped objective
                ratio = jnp.exp(log_prob - mb.log_prob)
                loss_pi1 = ratio * adv
                loss_pi2 = jnp.clip(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"]) * adv
                loss_actor = -jnp.minimum(loss_pi1, loss_pi2).mean()

                # Value loss
                loss_critic = 0.5 * ((value - mb_returns) ** 2).mean()

                total_loss = loss_actor + config["vf_coef"] * loss_critic - config["ent_coef"] * entropy
                return total_loss, {"actor_loss": loss_actor, "critic_loss": loss_critic, "entropy": entropy}

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, aux), grads = grad_fn(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            aux["total_loss"] = total_loss
            return train_state, aux

        train_state, loss_info = jax.lax.scan(
            _update_minibatch, train_state, (mb_batch, mb_adv, mb_ret)
        )
        return train_state, loss_info

    epoch_rngs = jax.random.split(rng, config["num_epochs"])
    train_state, loss_info = jax.lax.scan(_update_epoch, train_state, epoch_rngs)

    # Average loss over epochs and minibatches
    loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
    return train_state, loss_info


def create_train_state(rng, network, obs_shape, lr, max_grad_norm=0.5):
    """Initialize a TrainState with Adam optimizer and gradient clipping."""
    dummy_obs = jnp.zeros(obs_shape)
    params = network.init(rng, dummy_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)
