"""Maximum Entropy Population (MEP) training in JAX.

Maintains a population of N agents. Each training step:
1. Randomly select one agent to train (as ego, playing with a copy of itself).
2. Compute entropy bonus: -alpha * log(mean action probability across population).
3. Add bonus to the reward.
4. Update the selected agent via PPO.

The entropy bonus encourages behavioral diversity across the population.
"""

import argparse
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jaxmarl import make

from networks import ActorCritic
from ppo import Transition, compute_gae, ppo_update, create_train_state


def make_train(config):
    env = make("overcooked", layout=config["layout"])
    agents = env.agents

    sample_key = jax.random.PRNGKey(0)
    sample_obs, _ = env.reset(sample_key)
    obs_shape = sample_obs[agents[0]].shape
    obs_flat_dim = int(jnp.prod(jnp.array(obs_shape)))
    action_dim = env.action_space(agents[0]).n

    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])
    pop_size = config["pop_size"]

    def train(rng):
        # Initialize population: stack params along axis 0
        init_rngs = jax.random.split(rng, pop_size + 2)
        rng = init_rngs[0]
        env_rng = init_rngs[1]

        # Create stacked population params
        def _init_one(init_rng):
            dummy = jnp.zeros((obs_flat_dim,))
            return network.init(init_rng, dummy)

        all_params = jax.vmap(_init_one)(init_rngs[2 : 2 + pop_size])

        # Single optimizer (applied per-agent)
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(config["lr"], eps=1e-5),
        )
        all_opt_states = jax.vmap(tx.init)(all_params)

        # Reset environments
        env_rngs = jax.random.split(env_rng, config["num_envs"])
        obs, env_state = jax.vmap(env.reset)(env_rngs)

        def _compute_entropy_bonus(obs_flat, actions, ego_idx, all_params):
            """Compute MEP entropy bonus for the ego agent's actions.

            bonus = -alpha * log(mean probability of the taken action across all population members)
            """
            alpha = config["mep_alpha"]

            # Get action probabilities from all population members: (pop_size, batch, action_dim)
            def _get_probs(params):
                pi, _ = network.apply(params, obs_flat)
                return jax.nn.softmax(pi.logits, axis=-1)

            all_probs = jax.vmap(_get_probs)(all_params)  # (pop_size, batch, action_dim)

            # Mean probability across population for the taken actions
            # actions shape: (batch,)
            action_probs = jnp.take_along_axis(
                all_probs, actions[None, :, None].repeat(pop_size, axis=0), axis=-1
            ).squeeze(-1)  # (pop_size, batch)
            mean_prob = action_probs.mean(axis=0)  # (batch,)
            bonus = -alpha * jnp.log(mean_prob + 1e-8)
            return bonus

        def _update_step(runner_state, unused):
            all_params, all_opt_states, env_state, obs, rng = runner_state

            # Select ego agent
            rng, select_rng, act_rng, step_rng, update_rng = jax.random.split(rng, 5)
            ego_idx = jax.random.randint(select_rng, (), 0, pop_size)
            ego_params = jax.tree.map(lambda x: x[ego_idx], all_params)

            # --- Collect rollout (ego plays with itself, like SP) ---
            def _env_step(carry, unused):
                env_state, obs, rng = carry
                rng, a_rng0, a_rng1, s_rng = jax.random.split(rng, 4)

                obs_0 = obs[agents[0]].reshape(config["num_envs"], -1)
                obs_1 = obs[agents[1]].reshape(config["num_envs"], -1)

                pi_0, val_0 = network.apply(ego_params, obs_0)
                pi_1, _ = network.apply(ego_params, obs_1)

                a0 = pi_0.sample(seed=a_rng0)
                a1 = pi_1.sample(seed=a_rng1)
                lp0 = pi_0.log_prob(a0)

                actions = {agents[0]: a0, agents[1]: a1}
                step_rngs = jax.random.split(s_rng, config["num_envs"])
                next_obs, next_state, rewards, dones, infos = jax.vmap(env.step)(
                    step_rngs, env_state, actions
                )

                # Compute entropy bonus for ego agent's actions
                bonus = _compute_entropy_bonus(obs_0, a0, ego_idx, all_params)
                reward = rewards[agents[0]] + bonus
                done_all = dones["__all__"]

                trans = Transition(obs_0, a0, reward, done_all, lp0, val_0)
                return (next_state, next_obs, rng), trans

            carry, traj_batch = jax.lax.scan(
                _env_step, (env_state, obs, act_rng), None, length=config["num_steps"]
            )
            env_state, obs, _ = carry

            # Bootstrap
            obs_0 = obs[agents[0]].reshape(config["num_envs"], -1)
            _, last_val = network.apply(ego_params, obs_0)

            # GAE
            advantages, returns = compute_gae(
                traj_batch, last_val, config["gamma"], config["gae_lambda"]
            )

            # Flatten
            flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj_batch)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)

            # PPO update for ego agent only
            ego_opt_state = jax.tree.map(lambda x: x[ego_idx], all_opt_states)
            ego_train_state = TrainState(
                step=0,
                apply_fn=network.apply,
                params=ego_params,
                tx=tx,
                opt_state=ego_opt_state,
            )
            ego_train_state, loss_info = ppo_update(
                ego_train_state, flat_batch, flat_adv, flat_ret, config, update_rng
            )

            # Update population arrays
            new_all_params = jax.tree.map(
                lambda old, new: old.at[ego_idx].set(new),
                all_params,
                ego_train_state.params,
            )
            new_all_opt_states = jax.tree.map(
                lambda old, new: old.at[ego_idx].set(new),
                all_opt_states,
                ego_train_state.opt_state,
            )

            metrics = {
                "mean_reward": jnp.mean(traj_batch.reward),
                "total_loss": loss_info["total_loss"],
            }
            return (new_all_params, new_all_opt_states, env_state, obs, rng), metrics

        runner_state = (all_params, all_opt_states, env_state, obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=config["num_updates"]
        )
        return {"all_params": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    num_envs = args.num_envs
    num_steps = args.num_steps
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // (num_steps * num_envs)
    batch_size = num_steps * num_envs
    minibatch_size = min(args.minibatch_size, batch_size)

    return {
        "layout": {"layout_name": args.layout},
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_updates": num_updates,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef,
        "num_epochs": args.num_epochs,
        "minibatch_size": minibatch_size,
        "hidden_dims": tuple([args.hidden_dim] * args.num_layers),
        "pop_size": args.pop_size,
        "mep_alpha": args.mep_alpha,
    }


def main():
    parser = argparse.ArgumentParser(description="MEP training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.3)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pop-size", type=int, default=4)
    parser.add_argument("--mep-alpha", type=float, default=0.01)
    args = parser.parse_args()

    config = get_config(args)
    print(f"[MEP] seed={args.seed} layout={args.layout} pop_size={args.pop_size} "
          f"alpha={args.mep_alpha} updates={config['num_updates']}")

    rng = jax.random.PRNGKey(args.seed)
    train_fn = make_train(config)

    t0 = time.time()
    result = train_fn(rng)
    jax.block_until_ready(result)
    elapsed = time.time() - t0

    rewards = result["metrics"]["mean_reward"]
    print(f"[MEP] Done in {elapsed:.1f}s. Final mean reward: {rewards[-1]:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "all_params": jax.device_get(result["all_params"]),
        "config": config,
        "metrics": jax.device_get(result["metrics"]),
    }
    ckpt_path = f"checkpoints/mep_seed{args.seed}_{args.layout}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[MEP] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
