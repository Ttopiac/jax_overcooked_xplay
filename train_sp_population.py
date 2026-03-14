"""Parallel SP population training in JAX.

Trains N SP agents simultaneously using jax.vmap over the agent dimension.
Each agent runs its own independent self-play training loop with its own
set of parallel environments.

Saves checkpoints at regular intervals for use by BCP and FCP.
"""

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts

from networks import ActorCritic
from ppo import Transition, compute_gae, ppo_update, create_train_state


def flatten_obs(obs, num_envs):
    return obs.reshape(num_envs, -1).astype(jnp.float32) 


def make_parallel_sp_train(config):
    layout = overcooked_layouts[config["layout_name"]]
    env = make("overcooked", layout=layout)
    agents = env.agents

    sample_obs, _ = env.reset(jax.random.PRNGKey(0))
    obs_shape = sample_obs[agents[0]].shape
    obs_flat_dim = int(jnp.prod(jnp.array(obs_shape)))
    action_dim = env.action_space(agents[0]).n

    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    num_updates = config["num_updates"]
    num_ckpts = config["num_checkpoints"]
    ck_interval = max(1, num_updates // num_ckpts)

    def train_single_agent(rng):
        """Train one SP agent. This function is vmapped over the population."""
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        train_state = create_train_state(init_rng, network, (obs_flat_dim,), config["lr"])

        # Init environments
        env_rngs = jax.random.split(env_rng, config["num_envs"])
        obs, env_state = jax.vmap(env.reset)(env_rngs)

        # Pre-allocate checkpoint storage
        ckpt_params = jax.tree.map(
            lambda p: jnp.zeros((num_ckpts, *p.shape)), train_state.params
        )
        ckpt_rewards = jnp.zeros(num_ckpts)

        def _update_step(runner_state, step_idx):
            train_state, env_state, obs, rng, ckpt_params, ckpt_rewards = runner_state

            # --- Collect rollout ---
            def _env_step(carry, unused):
                env_state, obs, rng = carry
                rng, a_rng0, a_rng1, s_rng = jax.random.split(rng, 4)

                obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
                obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])

                pi_0, val_0 = network.apply(train_state.params, obs_0)
                pi_1, val_1 = network.apply(train_state.params, obs_1)

                a0 = pi_0.sample(seed=a_rng0)
                a1 = pi_1.sample(seed=a_rng1)
                lp0 = pi_0.log_prob(a0)
                lp1 = pi_1.log_prob(a1)

                actions = {agents[0]: a0, agents[1]: a1}
                step_rngs = jax.random.split(s_rng, config["num_envs"])
                next_obs, next_state, rewards, dones, infos = jax.vmap(env.step)(
                    step_rngs, env_state, actions
                )

                reward = rewards[agents[0]]
                done_all = dones["__all__"]
                trans_0 = Transition(obs_0, a0, reward, done_all, lp0, val_0)
                trans_1 = Transition(obs_1, a1, reward, done_all, lp1, val_1)
                return (next_state, next_obs, rng), (trans_0, trans_1)

            carry, (traj_0, traj_1) = jax.lax.scan(
                _env_step, (env_state, obs, rng), None, length=config["num_steps"]
            )
            env_state, obs, rng = carry

            traj_batch = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=1), traj_0, traj_1
            )

            # Bootstrap
            obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
            obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])
            _, lv0 = network.apply(train_state.params, obs_0)
            _, lv1 = network.apply(train_state.params, obs_1)
            last_val = jnp.concatenate([lv0, lv1])

            # GAE
            advantages, returns = compute_gae(
                traj_batch, last_val, config["gamma"], config["gae_lambda"]
            )

            # Flatten and PPO update
            flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj_batch)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)

            rng, update_rng = jax.random.split(rng)
            train_state, loss_info = ppo_update(
                train_state, flat_batch, flat_adv, flat_ret, config, update_rng
            )

            # --- Checkpoint logic ---
            mean_reward = jnp.mean(traj_batch.reward)
            ck_idx = jnp.clip(step_idx // ck_interval, 0, num_ckpts - 1)
            is_boundary = (step_idx % ck_interval == 0)

            ckpt_params = jax.tree.map(
                lambda ck, p: ck.at[ck_idx].set(jnp.where(is_boundary, p, ck[ck_idx])),
                ckpt_params,
                train_state.params,
            )
            ckpt_rewards = ckpt_rewards.at[ck_idx].set(
                jnp.where(is_boundary, mean_reward, ckpt_rewards[ck_idx])
            )

            return (
                train_state, env_state, obs, rng, ckpt_params, ckpt_rewards
            ), {"mean_reward": mean_reward}

        init_state = (train_state, env_state, obs, rng, ckpt_params, ckpt_rewards)
        steps = jnp.arange(num_updates)
        final_state, metrics = jax.lax.scan(_update_step, init_state, steps)

        return {
            "final_params": final_state[0].params,
            "ckpt_params": final_state[4],
            "ckpt_rewards": final_state[5],
            "metrics": metrics,
        }

    # vmap over entire population
    return jax.jit(jax.vmap(train_single_agent))


def get_config(args):
    num_envs = args.num_envs
    num_steps = args.num_steps
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // (num_steps * num_envs)
    batch_size = num_steps * num_envs * 2  # both agents
    minibatch_size = min(args.minibatch_size, batch_size)

    return {
        "layout_name": args.layout,
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
        "num_checkpoints": args.num_checkpoints,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel SP population training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--total-timesteps", type=int, default=60_000_000)
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
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--num-checkpoints", type=int, default=15)
    args = parser.parse_args()

    config = get_config(args)
    print(
        f"[SP-POP] seed={args.seed} layout={args.layout} pop_size={args.pop_size} "
        f"updates={config['num_updates']} timesteps={args.total_timesteps/1e6:.0f}M "
        f"({args.pop_size} agents in parallel via vmap)"
    )

    rng = jax.random.PRNGKey(args.seed)
    train_fn = make_parallel_sp_train(config)

    # Split RNGs for each agent in the population
    pop_rngs = jax.random.split(rng, args.pop_size)

    t0 = time.time()
    result = train_fn(pop_rngs)
    jax.block_until_ready(result["final_params"])
    elapsed = time.time() - t0

    # result shapes: final_params tree of (pop_size, ...), ckpt_params tree of (pop_size, num_ckpts, ...)
    mean_rewards = result["metrics"]["mean_reward"]  # (pop_size, num_updates)
    for i in range(args.pop_size):
        final_rew = float(mean_rewards[i, -1])
        print(f"  Agent {i}: final mean reward = {final_rew:.4f}")

    print(f"[SP-POP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "final_params": jax.device_get(result["final_params"]),
        "ckpt_params": jax.device_get(result["ckpt_params"]),
        "ckpt_rewards": jax.device_get(result["ckpt_rewards"]),
        "config": config,
    }
    ckpt_path = f"checkpoints/sp_pop_seed{args.seed}_{args.layout}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[SP-POP] Saved population to {ckpt_path}")


if __name__ == "__main__":
    main()
