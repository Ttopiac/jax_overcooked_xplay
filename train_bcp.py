"""Best Cross-Play (BCP) training in JAX.

Loads a pre-trained SP population (from train_sp_population.py),
selects the best agent, and trains a new best-response agent against it.
"""

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts

from networks import ActorCritic
from ppo import Transition, compute_gae, ppo_update, create_train_state


def flatten_obs(obs, num_envs):
    return obs.reshape(num_envs, -1).astype(jnp.float32) 


def make_bcp_train(config, network, obs_flat_dim, env, agents):
    """Train a best-response agent against a frozen partner."""

    def train(rng, frozen_params):
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        train_state = create_train_state(init_rng, network, (obs_flat_dim,), config["lr"])

        env_rngs = jax.random.split(env_rng, config["num_envs"])
        obs, env_state = jax.vmap(env.reset)(env_rngs)

        def _update_step(runner_state, unused):
            train_state, env_state, obs, rng = runner_state

            def _env_step(carry, unused):
                env_state, obs, rng = carry
                rng, a_rng0, a_rng1, s_rng = jax.random.split(rng, 4)

                obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
                obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])

                # Ego uses trainable params, partner uses frozen params
                pi_0, val_0 = network.apply(train_state.params, obs_0)
                pi_1, _ = network.apply(frozen_params, obs_1)

                a0 = pi_0.sample(seed=a_rng0)
                a1 = pi_1.sample(seed=a_rng1)
                lp0 = pi_0.log_prob(a0)

                actions = {agents[0]: a0, agents[1]: a1}
                step_rngs = jax.random.split(s_rng, config["num_envs"])
                next_obs, next_state, rewards, dones, infos = jax.vmap(env.step)(
                    step_rngs, env_state, actions
                )

                trans = Transition(
                    obs_0, a0, rewards[agents[0]], dones["__all__"], lp0, val_0
                )
                return (next_state, next_obs, rng), trans

            carry, traj_batch = jax.lax.scan(
                _env_step, (env_state, obs, rng), None, length=config["num_steps"]
            )
            env_state, obs, rng = carry

            obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
            _, last_val = network.apply(train_state.params, obs_0)

            advantages, returns = compute_gae(
                traj_batch, last_val, config["gamma"], config["gae_lambda"]
            )

            flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj_batch)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)

            rng, update_rng = jax.random.split(rng)
            train_state, loss_info = ppo_update(
                train_state, flat_batch, flat_adv, flat_ret, config, update_rng
            )

            metrics = {
                "mean_reward": jnp.mean(traj_batch.reward),
                "total_loss": loss_info["total_loss"],
            }
            return (train_state, env_state, obs, rng), metrics

        runner_state = (train_state, env_state, obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=config["num_updates"]
        )
        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    num_envs = args.num_envs
    num_steps = args.num_steps
    total = args.total_timesteps
    num_updates = total // (num_steps * num_envs)
    batch_size = num_steps * num_envs
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
    }


def main():
    parser = argparse.ArgumentParser(description="BCP training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--total-timesteps", type=int, default=60_000_000,
                        help="Timesteps for BCP agent training (SP pop loaded separately)")
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
    args = parser.parse_args()

    config = get_config(args)

    # --- Load shared SP population ---
    pop_path = f"checkpoints/sp_pop_seed{args.seed}_{args.layout}.pkl"
    print(f"[BCP] Loading SP population from {pop_path}")
    with open(pop_path, "rb") as f:
        pop_data = pickle.load(f)

    all_final_params = pop_data["final_params"]  # tree of (pop_size, ...)
    all_ckpt_rewards = pop_data["ckpt_rewards"]  # (pop_size, num_checkpoints)

    # Select best agent: highest mean checkpoint reward
    agent_scores = all_ckpt_rewards.mean(axis=1)  # (pop_size,)
    best_idx = int(jnp.argmax(agent_scores))
    frozen_params = jax.tree.map(lambda x: x[best_idx], all_final_params)
    print(f"[BCP] Best SP agent idx={best_idx} score={float(agent_scores[best_idx]):.4f}")

    # --- Setup env and network ---
    layout = overcooked_layouts[config["layout_name"]]
    env = make("overcooked", layout=layout)
    agent_names = env.agents

    sample_obs, _ = env.reset(jax.random.PRNGKey(0))
    obs_flat_dim = int(jnp.prod(jnp.array(sample_obs[agent_names[0]].shape)))
    action_dim = env.action_space(agent_names[0]).n
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    # --- Train BCP agent ---
    print(f"[BCP] Training best-response agent ({config['num_updates']} updates, "
          f"{args.total_timesteps/1e6:.0f}M timesteps)")
    bcp_train_fn = make_bcp_train(config, network, obs_flat_dim, env, agent_names)

    rng = jax.random.PRNGKey(args.seed + 1000)  # different from population seed
    frozen_params = jax.device_put(frozen_params)

    t0 = time.time()
    result = bcp_train_fn(rng, frozen_params)
    jax.block_until_ready(result)
    elapsed = time.time() - t0

    rewards = result["metrics"]["mean_reward"]
    print(f"[BCP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min). "
          f"Final mean reward: {rewards[-1]:.4f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "params": jax.device_get(result["train_state"].params),
        "frozen_params": jax.device_get(frozen_params),
        "best_sp_idx": best_idx,
        "config": config,
        "metrics": jax.device_get(result["metrics"]),
    }
    ckpt_path = f"checkpoints/bcp_seed{args.seed}_{args.layout}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[BCP] Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
