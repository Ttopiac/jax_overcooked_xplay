"""Self-Play (SP) training in JAX.

Both agents share the same network parameters. Transitions from both agents
are used to update the shared policy via PPO.
"""

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts

from networks import ActorCritic
from ppo import Transition, compute_gae, ppo_update, create_train_state


def flatten_obs(obs, num_envs):
    """Flatten (num_envs, H, W, C) uint8 obs to (num_envs, H*W*C) float32."""
    return obs.reshape(num_envs, -1).astype(jnp.float32) / 255.0


def make_train(config):
    layout = overcooked_layouts[config["layout_name"]]
    env = make("overcooked", layout=layout)
    agents = env.agents  # ["agent_0", "agent_1"]

    # Determine observation and action shapes
    sample_key = jax.random.PRNGKey(0)
    sample_obs, _ = env.reset(sample_key)
    obs_shape = sample_obs[agents[0]].shape
    obs_flat_dim = int(jnp.prod(jnp.array(obs_shape)))
    action_dim = env.action_space(agents[0]).n

    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    def train(rng):
        # Initialize
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        train_state = create_train_state(init_rng, network, (obs_flat_dim,), config["lr"])

        # Reset environments
        env_rngs = jax.random.split(env_rng, config["num_envs"])
        obs, env_state = jax.vmap(env.reset)(env_rngs)

        def _update_step(runner_state, unused):
            train_state, env_state, obs, rng = runner_state

            # --- Collect rollout ---
            def _env_step(carry, unused):
                env_state, obs, rng = carry
                rng, act_rng0, act_rng1, step_rng = jax.random.split(rng, 4)

                # Both agents share same params
                obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
                obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])

                pi_0, val_0 = network.apply(train_state.params, obs_0)
                pi_1, val_1 = network.apply(train_state.params, obs_1)

                a0 = pi_0.sample(seed=act_rng0)
                a1 = pi_1.sample(seed=act_rng1)
                lp0 = pi_0.log_prob(a0)
                lp1 = pi_1.log_prob(a1)

                actions = {agents[0]: a0, agents[1]: a1}
                step_rngs = jax.random.split(step_rng, config["num_envs"])
                next_obs, next_state, rewards, dones, infos = jax.vmap(env.step)(
                    step_rngs, env_state, actions
                )

                done_all = dones["__all__"]
                # Use agent_0 reward (cooperative game, both agents get same reward)
                reward = rewards[agents[0]]

                trans_0 = Transition(obs_0, a0, reward, done_all, lp0, val_0)
                trans_1 = Transition(obs_1, a1, reward, done_all, lp1, val_1)
                return (next_state, next_obs, rng), (trans_0, trans_1)

            carry, (traj_0, traj_1) = jax.lax.scan(
                _env_step, (env_state, obs, rng), None, length=config["num_steps"]
            )
            env_state, obs, rng = carry

            # Combine both agents' transitions: (T, N, ...) -> (T, 2N, ...)
            traj_batch = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=1), traj_0, traj_1
            )

            # Bootstrap value
            obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
            obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])
            _, last_val_0 = network.apply(train_state.params, obs_0)
            _, last_val_1 = network.apply(train_state.params, obs_1)
            last_val = jnp.concatenate([last_val_0, last_val_1])

            # GAE
            advantages, returns = compute_gae(
                traj_batch, last_val, config["gamma"], config["gae_lambda"]
            )

            # Flatten (T, 2N, ...) -> (T*2N, ...)
            flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj_batch)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)

            # PPO update
            rng, update_rng = jax.random.split(rng)
            train_state_new, loss_info = ppo_update(
                train_state, flat_batch, flat_adv, flat_ret, config, update_rng
            )

            metrics = {
                "mean_reward": jnp.mean(traj_batch.reward),
                "total_loss": loss_info["total_loss"],
            }
            return (train_state_new, env_state, obs, rng), metrics

        # Run training loop
        runner_state = (train_state, env_state, obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=config["num_updates"]
        )
        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    num_envs = args.num_envs
    num_steps = args.num_steps
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // (num_steps * num_envs)
    # For SP, batch includes both agents
    batch_size = num_steps * num_envs * 2
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
    parser = argparse.ArgumentParser(description="Self-Play training")
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
    args = parser.parse_args()

    config = get_config(args)
    print(f"[SP] seed={args.seed} layout={args.layout} updates={config['num_updates']}")

    rng = jax.random.PRNGKey(args.seed)
    train_fn = make_train(config)

    t0 = time.time()
    result = train_fn(rng)
    jax.block_until_ready(result)
    elapsed = time.time() - t0

    rewards = result["metrics"]["mean_reward"]
    print(f"[SP] Done in {elapsed:.1f}s. Final mean reward: {rewards[-1]:.3f}")

    # Save checkpoint
    import pickle
    ckpt = {
        "params": jax.device_get(result["train_state"].params),
        "config": config,
        "metrics": jax.device_get(result["metrics"]),
    }
    ckpt_path = f"checkpoints/sp_seed{args.seed}_{args.layout}.pkl"
    import os; os.makedirs("checkpoints", exist_ok=True)
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[SP] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
