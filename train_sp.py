"""Self-Play (SP) training in JAX.

Both agents share the same network parameters. Transitions from both agents
across all 5 layouts are used to update the shared policy via PPO.
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
from obs_encoding import ObsEncoder
from ppo import Transition, compute_gae, ppo_update, create_train_state
from reward_shaping import RewardShaper

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]


def make_train(config):
    envs = [make("overcooked", layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    action_dim = envs[0].action_space(agents[0]).n
    n_layouts = len(LAYOUT_NAMES)
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * n_layouts

    encoder = ObsEncoder(mode=config["obs_encoding"])
    reward_shaper = RewardShaper(mode=config["reward_mode"])
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    def train(rng):
        rng, init_rng = jax.random.split(rng)
        train_state = create_train_state(init_rng, network, (encoder.obs_flat_dim,), config["lr"])

        # Init envs for all 5 layouts
        all_env_states = []
        all_obs = []
        for env in envs:
            rng, env_rng = jax.random.split(rng)
            env_rngs = jax.random.split(env_rng, n_env_l)
            obs, state = jax.vmap(env.reset)(env_rngs)
            all_obs.append(obs)
            all_env_states.append(state)
        all_env_states = tuple(all_env_states)
        all_obs = tuple(all_obs)

        def _update_step(runner_state, update_idx):
            train_state, all_env_states, all_obs, rng = runner_state

            all_traj_0 = []
            all_traj_1 = []
            new_states = []
            new_obs_list = []

            for i, env in enumerate(envs):

                def _env_step(carry, inner_step, _env=env):
                    env_state, obs, rng = carry
                    rng, a0r, a1r, sr = jax.random.split(rng, 4)

                    obs_0 = encoder.encode(obs, env_state, 0, n_env_l)
                    obs_1 = encoder.encode(obs, env_state, 1, n_env_l)

                    pi_0, val_0 = network.apply(train_state.params, obs_0)
                    pi_1, val_1 = network.apply(train_state.params, obs_1)

                    a0 = pi_0.sample(seed=a0r)
                    a1 = pi_1.sample(seed=a1r)
                    lp0 = pi_0.log_prob(a0)
                    lp1 = pi_1.log_prob(a1)

                    actions = {agents[0]: a0, agents[1]: a1}
                    step_rngs = jax.random.split(sr, n_env_l)
                    next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                        step_rngs, env_state, actions
                    )

                    step_count = (update_idx * config["num_steps"] + inner_step) * n_env_total
                    reward_0 = reward_shaper.compute(rewards, infos, agents[0], step_count)
                    reward_1 = reward_shaper.compute(rewards, infos, agents[1], step_count)
                    done_all = dones["__all__"]

                    t0 = Transition(obs_0, a0, reward_0, done_all, lp0, val_0)
                    t1 = Transition(obs_1, a1, reward_1, done_all, lp1, val_1)
                    return (next_state, next_obs, rng), (t0, t1)

                rng, scan_rng = jax.random.split(rng)
                carry, (traj_0, traj_1) = jax.lax.scan(
                    _env_step,
                    (all_env_states[i], all_obs[i], scan_rng),
                    jnp.arange(config["num_steps"]),
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                all_traj_0.append(traj_0)
                all_traj_1.append(traj_1)

            all_env_states = tuple(new_states)
            all_obs = tuple(new_obs_list)

            # Concatenate across layouts: (T, n_env_l, ...) -> (T, n_env_total, ...)
            combined_0 = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_traj_0)
            combined_1 = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_traj_1)
            # Concatenate both agents: (T, n_env_total*2, ...)
            traj_batch = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=1), combined_0, combined_1
            )

            # Bootstrap values
            all_last_vals = []
            for i, env in enumerate(envs):
                o0 = encoder.encode(all_obs[i], all_env_states[i], 0, n_env_l)
                o1 = encoder.encode(all_obs[i], all_env_states[i], 1, n_env_l)
                _, lv0 = network.apply(train_state.params, o0)
                _, lv1 = network.apply(train_state.params, o1)
                all_last_vals.extend([lv0, lv1])
            last_val = jnp.concatenate(all_last_vals)

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
            return (train_state, all_env_states, all_obs, rng), metrics

        runner_state = (train_state, all_env_states, all_obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["num_updates"])
        )
        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    n_env_l = args.num_envs_per_layout
    n_layouts = len(LAYOUT_NAMES)
    n_env_total = n_env_l * n_layouts
    num_steps = args.num_steps
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // (num_steps * n_env_total)
    batch_size = num_steps * n_env_total * 2  # both agents
    minibatch_size = min(args.minibatch_size, batch_size)

    return {
        "num_envs_per_layout": n_env_l,
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
        "obs_encoding": args.obs_encoding,
        "reward_mode": args.reward_mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Play training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=120_000_000)
    parser.add_argument("--num-envs-per-layout", type=int, default=8)
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
    parser.add_argument("--obs-encoding", type=str, default="egocentric",
                        choices=["egocentric", "full"])
    parser.add_argument("--reward-mode", type=str, default="shaped",
                        choices=["shaped", "sparse"])
    args = parser.parse_args()

    config = get_config(args)
    print(f"[SP] seed={args.seed} updates={config['num_updates']} "
          f"obs={args.obs_encoding} reward={args.reward_mode} "
          f"layouts={len(LAYOUT_NAMES)} envs/layout={args.num_envs_per_layout}")

    rng = jax.random.PRNGKey(args.seed)
    train_fn = make_train(config)

    t0 = time.time()
    result = train_fn(rng)
    jax.block_until_ready(result)
    elapsed = time.time() - t0

    rewards = result["metrics"]["mean_reward"]
    print(f"[SP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min). "
          f"Final mean reward: {rewards[-1]:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "params": jax.device_get(result["train_state"].params),
        "config": config,
        "metrics": jax.device_get(result["metrics"]),
    }
    ckpt_path = f"checkpoints/sp_seed{args.seed}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[SP] Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
