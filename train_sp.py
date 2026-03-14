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
from overcooked_env import Overcooked, overcooked_layouts

from networks import ActorCritic
from obs_encoding import ObsEncoder
from ppo import Transition, compute_gae, ppo_update, create_train_state
from reward_shaping import RewardShaper
from wandb_utils import WandbLogger, setup_wandb_args

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]


def make_train(config):
    envs = [Overcooked(layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    action_dim = envs[0].action_space(agents[0]).n
    n_layouts = len(LAYOUT_NAMES)
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * n_layouts

    encoder = ObsEncoder()
    reward_shaper = RewardShaper(mode=config["reward_mode"], reward_shaping_factor=config["reward_shaping_factor"], reward_shaping_horizon=config["reward_shaping_horizon"])
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
            all_sparse = []
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
                    sparse_r = rewards[agents[0]]
                    return (next_state, next_obs, rng), (t0, t1, sparse_r)

                rng, scan_rng = jax.random.split(rng)
                carry, (traj_0, traj_1, sparse_r) = jax.lax.scan(
                    _env_step,
                    (all_env_states[i], all_obs[i], scan_rng),
                    jnp.arange(config["num_steps"]),
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                all_traj_0.append(traj_0)
                all_traj_1.append(traj_1)
                all_sparse.append(sparse_r)  # (num_steps, n_env_l)

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

            # Mean sparse reward across all layouts and envs
            mean_sparse = jnp.mean(jnp.concatenate(all_sparse, axis=1))

            metrics = {
                "mean_reward": jnp.mean(traj_batch.reward),
                "mean_sparse_reward": mean_sparse,
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
        "reward_mode": args.reward_mode,
        "reward_shaping_factor": args.reward_shaping_factor,
        "reward_shaping_horizon": args.reward_shaping_horizon,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Play training")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated seeds, e.g. '0,1,2'")
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--obs-encoding", type=str, default="egocentric",
                        choices=["egocentric", "full"])
    parser.add_argument("--reward-mode", type=str, default="shaped",
                        choices=["shaped", "sparse"])
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--reward-shaping-horizon", type=float, default=float("inf"))
    setup_wandb_args(parser)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    config = get_config(args)

    print(f"[SP] seeds={seeds} updates={config['num_updates']} "
          f"obs={args.obs_encoding} reward={args.reward_mode} "
          f"layouts={len(LAYOUT_NAMES)} envs/layout={args.num_envs_per_layout}")

    # Build train function and vmap over seeds
    train_fn = make_train(config)
    multi_train_fn = jax.jit(jax.vmap(train_fn))

    rngs = jnp.stack([jax.random.PRNGKey(s) for s in seeds])

    t0 = time.time()
    results = multi_train_fn(rngs)
    jax.block_until_ready(results)
    elapsed = time.time() - t0

    # Save per-seed and log to W&B
    os.makedirs("checkpoints", exist_ok=True)
    tags = args.wandb_tags.split(",") if args.wandb_tags else ["sp"]
    for i, seed in enumerate(seeds):
        reward = float(results["metrics"]["mean_reward"][i, -1])
        print(f"  seed {seed}: final mean reward = {reward:.4f}")

        metrics_i = jax.device_get(jax.tree.map(lambda x: x[i], results["metrics"]))
        ckpt = {
            "params": jax.device_get(jax.tree.map(lambda x: x[i], results["train_state"].params)),
            "config": config,
            "metrics": metrics_i,
        }
        with open(f"checkpoints/sp_seed{seed}.pkl", "wb") as f:
            pickle.dump(ckpt, f)

        # W&B logging
        logger = WandbLogger(
            project=args.wandb_project, name=f"sp_seed{seed}",
            config={**config, "seed": seed, "method": "SP"},
            enabled=args.wandb, group=args.wandb_group, tags=tags,
        )
        logger.log_training_curves(metrics_i, prefix="train")
        logger.log_summary({"final_reward": reward, "train_time_s": elapsed})
        logger.finish()

    print(f"[SP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
