"""Self-Play (SP) training in JAX with N-agent support.

All N agents share the same policy. Transitions from all N agents
are collected for PPO update. Pure self-play (X=0).
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
from nxplay import shuffle_agent_slots

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]


def make_train(config):
    num_agents = config["num_agents"]
    envs = [Overcooked(layout=overcooked_layouts[name], num_agents=num_agents, random_start_pos=config.get("random_start_pos", False))
            for name in LAYOUT_NAMES]
    agent_names = envs[0].agents
    action_dim = envs[0].action_space(agent_names[0]).n
    n_layouts = len(LAYOUT_NAMES)
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * n_layouts

    encoder = ObsEncoder()
    reward_shaper = RewardShaper(mode=config["reward_mode"],
                                 reward_shaping_factor=config["reward_shaping_factor"],
                                 reward_shaping_horizon=config["reward_shaping_horizon"])
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    def train(rng):
        rng, init_rng = jax.random.split(rng)
        train_state = create_train_state(init_rng, network, (encoder.obs_flat_dim,), config["lr"])

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

            # Shuffle agent slots (multiHRI-style: randomize ego perspective)
            rng, shuffle_rng = jax.random.split(rng)
            all_env_states = tuple(
                jax.vmap(lambda s, k: shuffle_agent_slots(k, s, num_agents))(
                    all_env_states[i], jax.random.split(shuffle_rng, n_env_l)
                ) for i in range(len(envs))
            )

            all_agent_traj = []
            all_sparse = []
            new_states = []
            new_obs_list = []

            for layout_i, env in enumerate(envs):

                def _env_step(carry, inner_step, _env=env):
                    env_state, obs, rng = carry
                    rng, action_rng, step_rng = jax.random.split(rng, 3)
                    action_rngs = jax.random.split(action_rng, num_agents)

                    # All N agents use ego params
                    transitions = []
                    actions_list = []
                    for k in range(num_agents):
                        obs_k = encoder.encode(obs, env_state, k, n_env_l)
                        pi, val = network.apply(train_state.params, obs_k)
                        a = pi.sample(seed=action_rngs[k])
                        lp = pi.log_prob(a)
                        transitions.append((obs_k, a, lp, val))
                        actions_list.append(a)

                    actions_dict = {agent_names[k]: actions_list[k] for k in range(num_agents)}
                    step_rngs = jax.random.split(step_rng, n_env_l)
                    next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                        step_rngs, env_state, actions_dict
                    )

                    step_count = (update_idx * config["num_steps"] + inner_step) * n_env_total
                    done_all = dones["__all__"]

                    # Build Transition for each agent
                    trans_list = []
                    for k in range(num_agents):
                        obs_k, a_k, lp_k, val_k = transitions[k]
                        reward_k = reward_shaper.compute(rewards, infos, agent_names[k], step_count)
                        trans_list.append(Transition(obs_k, a_k, reward_k, done_all, lp_k, val_k))

                    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *trans_list)
                    sparse_r = rewards[agent_names[0]]
                    return (next_state, next_obs, rng), (stacked, sparse_r)

                rng, scan_rng = jax.random.split(rng)
                carry, (traj_stacked, sparse_r) = jax.lax.scan(
                    _env_step,
                    (all_env_states[layout_i], all_obs[layout_i], scan_rng),
                    jnp.arange(config["num_steps"]),
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                # traj_stacked: (T, N, n_env_l, ...) -> (T, N*n_env_l, ...)
                traj_flat = jax.tree.map(
                    lambda x: x.reshape(x.shape[0], -1, *x.shape[3:]), traj_stacked
                )
                all_agent_traj.append(traj_flat)
                all_sparse.append(sparse_r)

            all_env_states = tuple(new_states)
            all_obs = tuple(new_obs_list)

            # Concatenate across layouts
            traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_agent_traj)

            # Bootstrap values for all N agents
            all_last_vals = []
            for layout_i in range(len(envs)):
                for k in range(num_agents):
                    o = encoder.encode(all_obs[layout_i], all_env_states[layout_i], k, n_env_l)
                    _, lv = network.apply(train_state.params, o)
                    all_last_vals.append(lv)
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
    batch_size = num_steps * n_env_total * args.num_agents
    minibatch_size = min(args.minibatch_size, batch_size)

    return {
        "num_agents": args.num_agents,
        "random_start_pos": args.random_start_pos,
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
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--random-start-pos", action="store_true", default=False,
                        help="Randomize agent starting positions (default: fixed layout positions)")
    parser.add_argument("--num-envs-per-layout", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=500)
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
    parser.add_argument("--reward-mode", type=str, default="originaler",
                        choices=["shaped", "sparse", "originaler"])
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--reward-shaping-horizon", type=float, default=float("inf"))
    setup_wandb_args(parser)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    config = get_config(args)

    print(f"[SP] seeds={seeds} N={args.num_agents} updates={config['num_updates']} "
          f"reward={args.reward_mode}")

    train_fn = make_train(config)
    multi_train_fn = jax.jit(jax.vmap(train_fn))
    rngs = jnp.stack([jax.random.PRNGKey(s) for s in seeds])

    t0 = time.time()
    results = multi_train_fn(rngs)
    jax.block_until_ready(results)
    elapsed = time.time() - t0

    os.makedirs("checkpoints", exist_ok=True)
    tags = args.wandb_tags.split(",") if args.wandb_tags else ["sp"]
    for i, seed in enumerate(seeds):
        reward = float(results["metrics"]["mean_reward"][i, -1])
        sparse = float(results["metrics"]["mean_sparse_reward"][i, -1])
        print(f"  seed {seed}: reward={reward:.4f}, sparse={sparse:.4f}")
        metrics_i = jax.device_get(jax.tree.map(lambda x: x[i], results["metrics"]))
        ckpt = {
            "params": jax.device_get(jax.tree.map(lambda x: x[i], results["train_state"].params)),
            "config": config, "metrics": metrics_i,
        }
        with open(f"checkpoints/sp_n{args.num_agents}_seed{seed}.pkl", "wb") as f:
            pickle.dump(ckpt, f)
        logger = WandbLogger(
            project=args.wandb_project, name=f"sp_n{args.num_agents}_seed{seed}",
            config={**config, "seed": seed, "method": "SP"},
            enabled=args.wandb, group=args.wandb_group, tags=tags,
        )
        logger.log_training_curves(metrics_i, prefix="train")
        logger.log_summary({"final_reward": reward, "final_sparse": sparse, "train_time_s": elapsed})
        logger.finish()

    print(f"[SP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
