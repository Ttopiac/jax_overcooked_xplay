"""Parallel SP population training in JAX.

Trains N SP agents simultaneously using jax.vmap over the agent dimension.
Each agent trains across all 5 Overcooked layouts with egocentric encoding.
Saves checkpoints at regular intervals for use by BCP and FCP.
"""

import argparse
import gc
import os
import pickle
import time

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from overcooked_env import Overcooked, overcooked_layouts

from networks import ActorCritic
from obs_encoding import ObsEncoder
from ppo import Transition, compute_gae, ppo_update, create_train_state
from reward_shaping import RewardShaper
from wandb_utils import WandbLogger, setup_wandb_args

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]


def make_parallel_sp_train(config):
    num_agents = config["num_agents"]
    envs = [Overcooked(layout=overcooked_layouts[name], num_agents=num_agents, random_start_pos=config.get("random_start_pos", False))
            for name in LAYOUT_NAMES]
    agent_names = envs[0].agents
    action_dim = envs[0].action_space(agent_names[0]).n
    n_layouts = len(LAYOUT_NAMES)
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * n_layouts

    encoder = ObsEncoder()
    reward_shaper = RewardShaper(mode=config["reward_mode"], reward_shaping_factor=config["reward_shaping_factor"], reward_shaping_horizon=config["reward_shaping_horizon"])
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    num_updates = config["num_updates"]
    num_ckpts = config["num_checkpoints"]
    ck_interval = max(1, num_updates // num_ckpts)

    def train_single_agent(rng):
        rng, init_rng = jax.random.split(rng)
        train_state = create_train_state(init_rng, network, (encoder.obs_flat_dim,), config["lr"])

        # Init envs for all layouts
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

        # Checkpoint storage
        ckpt_params = jax.tree.map(
            lambda p: jnp.zeros((num_ckpts, *p.shape)), train_state.params
        )
        ckpt_rewards = jnp.zeros(num_ckpts)

        def _update_step(runner_state, update_idx):
            train_state, all_env_states, all_obs, rng, ckpt_params, ckpt_rewards = runner_state

            all_agent_traj = []
            all_sparse = []
            new_states = []
            new_obs_list = []

            for layout_i, env in enumerate(envs):

                def _env_step(carry, inner_step, _env=env):
                    env_state, obs, rng = carry
                    rng, action_rng, step_rng = jax.random.split(rng, 3)
                    action_rngs = jax.random.split(action_rng, num_agents)

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
                traj_flat = jax.tree.map(
                    lambda x: x.reshape(x.shape[0], -1, *x.shape[3:]), traj_stacked
                )
                all_agent_traj.append(traj_flat)
                all_sparse.append(sparse_r)

            all_env_states = tuple(new_states)
            all_obs = tuple(new_obs_list)

            traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_agent_traj)

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

            # Checkpoint logic
            mean_reward = jnp.mean(traj_batch.reward)
            mean_sparse = jnp.mean(jnp.concatenate(all_sparse, axis=1))
            ck_idx = jnp.clip(update_idx // ck_interval, 0, num_ckpts - 1)
            is_boundary = (update_idx % ck_interval == 0)

            ckpt_params = jax.tree.map(
                lambda ck, p: ck.at[ck_idx].set(jnp.where(is_boundary, p, ck[ck_idx])),
                ckpt_params, train_state.params,
            )
            ckpt_rewards = ckpt_rewards.at[ck_idx].set(
                jnp.where(is_boundary, mean_reward, ckpt_rewards[ck_idx])
            )

            return (
                train_state, all_env_states, all_obs, rng, ckpt_params, ckpt_rewards
            ), {"mean_reward": mean_reward, "mean_sparse_reward": mean_sparse}

        init_state = (train_state, all_env_states, all_obs, rng, ckpt_params, ckpt_rewards)
        steps = jnp.arange(num_updates)
        final_state, metrics = jax.lax.scan(_update_step, init_state, steps)

        return {
            "final_params": final_state[0].params,
            "ckpt_params": final_state[4],
            "ckpt_rewards": final_state[5],
            "metrics": metrics,
        }

    return jax.jit(jax.vmap(train_single_agent))


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
        "random_start_pos": getattr(args, "random_start_pos", False),
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
        "pop_size": args.pop_size,
        "num_checkpoints": args.num_checkpoints,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel SP population training")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated seeds, e.g. '0,1,2,3,4'")
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
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--num-checkpoints", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of agents to train in parallel per GPU batch")
    parser.add_argument("--obs-encoding", type=str, default="egocentric",
                        choices=["egocentric", "full"])
    parser.add_argument("--reward-mode", type=str, default="originaler",
                        choices=["shaped", "sparse", "originaler"])
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--reward-shaping-horizon", type=float, default=float("inf"))
    setup_wandb_args(parser)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    config = get_config(args)
    pop_size = args.pop_size
    total_agents = len(seeds) * pop_size
    batch_size = min(args.batch_size, total_agents)
    num_batches = (total_agents + batch_size - 1) // batch_size

    print(
        f"[SP-POP] seeds={seeds} pop_size={pop_size} total_agents={total_agents} "
        f"updates={config['num_updates']} timesteps={args.total_timesteps/1e6:.0f}M "
        f"(batch_size={batch_size}, {num_batches} batches)"
    )

    # Flatten all seeds × pop_size into one array of RNGs
    train_fn = make_parallel_sp_train(config)
    all_rngs = []
    for seed in seeds:
        base_rng = jax.random.PRNGKey(seed)
        all_rngs.append(jax.random.split(base_rng, pop_size))
    all_rngs = jnp.concatenate(all_rngs)  # (total_agents,)

    # Train in batches to fit in VRAM
    all_results = []
    t0 = time.time()
    for batch_idx in range(num_batches):
        b_start = batch_idx * batch_size
        b_end = min(b_start + batch_size, total_agents)
        actual_bs = b_end - b_start
        batch_rngs = all_rngs[b_start:b_end]

        print(f"  Batch {batch_idx+1}/{num_batches}: agents {b_start}-{b_end-1} "
              f"({actual_bs} agents)")

        batch_result = train_fn(batch_rngs)
        # Move results to CPU and explicitly free GPU memory
        batch_result_cpu = jax.device_get(batch_result)
        del batch_result
        gc.collect()
        all_results.append(batch_result_cpu)

        batch_rewards = batch_result_cpu["metrics"]["mean_reward"]  # (bs, num_updates)
        for i in range(actual_bs):
            global_i = b_start + i
            seed_idx = global_i // pop_size
            agent_idx = global_i % pop_size
            print(f"    seed {seeds[seed_idx]} agent {agent_idx}: "
                  f"final reward = {float(batch_rewards[i, -1]):.4f}")

    elapsed = time.time() - t0
    print(f"[SP-POP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Merge batch results and save per-seed
    import numpy as np
    merged_final_params = jax.tree.map(
        lambda *xs: np.concatenate(xs, axis=0),
        *[r["final_params"] for r in all_results]
    )
    merged_ckpt_params = jax.tree.map(
        lambda *xs: np.concatenate(xs, axis=0),
        *[r["ckpt_params"] for r in all_results]
    )
    merged_ckpt_rewards = np.concatenate(
        [r["ckpt_rewards"] for r in all_results], axis=0
    )
    merged_mean_rewards = np.concatenate(
        [r["metrics"]["mean_reward"] for r in all_results], axis=0
    )

    os.makedirs("checkpoints", exist_ok=True)
    for si, seed in enumerate(seeds):
        start, end = si * pop_size, (si + 1) * pop_size
        ckpt = {
            "final_params": jax.tree.map(lambda x: x[start:end], merged_final_params),
            "ckpt_params": jax.tree.map(lambda x: x[start:end], merged_ckpt_params),
            "ckpt_rewards": merged_ckpt_rewards[start:end],
            "config": config,
        }
        ckpt_path = f"checkpoints/sp_pop_seed{seed}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(ckpt, f)
        print(f"  Saved seed {seed} to {ckpt_path}")

        # W&B: log mean reward across agents for this seed
        tags = args.wandb_tags.split(",") if args.wandb_tags else ["sp_pop"]
        logger = WandbLogger(
            project=args.wandb_project, name=f"sp_pop_seed{seed}",
            config={**config, "seed": seed, "method": "SP_POP"},
            enabled=args.wandb, group=args.wandb_group, tags=tags,
        )
        seed_rewards = merged_mean_rewards[start:end]  # (pop_size, num_updates)
        logger.log_training_curves(
            {"mean_reward": seed_rewards.mean(axis=0)}, prefix="train"
        )
        logger.log_summary({"final_mean_reward": float(seed_rewards[:, -1].mean()),
                            "train_time_s": elapsed})
        logger.finish()


if __name__ == "__main__":
    main()
