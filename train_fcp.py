"""Fictitious Co-Play (FCP) training in JAX with N-XPlay support.

N-XPlay framework:
  - N total agents in the game
  - N-X use ego's policy (ego + clones, transitions collected for PPO)
  - X use frozen partner policies from SP population H/M/L tiers
  - Hetero: X partners independently sampled
  - Homo: X partners all use same policy

Agent positions are shuffled each update for position diversity.
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
from nxplay import select_partners, shuffle_agent_slots

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]


def categorize_checkpoints(ckpt_params, ckpt_rewards):
    """Categorize checkpoint params into High/Medium/Low tiers."""
    flat_rewards = ckpt_rewards.reshape(-1)
    n_total = flat_rewards.shape[0]
    third = max(1, n_total // 3)
    sorted_idx = jnp.argsort(-flat_rewards)

    flat_params = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), ckpt_params)
    high = jax.tree.map(lambda x: x[sorted_idx[:third]], flat_params)
    mid = jax.tree.map(lambda x: x[sorted_idx[third:2*third]], flat_params)
    low = jax.tree.map(lambda x: x[sorted_idx[2*third:]], flat_params)
    return high, mid, low


def make_fcp_train(config, network, encoder, reward_shaper):
    num_agents = config["num_agents"]
    num_unseen = config["num_unseen"]
    num_ego = num_agents - num_unseen
    homo = config["xplay_mode"] == "homo"

    envs = [Overcooked(layout=overcooked_layouts[name], num_agents=num_agents, random_start_pos=config.get("random_start_pos", False))
            for name in LAYOUT_NAMES]
    agent_names = envs[0].agents
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * len(LAYOUT_NAMES)

    def train(rng, pool_params):
        """
        pool_params: stacked partner pool, tree of (pool_size, ...)
        """
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

            # Select X partner params from pool
            rng, partner_rng, shuffle_rng = jax.random.split(rng, 3)

            if num_unseen > 0:
                # For FCP: pool is the combined H/M/L checkpoints
                partner_params = select_partners(
                    partner_rng, pool_params, num_unseen, homo=homo
                )  # tree of (num_unseen, ...)
            else:
                partner_params = None

            # Shuffle agent positions for diversity
            all_env_states_shuffled = tuple(
                jax.vmap(lambda s, k: shuffle_agent_slots(k, s, num_agents))(
                    all_env_states[i],
                    jax.random.split(shuffle_rng, n_env_l)
                ) if num_agents > 1 else all_env_states[i]
                for i in range(len(envs))
            )

            all_ego_traj = []  # transitions from ego/clone agents
            all_sparse = []
            new_states = []
            new_obs_list = []

            for layout_i, env in enumerate(envs):

                def _env_step(carry, inner_step, _env=env):
                    env_state, obs, rng = carry
                    rng, action_rng, step_rng = jax.random.split(rng, 3)
                    action_rngs = jax.random.split(action_rng, num_agents)

                    # Encode obs for all agents
                    all_obs_encoded = [
                        encoder.encode(obs, env_state, k, n_env_l)
                        for k in range(num_agents)
                    ]

                    # Compute ego actions for ALL agents (need for ego/clones)
                    # We compute ego forward pass for ego/clone slots (0..num_ego-1)
                    ego_transitions = []
                    actions_list = []

                    for k in range(num_ego):
                        pi, val = network.apply(train_state.params, all_obs_encoded[k])
                        a = pi.sample(seed=action_rngs[k])
                        lp = pi.log_prob(a)

                        step_count = (update_idx * config["num_steps"] + inner_step) * n_env_total
                        # We'll compute reward after stepping, so store obs/action/lp/val
                        ego_transitions.append((all_obs_encoded[k], a, lp, val))
                        actions_list.append(a)

                    # Compute partner actions
                    for j in range(num_unseen):
                        p_params = jax.tree.map(lambda x: x[j], partner_params)
                        pi, _ = network.apply(p_params, all_obs_encoded[num_ego + j])
                        a = pi.sample(seed=action_rngs[num_ego + j])
                        actions_list.append(a)

                    # Step environment
                    actions_dict = {agent_names[k]: actions_list[k] for k in range(num_agents)}
                    step_rngs = jax.random.split(step_rng, n_env_l)
                    next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                        step_rngs, env_state, actions_dict
                    )

                    step_count = (update_idx * config["num_steps"] + inner_step) * n_env_total
                    done_all = dones["__all__"]

                    # Build transitions for each ego/clone agent
                    transitions = []
                    for k in range(num_ego):
                        obs_k, a_k, lp_k, val_k = ego_transitions[k]
                        reward_k = reward_shaper.compute(rewards, infos, agent_names[k], step_count)
                        t = Transition(obs_k, a_k, reward_k, done_all, lp_k, val_k)
                        transitions.append(t)

                    # Stack ego/clone transitions: each field becomes (num_ego, num_envs, ...)
                    # Then reshape to (num_ego * num_envs, ...) later
                    stacked_trans = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)

                    sparse_r = rewards[agent_names[0]]  # team reward
                    return (next_state, next_obs, rng), (stacked_trans, sparse_r)

                rng, scan_rng = jax.random.split(rng)
                carry, (traj_stacked, sparse_r) = jax.lax.scan(
                    _env_step,
                    (all_env_states_shuffled[layout_i], all_obs[layout_i], scan_rng),
                    jnp.arange(config["num_steps"]),
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                # traj_stacked: each field shape (num_steps, num_ego, n_env_l, ...)
                # Reshape to (num_steps, num_ego * n_env_l, ...)
                traj_flat = jax.tree.map(
                    lambda x: x.reshape(x.shape[0], -1, *x.shape[3:]),
                    traj_stacked
                )
                all_ego_traj.append(traj_flat)
                all_sparse.append(sparse_r)

            all_env_states = tuple(new_states)
            all_obs = tuple(new_obs_list)

            # Concatenate across layouts: (T, num_ego*n_env_l*n_layouts, ...)
            traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_ego_traj)

            # Bootstrap values for ego/clone agents
            all_last_vals = []
            for layout_i in range(len(envs)):
                for k in range(num_ego):
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
    n_env_total = n_env_l * len(LAYOUT_NAMES)
    num_steps = args.num_steps
    num_updates = args.total_timesteps // (num_steps * n_env_total)
    num_ego = args.num_agents - args.num_unseen
    batch_size = num_steps * n_env_total * num_ego
    minibatch_size = min(args.minibatch_size, batch_size)

    return {
        "num_agents": args.num_agents,
        "random_start_pos": getattr(args, "random_start_pos", False),
        "num_unseen": args.num_unseen,
        "xplay_mode": args.xplay_mode,
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
    parser = argparse.ArgumentParser(description="FCP training with N-XPlay")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--num-agents", type=int, default=2,
    parser.add_argument("--random-start-pos", action="store_true", default=False,
                        help="Randomize agent starting positions (default: fixed layout positions)")
                        help="N: total agents in the game")
    parser.add_argument("--num-unseen", type=int, default=1,
                        help="X: number of unseen teammate slots")
    parser.add_argument("--xplay-mode", type=str, default="hetero",
                        choices=["hetero", "homo"])
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
    parser.add_argument("--reward-mode", type=str, default="originaler")
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--reward-shaping-horizon", type=float, default=float("inf"))
    setup_wandb_args(parser)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    config = get_config(args)

    print(f"[FCP] N={args.num_agents}, X={args.num_unseen}, mode={args.xplay_mode}")

    # Load per-seed SP population
    all_pool_params = []
    for seed in seeds:
        pop_path = f"checkpoints/sp_pop_seed{seed}.pkl"
        print(f"  Loading {pop_path}")
        with open(pop_path, "rb") as f:
            pop_data = pickle.load(f)
        ckpt_params = jax.device_put(pop_data["ckpt_params"])
        ckpt_rewards = jax.device_put(pop_data["ckpt_rewards"])
        h, m, l = categorize_checkpoints(ckpt_params, ckpt_rewards)
        # Combine all tiers into one pool for uniform sampling
        pool = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), h, m, l)
        all_pool_params.append(pool)

    pool_size = jax.tree.leaves(all_pool_params[0])[0].shape[0]
    print(f"  Pool size per seed: {pool_size}")

    # Stack: (n_seeds, pool_size, ...)
    stacked_pool = jax.tree.map(lambda *xs: jnp.stack(xs), *all_pool_params)

    encoder = ObsEncoder()
    reward_shaper = RewardShaper(
        mode=config["reward_mode"],
        reward_shaping_factor=config["reward_shaping_factor"],
        reward_shaping_horizon=config["reward_shaping_horizon"],
    )
    action_dim = 6
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    fcp_train_fn = make_fcp_train(config, network, encoder, reward_shaper)
    multi_fcp_fn = jax.jit(jax.vmap(fcp_train_fn))

    rngs = jnp.stack([jax.random.PRNGKey(s + 2000) for s in seeds])

    print(f"[FCP] Training {len(seeds)} seeds ({config['num_updates']} updates, "
          f"{args.total_timesteps/1e6:.0f}M)")
    t0 = time.time()
    results = multi_fcp_fn(rngs, stacked_pool)
    jax.block_until_ready(results)
    elapsed = time.time() - t0

    os.makedirs("checkpoints", exist_ok=True)
    tags = args.wandb_tags.split(",") if args.wandb_tags else ["fcp"]
    for i, seed in enumerate(seeds):
        r = float(results["metrics"]["mean_reward"][i, -1])
        sr = float(results["metrics"]["mean_sparse_reward"][i, -1])
        print(f"  seed {seed}: reward={r:.4f}, sparse={sr:.4f}")
        metrics_i = jax.device_get(jax.tree.map(lambda x: x[i], results["metrics"]))
        ckpt = {
            "params": jax.device_get(jax.tree.map(lambda x: x[i], results["train_state"].params)),
            "config": config,
            "metrics": metrics_i,
        }
        with open(f"checkpoints/fcp_n{args.num_agents}x{args.num_unseen}_{args.xplay_mode}_seed{seed}.pkl", "wb") as f:
            pickle.dump(ckpt, f)

        logger = WandbLogger(
            project=args.wandb_project,
            name=f"fcp_n{args.num_agents}x{args.num_unseen}_{args.xplay_mode}_seed{seed}",
            config={**config, "seed": seed, "method": "FCP"},
            enabled=args.wandb, group=args.wandb_group, tags=tags,
        )
        logger.log_training_curves(metrics_i, prefix="train")
        logger.log_summary({"final_reward": r, "final_sparse": sr, "train_time_s": elapsed})
        logger.finish()

    print(f"[FCP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
