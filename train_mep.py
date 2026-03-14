"""Maximum Entropy Population (MEP) training in JAX.

Two phases:
  Phase 1 (Population): Train N agents with entropy bonus for diversity.
    Each round: pick 1 random agent, train for ~epoch_steps, recompute
    entropy bonus across all agents. Bonus = -alpha * log(mean_prob).
  Phase 2 (Ego): Train 1 ego agent against the diverse MEP population
    with uniform random partner selection.
"""

import argparse
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


def make_mep_pop_train(config):
    """Phase 1: Train MEP population with entropy bonus."""
    envs = [Overcooked(layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    action_dim = envs[0].action_space(agents[0]).n
    n_layouts = len(LAYOUT_NAMES)
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * n_layouts
    pop_size = config["pop_size"]

    encoder = ObsEncoder()
    reward_shaper = RewardShaper(mode=config["reward_mode"], reward_shaping_factor=config["reward_shaping_factor"], reward_shaping_horizon=config["reward_shaping_horizon"])
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(config["lr"], eps=1e-5))

    # Epoch steps: how many PPO updates per agent selection round
    epoch_updates = config["epoch_updates"]

    def train(rng):
        # Init population
        init_rngs = jax.random.split(rng, pop_size + 1)
        rng = init_rngs[0]

        def _init_one(r):
            return network.init(r, jnp.zeros((encoder.obs_flat_dim,)))

        all_params = jax.vmap(_init_one)(init_rngs[1 : 1 + pop_size])
        all_opt_states = jax.vmap(tx.init)(all_params)

        # Init envs (shared across all agents, reset each round)
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

        num_rounds = config["num_updates"] // epoch_updates

        def _compute_entropy_bonus(obs_flat, actions, all_params):
            """MEP entropy bonus: -alpha * log(mean_prob across population)."""
            alpha = config["mep_alpha"]

            def _get_probs(params):
                pi, _ = network.apply(params, obs_flat)
                return jax.nn.softmax(pi.logits, axis=-1)

            all_probs = jax.vmap(_get_probs)(all_params)  # (pop, batch, action_dim)
            action_probs = jnp.take_along_axis(
                all_probs,
                jnp.broadcast_to(actions[None, :, None], (pop_size, actions.shape[0], 1)),
                axis=-1,
            ).squeeze(-1)  # (pop, batch)
            mean_prob = action_probs.mean(axis=0)
            return -alpha * jnp.log(mean_prob + 1e-8)

        def _train_round(carry, round_idx):
            all_params, all_opt_states, all_env_states, all_obs, rng = carry

            # Pick random agent
            rng, select_rng = jax.random.split(rng)
            ego_idx = jax.random.randint(select_rng, (), 0, pop_size)
            ego_params = jax.tree.map(lambda x: x[ego_idx], all_params)
            ego_opt_state = jax.tree.map(lambda x: x[ego_idx], all_opt_states)

            ego_train_state = TrainState(
                step=0, apply_fn=network.apply, params=ego_params, tx=tx, opt_state=ego_opt_state,
            )

            # Train ego for epoch_updates PPO updates
            def _update_step(inner_carry, update_offset):
                ego_ts, all_env_states, all_obs, rng = inner_carry
                update_idx = round_idx * epoch_updates + update_offset

                all_traj = []
                all_sparse = []
                new_states = []
                new_obs_list = []

                for i, env in enumerate(envs):

                    def _env_step(carry, inner_step, _env=env):
                        env_state, obs, rng = carry
                        rng, a0r, a1r, sr = jax.random.split(rng, 4)

                        obs_0 = encoder.encode(obs, env_state, 0, n_env_l)
                        obs_1 = encoder.encode(obs, env_state, 1, n_env_l)

                        pi_0, val_0 = network.apply(ego_ts.params, obs_0)
                        pi_1, _ = network.apply(ego_ts.params, obs_1)

                        a0 = pi_0.sample(seed=a0r)
                        a1 = pi_1.sample(seed=a1r)
                        lp0 = pi_0.log_prob(a0)

                        actions = {agents[0]: a0, agents[1]: a1}
                        step_rngs = jax.random.split(sr, n_env_l)
                        next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                            step_rngs, env_state, actions
                        )

                        step_count = (update_idx * config["num_steps"] + inner_step) * n_env_total
                        reward = reward_shaper.compute(rewards, infos, agents[0], step_count)
                        bonus = _compute_entropy_bonus(obs_0, a0, all_params)
                        reward = reward + bonus

                        trans = Transition(obs_0, a0, reward, dones["__all__"], lp0, val_0)
                        sparse_r = rewards[agents[0]]
                        return (next_state, next_obs, rng), (trans, sparse_r)

                    rng, scan_rng = jax.random.split(rng)
                    carry_i, (traj, sparse_r) = jax.lax.scan(
                        _env_step,
                        (all_env_states[i], all_obs[i], scan_rng),
                        jnp.arange(config["num_steps"]),
                    )
                    new_states.append(carry_i[0])
                    new_obs_list.append(carry_i[1])
                    all_traj.append(traj)
                    all_sparse.append(sparse_r)

                all_env_states = tuple(new_states)
                all_obs = tuple(new_obs_list)

                traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_traj)

                all_last_vals = []
                for j in range(len(envs)):
                    o0 = encoder.encode(all_obs[j], all_env_states[j], 0, n_env_l)
                    _, lv = network.apply(ego_ts.params, o0)
                    all_last_vals.append(lv)
                last_val = jnp.concatenate(all_last_vals)

                advantages, returns = compute_gae(
                    traj_batch, last_val, config["gamma"], config["gae_lambda"]
                )
                flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj_batch)
                flat_adv = advantages.reshape(-1)
                flat_ret = returns.reshape(-1)

                rng, update_rng = jax.random.split(rng)
                ego_ts, loss_info = ppo_update(
                    ego_ts, flat_batch, flat_adv, flat_ret, config, update_rng
                )

                return (ego_ts, all_env_states, all_obs, rng), jnp.mean(traj_batch.reward)

            inner_carry = (ego_train_state, all_env_states, all_obs, rng)
            inner_carry, round_rewards = jax.lax.scan(
                _update_step, inner_carry, jnp.arange(epoch_updates)
            )
            ego_train_state, all_env_states, all_obs, rng = inner_carry

            # Update population with trained ego params
            new_all_params = jax.tree.map(
                lambda old, new: old.at[ego_idx].set(new),
                all_params, ego_train_state.params,
            )
            new_all_opt_states = jax.tree.map(
                lambda old, new: old.at[ego_idx].set(new),
                all_opt_states, ego_train_state.opt_state,
            )

            return (new_all_params, new_all_opt_states, all_env_states, all_obs, rng), {
                "mean_reward": round_rewards.mean()
            }

        init_carry = (all_params, all_opt_states, all_env_states, all_obs, rng)
        final_carry, metrics = jax.lax.scan(
            _train_round, init_carry, jnp.arange(num_rounds)
        )

        return {"all_params": final_carry[0], "metrics": metrics}

    return jax.jit(train), network, encoder


def make_mep_ego_train(config, network, encoder, reward_shaper):
    """Phase 2: Train ego agent against MEP population with prioritized sampling.

    Prioritized sampling (matching MEP paper):
      - Track per-partner mean reward
      - Lower reward → higher sampling probability
      - priority = (1/reward_rank) ^ alpha, then normalize
    """
    envs = [Overcooked(layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * len(LAYOUT_NAMES)
    priority_alpha = config.get("priority_alpha", 3.0)

    def train(rng, pop_params):
        """pop_params: tree of (pop_size, ...)"""
        pop_size = jax.tree.leaves(pop_params)[0].shape[0]

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

        # Track per-partner performance for prioritized sampling
        partner_rewards = jnp.zeros(pop_size)  # running mean reward per partner

        def _update_step(runner_state, update_idx):
            train_state, all_env_states, all_obs, rng, partner_rewards = runner_state

            # Prioritized sampling: lower reward → higher probability
            # Use inverse rank with alpha exponent (matching MEP paper)
            rng, partner_rng = jax.random.split(rng)

            # Compute priorities: invert rewards (lower = higher priority)
            epsilon = 1e-6
            inv_rewards = 1.0 / (partner_rewards + epsilon)
            # Rank-based: sort and use ranks
            ranks = jnp.argsort(jnp.argsort(-inv_rewards)) + 1  # rank 1 = highest priority
            priorities = (1.0 / ranks) ** priority_alpha
            probs = priorities / priorities.sum()

            partner_idx = jax.random.choice(partner_rng, pop_size, p=probs)
            partner_params = jax.tree.map(lambda x: x[partner_idx], pop_params)

            all_traj = []
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
                    pi_1, _ = network.apply(partner_params, obs_1)

                    a0 = pi_0.sample(seed=a0r)
                    a1 = pi_1.sample(seed=a1r)
                    lp0 = pi_0.log_prob(a0)

                    actions = {agents[0]: a0, agents[1]: a1}
                    step_rngs = jax.random.split(sr, n_env_l)
                    next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                        step_rngs, env_state, actions
                    )

                    step_count = (update_idx * config["num_steps"] + inner_step) * n_env_total
                    reward = reward_shaper.compute(rewards, infos, agents[0], step_count)

                    trans = Transition(obs_0, a0, reward, dones["__all__"], lp0, val_0)
                    sparse_r = rewards[agents[0]]
                    return (next_state, next_obs, rng), (trans, sparse_r)

                rng, scan_rng = jax.random.split(rng)
                carry, (traj, sparse_r) = jax.lax.scan(
                    _env_step,
                    (all_env_states[i], all_obs[i], scan_rng),
                    jnp.arange(config["num_steps"]),
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                all_traj.append(traj)
                all_sparse.append(sparse_r)

            all_env_states = tuple(new_states)
            all_obs = tuple(new_obs_list)

            traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_traj)

            all_last_vals = []
            for i in range(len(envs)):
                o0 = encoder.encode(all_obs[i], all_env_states[i], 0, n_env_l)
                _, lv = network.apply(train_state.params, o0)
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

            mean_reward = jnp.mean(traj_batch.reward)

            # Update running mean reward for this partner (exponential moving avg)
            ema_alpha = 0.1
            partner_rewards = partner_rewards.at[partner_idx].set(
                (1 - ema_alpha) * partner_rewards[partner_idx] + ema_alpha * mean_reward
            )

            mean_sparse = jnp.mean(jnp.concatenate(all_sparse, axis=1))
            metrics = {"mean_reward": mean_reward, "mean_sparse_reward": mean_sparse, "total_loss": loss_info["total_loss"]}
            return (train_state, all_env_states, all_obs, rng, partner_rewards), metrics

        runner_state = (train_state, all_env_states, all_obs, rng, partner_rewards)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["ego_num_updates"])
        )
        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    n_env_l = args.num_envs_per_layout
    n_layouts = len(LAYOUT_NAMES)
    n_env_total = n_env_l * n_layouts
    num_steps = args.num_steps
    pop_updates = args.pop_timesteps // (num_steps * n_env_total)
    ego_updates = args.ego_timesteps // (num_steps * n_env_total)
    batch_size = num_steps * n_env_total
    minibatch_size = min(args.minibatch_size, batch_size)

    return {
        "num_envs_per_layout": n_env_l,
        "num_steps": num_steps,
        "num_updates": pop_updates,
        "ego_num_updates": ego_updates,
        "epoch_updates": args.epoch_updates,
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
        "mep_alpha": args.mep_alpha,
        "priority_alpha": args.priority_alpha,
    }


def main():
    parser = argparse.ArgumentParser(description="MEP training (2-phase)")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated seeds for population training")
    parser.add_argument("--ego-seeds", type=str, default=None,
                        help="Comma-separated seeds for ego training (default: same as --seeds)")
    parser.add_argument("--pop-timesteps", type=int, default=60_000_000)
    parser.add_argument("--ego-timesteps", type=int, default=60_000_000)
    parser.add_argument("--num-envs-per-layout", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--epoch-updates", type=int, default=20,
                        help="PPO updates per agent selection round (~100K steps with 40 envs × 128 steps)")
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
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--mep-alpha", type=float, default=0.01)
    parser.add_argument("--priority-alpha", type=float, default=3.0,
                        help="Prioritized sampling exponent for ego phase (MEP paper: 3.0)")
    parser.add_argument("--obs-encoding", type=str, default="egocentric")
    parser.add_argument("--reward-mode", type=str, default="shaped")
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--reward-shaping-horizon", type=float, default=float("inf"))
    setup_wandb_args(parser)
    args = parser.parse_args()

    pop_seeds = [int(s) for s in args.seeds.split(",")]
    ego_seeds = [int(s) for s in args.ego_seeds.split(",")] if args.ego_seeds else pop_seeds
    config = get_config(args)

    # Phase 1: MEP population training (vmap over seeds)
    pop_rounds = config["num_updates"] // config["epoch_updates"]
    print(f"[MEP] Phase 1: pop={args.pop_size}, seeds={pop_seeds}, "
          f"{pop_rounds} rounds × {args.epoch_updates} updates/round, "
          f"{args.pop_timesteps/1e6:.0f}M ({len(pop_seeds)} seeds vmapped)")

    train_pop_fn, network, encoder = make_mep_pop_train(config)
    multi_pop_fn = jax.jit(jax.vmap(train_pop_fn))
    pop_rngs = jnp.stack([jax.random.PRNGKey(s) for s in pop_seeds])

    t0 = time.time()
    pop_results = multi_pop_fn(pop_rngs)
    jax.block_until_ready(pop_results["all_params"])
    pop_time = time.time() - t0

    for i, seed in enumerate(pop_seeds):
        r = float(pop_results["metrics"]["mean_reward"][i, -1])
        print(f"  seed {seed}: final pop reward = {r:.4f}")
    print(f"[MEP] Phase 1 done in {pop_time:.1f}s ({pop_time/60:.1f} min)")

    # Save pop params per seed + W&B
    os.makedirs("checkpoints", exist_ok=True)
    tags = args.wandb_tags.split(",") if args.wandb_tags else [f"mep{args.pop_size}"]
    for i, seed in enumerate(pop_seeds):
        pop_ckpt = {
            "all_params": jax.device_get(jax.tree.map(lambda x: x[i], pop_results["all_params"])),
            "config": config,
        }
        with open(f"checkpoints/mep{args.pop_size}_pop_seed{seed}.pkl", "wb") as f:
            pickle.dump(pop_ckpt, f)

        logger = WandbLogger(
            project=args.wandb_project, name=f"mep{args.pop_size}_pop_seed{seed}",
            config={**config, "seed": seed, "method": f"MEP{args.pop_size}_POP"},
            enabled=args.wandb, group=args.wandb_group, tags=tags,
        )
        pop_metrics_i = jax.device_get(jax.tree.map(lambda x: x[i], pop_results["metrics"]))
        logger.log_training_curves(pop_metrics_i, prefix="pop")
        logger.log_summary({"final_pop_reward": float(pop_metrics_i["mean_reward"][-1]),
                            "pop_time_s": pop_time})
        logger.finish()

    # Phase 2: Ego agent training (vmap over ego_seeds)
    if config["ego_num_updates"] > 0 and ego_seeds:
        print(f"[MEP] Phase 2: ego_seeds={ego_seeds}, "
              f"{config['ego_num_updates']} updates, {args.ego_timesteps/1e6:.0f}M "
              f"({len(ego_seeds)} seeds vmapped)")

        reward_shaper = RewardShaper(
            mode=config["reward_mode"],
            reward_shaping_factor=config["reward_shaping_factor"],
            reward_shaping_horizon=config["reward_shaping_horizon"],
        )
        ego_train_fn = make_mep_ego_train(config, network, encoder, reward_shaper)
        multi_ego_fn = jax.jit(jax.vmap(ego_train_fn))

        # Stack per-seed pop params for ego seeds
        ego_pop_params = []
        for seed in ego_seeds:
            si = pop_seeds.index(seed)
            ego_pop_params.append(jax.tree.map(lambda x: x[si], pop_results["all_params"]))
        stacked_pop = jax.tree.map(lambda *xs: jnp.stack(xs), *ego_pop_params)

        ego_rngs = jnp.stack([jax.random.PRNGKey(s + 3000) for s in ego_seeds])

        t0 = time.time()
        ego_results = multi_ego_fn(ego_rngs, stacked_pop)
        jax.block_until_ready(ego_results)
        ego_time = time.time() - t0

        for i, seed in enumerate(ego_seeds):
            r = float(ego_results["metrics"]["mean_reward"][i, -1])
            print(f"  seed {seed}: final ego reward = {r:.4f}")
        print(f"[MEP] Phase 2 done in {ego_time:.1f}s ({ego_time/60:.1f} min)")

        # Save ego per seed + W&B
        for i, seed in enumerate(ego_seeds):
            ego_metrics_i = jax.device_get(jax.tree.map(lambda x: x[i], ego_results["metrics"]))
            ckpt = {
                "ego_params": jax.device_get(jax.tree.map(lambda x: x[i], ego_results["train_state"].params)),
                "pop_params": jax.device_get(jax.tree.map(lambda x: x[pop_seeds.index(seed)], pop_results["all_params"])),
                "config": config,
                "ego_metrics": ego_metrics_i,
            }
            with open(f"checkpoints/mep{args.pop_size}_seed{seed}.pkl", "wb") as f:
                pickle.dump(ckpt, f)
            print(f"  Saved seed {seed} to checkpoints/mep{args.pop_size}_seed{seed}.pkl")

            logger = WandbLogger(
                project=args.wandb_project, name=f"mep{args.pop_size}_ego_seed{seed}",
                config={**config, "seed": seed, "method": f"MEP{args.pop_size}_EGO"},
                enabled=args.wandb, group=args.wandb_group, tags=tags,
            )
            logger.log_training_curves(ego_metrics_i, prefix="ego")
            logger.log_summary({"final_ego_reward": float(ego_metrics_i["mean_reward"][-1]),
                                "ego_time_s": ego_time})
            logger.finish()


if __name__ == "__main__":
    main()
