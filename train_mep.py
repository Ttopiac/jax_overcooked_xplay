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
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts

from networks import ActorCritic
from obs_encoding import ObsEncoder
from ppo import Transition, compute_gae, ppo_update, create_train_state
from reward_shaping import RewardShaper

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]


def make_mep_pop_train(config):
    """Phase 1: Train MEP population with entropy bonus."""
    envs = [make("overcooked", layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    action_dim = envs[0].action_space(agents[0]).n
    n_layouts = len(LAYOUT_NAMES)
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * n_layouts
    pop_size = config["pop_size"]

    encoder = ObsEncoder(mode=config["obs_encoding"])
    reward_shaper = RewardShaper(mode=config["reward_mode"], magnifier=config["reward_magnifier"])
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
                        return (next_state, next_obs, rng), trans

                    rng, scan_rng = jax.random.split(rng)
                    carry_i, traj = jax.lax.scan(
                        _env_step,
                        (all_env_states[i], all_obs[i], scan_rng),
                        jnp.arange(config["num_steps"]),
                    )
                    new_states.append(carry_i[0])
                    new_obs_list.append(carry_i[1])
                    all_traj.append(traj)

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
    """Phase 2: Train ego agent against MEP population (uniform random partner)."""
    envs = [make("overcooked", layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * len(LAYOUT_NAMES)

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

        def _update_step(runner_state, update_idx):
            train_state, all_env_states, all_obs, rng = runner_state

            # Uniform random partner from population
            rng, partner_rng = jax.random.split(rng)
            partner_idx = jax.random.randint(partner_rng, (), 0, pop_size)
            partner_params = jax.tree.map(lambda x: x[partner_idx], pop_params)

            all_traj = []
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
                    return (next_state, next_obs, rng), trans

                rng, scan_rng = jax.random.split(rng)
                carry, traj = jax.lax.scan(
                    _env_step,
                    (all_env_states[i], all_obs[i], scan_rng),
                    jnp.arange(config["num_steps"]),
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                all_traj.append(traj)

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

            metrics = {"mean_reward": jnp.mean(traj_batch.reward), "total_loss": loss_info["total_loss"]}
            return (train_state, all_env_states, all_obs, rng), metrics

        runner_state = (train_state, all_env_states, all_obs, rng)
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
        "obs_encoding": args.obs_encoding,
        "reward_mode": args.reward_mode,
        "reward_magnifier": args.reward_magnifier,
        "pop_size": args.pop_size,
        "mep_alpha": args.mep_alpha,
    }


def main():
    parser = argparse.ArgumentParser(description="MEP training (2-phase)")
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--mep-alpha", type=float, default=0.01)
    parser.add_argument("--obs-encoding", type=str, default="egocentric")
    parser.add_argument("--reward-mode", type=str, default="originaler")
    parser.add_argument("--reward-magnifier", type=float, default=3.0)
    args = parser.parse_args()

    config = get_config(args)

    # Phase 1: MEP population training
    pop_rounds = config["num_updates"] // config["epoch_updates"]
    print(f"[MEP] Phase 1: Training population (pop={args.pop_size}, "
          f"{pop_rounds} rounds × {args.epoch_updates} updates/round, "
          f"{args.pop_timesteps/1e6:.0f}M)")

    train_pop_fn, network, encoder = make_mep_pop_train(config)
    rng = jax.random.PRNGKey(args.seed)

    t0 = time.time()
    pop_result = train_pop_fn(rng)
    jax.block_until_ready(pop_result["all_params"])
    pop_time = time.time() - t0

    pop_rewards = pop_result["metrics"]["mean_reward"]
    print(f"[MEP] Phase 1 done in {pop_time:.1f}s ({pop_time/60:.1f} min). "
          f"Final mean reward: {pop_rewards[-1]:.4f}")

    # Phase 2: Ego agent training
    print(f"[MEP] Phase 2: Training ego agent ({config['ego_num_updates']} updates, "
          f"{args.ego_timesteps/1e6:.0f}M)")

    reward_shaper = RewardShaper(mode=config["reward_mode"], magnifier=config["reward_magnifier"])
    ego_train_fn = make_mep_ego_train(config, network, encoder, reward_shaper)

    pop_params = jax.device_get(pop_result["all_params"])
    pop_params = jax.device_put(pop_params)

    rng = jax.random.PRNGKey(args.seed + 3000)
    t0 = time.time()
    ego_result = ego_train_fn(rng, pop_params)
    jax.block_until_ready(ego_result)
    ego_time = time.time() - t0

    ego_rewards = ego_result["metrics"]["mean_reward"]
    print(f"[MEP] Phase 2 done in {ego_time:.1f}s ({ego_time/60:.1f} min). "
          f"Final mean reward: {ego_rewards[-1]:.4f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "ego_params": jax.device_get(ego_result["train_state"].params),
        "pop_params": jax.device_get(pop_result["all_params"]),
        "config": config,
        "pop_metrics": jax.device_get(pop_result["metrics"]),
        "ego_metrics": jax.device_get(ego_result["metrics"]),
    }
    ckpt_path = f"checkpoints/mep{args.pop_size}_seed{args.seed}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[MEP] Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
