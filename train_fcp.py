"""Fictitious Co-Play (FCP) training in JAX.

Loads a pre-trained SP population with checkpoints, categorizes into
High/Medium/Low performance tiers, and trains a new FCP agent against
the diverse pool with uniform random partner selection (33/33/33).
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


def categorize_checkpoints(ckpt_params, ckpt_rewards):
    """Categorize checkpoint params into High/Medium/Low tiers."""
    flat_rewards = ckpt_rewards.reshape(-1)
    n_total = flat_rewards.shape[0]
    third = max(1, n_total // 3)

    sorted_idx = jnp.argsort(-flat_rewards)
    high_idx = sorted_idx[:third]
    mid_idx = sorted_idx[third : 2 * third]
    low_idx = sorted_idx[2 * third :]

    flat_params = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), ckpt_params)

    high_params = jax.tree.map(lambda x: x[high_idx], flat_params)
    mid_params = jax.tree.map(lambda x: x[mid_idx], flat_params)
    low_params = jax.tree.map(lambda x: x[low_idx], flat_params)

    return high_params, mid_params, low_params


def make_fcp_train(config, network, encoder, reward_shaper):
    envs = [make("overcooked", layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * len(LAYOUT_NAMES)

    def train(rng, high_params, mid_params, low_params):
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

        n_high = jax.tree.leaves(high_params)[0].shape[0]
        n_mid = jax.tree.leaves(mid_params)[0].shape[0]
        n_low = jax.tree.leaves(low_params)[0].shape[0]

        def _select_partner(rng):
            """Uniform random: 33% LOW, 33% MED, 33% HIGH."""
            rng, tier_rng, idx_rng = jax.random.split(rng, 3)
            tier = jax.random.randint(tier_rng, (), 0, 3)  # 0=LOW, 1=MED, 2=HIGH

            low_idx = jax.random.randint(idx_rng, (), 0, n_low)
            mid_idx = jax.random.randint(idx_rng, (), 0, n_mid)
            high_idx = jax.random.randint(idx_rng, (), 0, n_high)

            partner = jax.tree.map(
                lambda lo, md, hi: jnp.where(
                    tier == 0, lo[low_idx],
                    jnp.where(tier == 1, md[mid_idx], hi[high_idx]),
                ),
                low_params, mid_params, high_params,
            )
            return partner

        def _update_step(runner_state, update_idx):
            train_state, all_env_states, all_obs, rng = runner_state

            rng, partner_rng = jax.random.split(rng)
            partner_params = _select_partner(partner_rng)

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
    n_env_total = n_env_l * len(LAYOUT_NAMES)
    num_steps = args.num_steps
    num_updates = args.total_timesteps // (num_steps * n_env_total)
    batch_size = num_steps * n_env_total
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
        "reward_shaping_factor": args.reward_shaping_factor,
        "reward_shaping_horizon": args.reward_shaping_horizon,
    }


def main():
    parser = argparse.ArgumentParser(description="FCP training")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated seeds, e.g. '0,1,2'")
    parser.add_argument("--total-timesteps", type=int, default=60_000_000)
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
    parser.add_argument("--obs-encoding", type=str, default="egocentric")
    parser.add_argument("--reward-mode", type=str, default="shaped")
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--reward-shaping-horizon", type=float, default=float("inf"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    config = get_config(args)

    # Load per-seed SP population and categorize
    all_high, all_mid, all_low = [], [], []
    for seed in seeds:
        pop_path = f"checkpoints/sp_pop_seed{seed}.pkl"
        print(f"[FCP] Loading {pop_path}")
        with open(pop_path, "rb") as f:
            pop_data = pickle.load(f)
        ckpt_params = jax.device_put(pop_data["ckpt_params"])
        ckpt_rewards = jax.device_put(pop_data["ckpt_rewards"])
        h, m, l = categorize_checkpoints(ckpt_params, ckpt_rewards)
        all_high.append(h)
        all_mid.append(m)
        all_low.append(l)

    n_h = jax.tree.leaves(all_high[0])[0].shape[0]
    n_m = jax.tree.leaves(all_mid[0])[0].shape[0]
    n_l = jax.tree.leaves(all_low[0])[0].shape[0]
    print(f"[FCP] Tiers per seed: HIGH={n_h} MED={n_m} LOW={n_l}")

    # Stack per-seed tier params: (n_seeds, n_tier, ...)
    stacked_high = jax.tree.map(lambda *xs: jnp.stack(xs), *all_high)
    stacked_mid = jax.tree.map(lambda *xs: jnp.stack(xs), *all_mid)
    stacked_low = jax.tree.map(lambda *xs: jnp.stack(xs), *all_low)

    encoder = ObsEncoder(mode=config["obs_encoding"])
    reward_shaper = RewardShaper(
        mode=config["reward_mode"],
        reward_shaping_factor=config["reward_shaping_factor"],
        reward_shaping_horizon=config["reward_shaping_horizon"],
    )
    action_dim = make("overcooked", layout=overcooked_layouts[LAYOUT_NAMES[0]]).action_space("agent_0").n
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    fcp_train_fn = make_fcp_train(config, network, encoder, reward_shaper)
    multi_fcp_fn = jax.jit(jax.vmap(fcp_train_fn))

    rngs = jnp.stack([jax.random.PRNGKey(s + 2000) for s in seeds])

    print(f"[FCP] Training {len(seeds)} seeds vmapped ({config['num_updates']} updates, "
          f"{args.total_timesteps/1e6:.0f}M)")
    t0 = time.time()
    results = multi_fcp_fn(rngs, stacked_high, stacked_mid, stacked_low)
    jax.block_until_ready(results)
    elapsed = time.time() - t0

    os.makedirs("checkpoints", exist_ok=True)
    for i, seed in enumerate(seeds):
        r = float(results["metrics"]["mean_reward"][i, -1])
        print(f"  seed {seed}: final reward = {r:.4f}")
        ckpt = {
            "params": jax.device_get(jax.tree.map(lambda x: x[i], results["train_state"].params)),
            "config": config,
            "metrics": jax.device_get(jax.tree.map(lambda x: x[i], results["metrics"])),
        }
        with open(f"checkpoints/fcp_seed{seed}.pkl", "wb") as f:
            pickle.dump(ckpt, f)

    print(f"[FCP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
