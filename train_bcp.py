"""Behavioral Cloning Play (BCP) training in JAX.

Matches Carroll et al. (2019):
  1. Load pre-trained BC model (from train_bc.py)
  2. Train PPO agent with bc_schedule:
     - Start with self-play (bc_factor=0)
     - Linearly anneal to bc_factor=1.0 over bc_anneal_horizon steps
     - After annealing: always paired with BC partner

At each PPO update, flip a coin with probability bc_factor:
  - If < bc_factor: partner uses frozen BC params
  - If >= bc_factor: partner uses ego's own params (self-play)
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


def make_bcp_train(config, network, encoder, reward_shaper):
    envs = [make("overcooked", layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * len(LAYOUT_NAMES)

    def train(rng, bc_params):
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

        bc_horizon = config["bc_anneal_horizon"]

        def _get_bc_factor(update_idx):
            """Linear anneal: 0 -> 1 over bc_anneal_horizon updates."""
            if bc_horizon <= 0:
                return 1.0  # Always BC
            frac = jnp.minimum(update_idx / bc_horizon, 1.0)
            return frac

        def _update_step(runner_state, update_idx):
            train_state, all_env_states, all_obs, rng = runner_state

            # Determine partner: BC or self-play
            rng, coin_rng = jax.random.split(rng)
            bc_factor = _get_bc_factor(update_idx)
            use_bc = jax.random.uniform(coin_rng) < bc_factor

            # Partner params: BC if use_bc, else ego's own params
            partner_params = jax.tree.map(
                lambda bc, ego: jnp.where(use_bc, bc, ego),
                bc_params, train_state.params,
            )

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
                "bc_factor": bc_factor,
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
        "bc_anneal_horizon": args.bc_anneal_horizon,
    }


def main():
    parser = argparse.ArgumentParser(description="BCP training (Carroll-style)")
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--obs-encoding", type=str, default="egocentric")
    parser.add_argument("--reward-mode", type=str, default="shaped")
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--reward-shaping-horizon", type=float, default=float("inf"))
    parser.add_argument("--bc-anneal-horizon", type=int, default=0,
                        help="PPO updates over which bc_factor anneals 0->1. 0=always BC.")
    parser.add_argument("--bc-model", type=str, default="combined",
                        help="Which BC model to use: layout name or 'combined'")
    args = parser.parse_args()

    config = get_config(args)

    # Load BC model
    bc_path = f"checkpoints/bc_seed{args.seed}.pkl"
    print(f"[BCP] Loading BC model from {bc_path}")
    with open(bc_path, "rb") as f:
        bc_data = pickle.load(f)

    bc_params = jax.device_put(bc_data["bc_params"][args.bc_model])
    print(f"[BCP] Using BC model: {args.bc_model}")

    # Setup
    encoder = ObsEncoder(mode=config["obs_encoding"])
    reward_shaper = RewardShaper(
        mode=config["reward_mode"],
        reward_shaping_factor=config["reward_shaping_factor"],
        reward_shaping_horizon=config["reward_shaping_horizon"],
    )
    action_dim = make("overcooked", layout=overcooked_layouts[LAYOUT_NAMES[0]]).action_space("agent_0").n
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    horizon_str = f"anneal over {args.bc_anneal_horizon} updates" if args.bc_anneal_horizon > 0 else "always BC"
    print(f"[BCP] Training PPO_BC ({config['num_updates']} updates, "
          f"{args.total_timesteps/1e6:.0f}M, {horizon_str})")

    bcp_train_fn = make_bcp_train(config, network, encoder, reward_shaper)

    rng = jax.random.PRNGKey(args.seed + 1000)
    t0 = time.time()
    result = bcp_train_fn(rng, bc_params)
    jax.block_until_ready(result)
    elapsed = time.time() - t0

    rewards = result["metrics"]["mean_reward"]
    print(f"[BCP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min). "
          f"Final mean reward: {rewards[-1]:.4f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "params": jax.device_get(result["train_state"].params),
        "bc_model": args.bc_model,
        "config": config,
        "metrics": jax.device_get(result["metrics"]),
    }
    ckpt_path = f"checkpoints/bcp_seed{args.seed}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[BCP] Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
