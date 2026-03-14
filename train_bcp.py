"""Behavioral Cloning Play (BCP) training in JAX.

Matches Carroll et al. (2019):
  1. Load pre-trained BC population (from train_bc.py) — 3 BC agents per layout
  2. Train PPO agent on ONLY the 3 layouts with human data
  3. bc_schedule: anneal bc_factor 0→1 over bc_anneal_horizon ENV TIMESTEPS
  4. Each update: coin flip → partner is random BC agent or self-play

BC uses 64×64 MLP architecture (matching Carroll).
RL ego uses 256×256 MLP (matching all other methods).
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
from wandb_utils import WandbLogger, setup_wandb_args
from train_bc import BC_HIDDEN_DIMS, BC_LAYOUTS

# BCP only trains on layouts with human data
# JaxMARL layout names for the 3 Carroll layouts
BCP_JAXMARL_LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring"]


def make_bcp_train(config, ego_network, bc_network, encoder, reward_shaper, n_bc_agents):
    """Build BCP training function.

    ego_network: 256×256 MLP for the ego agent
    bc_network: 64×64 MLP for BC agents (different architecture!)
    """
    envs = [make("overcooked", layout=overcooked_layouts[name]) for name in BCP_JAXMARL_LAYOUTS]
    agents = envs[0].agents
    n_layouts = len(BCP_JAXMARL_LAYOUTS)
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * n_layouts

    def train(rng, all_bc_params):
        """
        all_bc_params: stacked BC params, tree of (n_bc_agents, ...)
        """
        rng, init_rng = jax.random.split(rng)
        train_state = create_train_state(init_rng, ego_network, (encoder.obs_flat_dim,), config["lr"])

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

        bc_horizon_steps = config["bc_anneal_horizon"]  # in env timesteps

        def _update_step(runner_state, update_idx):
            train_state, all_env_states, all_obs, rng, cumulative_steps = runner_state

            # bc_factor based on cumulative ENV TIMESTEPS (not PPO updates)
            bc_factor = jnp.where(
                bc_horizon_steps <= 0,
                1.0,
                jnp.minimum(cumulative_steps / bc_horizon_steps, 1.0),
            )

            # Coin flip: use BC or self-play?
            rng, coin_rng, bc_select_rng = jax.random.split(rng, 3)
            use_bc = jax.random.uniform(coin_rng) < bc_factor

            # Select random BC agent from population
            bc_idx = jax.random.randint(bc_select_rng, (), 0, n_bc_agents)
            bc_params = jax.tree.map(lambda x: x[bc_idx], all_bc_params)

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

                    # Ego: 256×256 network
                    pi_0, val_0 = ego_network.apply(train_state.params, obs_0)

                    # Partner: BC (64×64) or self-play (256×256)
                    bc_pi, _ = bc_network.apply(bc_params, obs_1)
                    sp_pi, _ = ego_network.apply(train_state.params, obs_1)

                    # Select partner action based on use_bc flag
                    a0 = pi_0.sample(seed=a0r)
                    a1_bc = bc_pi.sample(seed=a1r)
                    a1_sp = sp_pi.sample(seed=a1r)
                    a1 = jnp.where(use_bc, a1_bc, a1_sp)
                    lp0 = pi_0.log_prob(a0)

                    actions = {agents[0]: a0, agents[1]: a1}
                    step_rngs = jax.random.split(sr, n_env_l)
                    next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                        step_rngs, env_state, actions
                    )

                    step_count = cumulative_steps + inner_step * n_env_total
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

            # Update cumulative env steps
            steps_this_update = config["num_steps"] * n_env_total
            cumulative_steps = cumulative_steps + steps_this_update

            traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_traj)

            all_last_vals = []
            for i in range(len(envs)):
                o0 = encoder.encode(all_obs[i], all_env_states[i], 0, n_env_l)
                _, lv = ego_network.apply(train_state.params, o0)
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
                "bc_factor": bc_factor,
            }
            return (train_state, all_env_states, all_obs, rng, cumulative_steps), metrics

        cumulative_steps = jnp.int32(0)
        runner_state = (train_state, all_env_states, all_obs, rng, cumulative_steps)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["num_updates"])
        )
        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    n_env_l = args.num_envs_per_layout
    n_layouts = len(BCP_JAXMARL_LAYOUTS)
    n_env_total = n_env_l * n_layouts
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
        "reward_mode": args.reward_mode,
        "reward_shaping_factor": args.reward_shaping_factor,
        "reward_shaping_horizon": args.reward_shaping_horizon,
        "bc_anneal_horizon": args.bc_anneal_horizon,
    }


def main():
    parser = argparse.ArgumentParser(description="BCP training (Carroll-style)")
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
    parser.add_argument("--bc-anneal-horizon", type=int, default=8000,
                        help="Env timesteps over which bc_factor anneals 0->1. 0=always BC.")
    setup_wandb_args(parser)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    config = get_config(args)

    # Load per-seed BC populations and stack
    all_seed_bc_params = []
    bc_hidden_dims = None
    for seed in seeds:
        bc_path = f"checkpoints/bc_seed{seed}.pkl"
        print(f"[BCP] Loading {bc_path}")
        with open(bc_path, "rb") as f:
            bc_data = pickle.load(f)
        bc_hidden_dims = bc_data["bc_hidden_dims"]
        bc_params_list = []
        for layout in bc_data["layouts"]:
            bc_params_list.extend(bc_data["bc_params"][layout])
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *bc_params_list)
        all_seed_bc_params.append(stacked)

    total_bc = jax.tree.leaves(all_seed_bc_params[0])[0].shape[0]
    # Stack: (n_seeds, total_bc, ...)
    stacked_all_bc = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_bc_params)
    stacked_all_bc = jax.device_put(stacked_all_bc)
    print(f"[BCP] {total_bc} BC agents/seed, BC arch={bc_hidden_dims}, "
          f"layouts={BCP_JAXMARL_LAYOUTS}")

    encoder = ObsEncoder()
    reward_shaper = RewardShaper(
        mode=config["reward_mode"],
        reward_shaping_factor=config["reward_shaping_factor"],
        reward_shaping_horizon=config["reward_shaping_horizon"],
    )
    ego_network = ActorCritic(action_dim=6, hidden_dims=config["hidden_dims"])
    bc_network = ActorCritic(action_dim=6, hidden_dims=bc_hidden_dims)

    bcp_train_fn = make_bcp_train(
        config, ego_network, bc_network, encoder, reward_shaper, total_bc
    )
    multi_bcp_fn = jax.jit(jax.vmap(bcp_train_fn))

    rngs = jnp.stack([jax.random.PRNGKey(s + 1000) for s in seeds])

    print(f"[BCP] Training {len(seeds)} seeds vmapped ({config['num_updates']} updates, "
          f"{args.total_timesteps/1e6:.0f}M)")
    t0 = time.time()
    results = multi_bcp_fn(rngs, stacked_all_bc)
    jax.block_until_ready(results)
    elapsed = time.time() - t0

    os.makedirs("checkpoints", exist_ok=True)
    tags = args.wandb_tags.split(",") if args.wandb_tags else ["bcp"]
    for i, seed in enumerate(seeds):
        r = float(results["metrics"]["mean_reward"][i, -1])
        print(f"  seed {seed}: final reward = {r:.4f}")
        metrics_i = jax.device_get(jax.tree.map(lambda x: x[i], results["metrics"]))
        ckpt = {
            "params": jax.device_get(jax.tree.map(lambda x: x[i], results["train_state"].params)),
            "config": config,
            "metrics": metrics_i,
        }
        with open(f"checkpoints/bcp_seed{seed}.pkl", "wb") as f:
            pickle.dump(ckpt, f)

        logger = WandbLogger(
            project=args.wandb_project, name=f"bcp_seed{seed}",
            config={**config, "seed": seed, "method": "BCP"},
            enabled=args.wandb, group=args.wandb_group, tags=tags,
        )
        logger.log_training_curves(metrics_i, prefix="train")
        logger.log_summary({"final_reward": r, "train_time_s": elapsed})
        logger.finish()

    print(f"[BCP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
