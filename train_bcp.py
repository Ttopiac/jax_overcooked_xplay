"""Behavioral Cloning Play (BCP) with N-XPlay support.

N-X ego/clones use ego network (3×64).
X partner slots use BC network (2×64) or ego (self-play), based on bc_schedule.
Only trains on 3 layouts with human data.
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
from nxplay import select_partners, shuffle_agent_positions
from train_bc import BC_HIDDEN_DIMS, BC_LAYOUTS

BCP_JAXMARL_LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring"]


def make_bcp_train(config, ego_network, bc_network, encoder, reward_shaper, n_bc_agents):
    num_agents = config["num_agents"]
    num_unseen = config["num_unseen"]
    num_ego = num_agents - num_unseen
    homo = config["xplay_mode"] == "homo"

    envs = [Overcooked(layout=overcooked_layouts[name], num_agents=num_agents)
            for name in BCP_JAXMARL_LAYOUTS]
    agent_names = envs[0].agents
    n_layouts = len(BCP_JAXMARL_LAYOUTS)
    n_env_l = config["num_envs_per_layout"]
    n_env_total = n_env_l * n_layouts

    def train(rng, all_bc_params):
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

        bc_horizon_steps = config["bc_anneal_horizon"]

        def _update_step(runner_state, update_idx):
            train_state, all_env_states, all_obs, rng, cumulative_steps = runner_state

            # bc_factor annealing
            bc_factor = jnp.where(bc_horizon_steps <= 0, 1.0,
                                  jnp.minimum(cumulative_steps / bc_horizon_steps, 1.0))
            rng, coin_rng, partner_rng, shuffle_rng = jax.random.split(rng, 4)
            use_bc = jax.random.uniform(coin_rng) < bc_factor

            # Select BC partners
            bc_partner_params = select_partners(partner_rng, all_bc_params, num_unseen, homo=homo)

            # Shuffle positions
            all_env_states_s = tuple(
                jax.vmap(lambda s, k: shuffle_agent_positions(k, s, num_agents))(
                    all_env_states[i], jax.random.split(shuffle_rng, n_env_l)
                ) for i in range(len(envs))
            )

            all_ego_traj = []
            all_sparse = []
            new_states = []
            new_obs_list = []

            for layout_i, env in enumerate(envs):

                def _env_step(carry, inner_step, _env=env):
                    env_state, obs, rng = carry
                    rng, action_rng, step_rng = jax.random.split(rng, 3)
                    action_rngs = jax.random.split(action_rng, num_agents)

                    all_obs_enc = [encoder.encode(obs, env_state, k, n_env_l) for k in range(num_agents)]

                    ego_transitions = []
                    actions_list = []

                    # Ego/clone agents
                    for k in range(num_ego):
                        pi, val = ego_network.apply(train_state.params, all_obs_enc[k])
                        a = pi.sample(seed=action_rngs[k])
                        lp = pi.log_prob(a)
                        ego_transitions.append((all_obs_enc[k], a, lp, val))
                        actions_list.append(a)

                    # Partner agents: BC or self-play based on bc_factor
                    for j in range(num_unseen):
                        p_params = jax.tree.map(lambda x: x[j], bc_partner_params)
                        bc_pi, _ = bc_network.apply(p_params, all_obs_enc[num_ego + j])
                        sp_pi, _ = ego_network.apply(train_state.params, all_obs_enc[num_ego + j])
                        a_bc = bc_pi.sample(seed=action_rngs[num_ego + j])
                        a_sp = sp_pi.sample(seed=action_rngs[num_ego + j])
                        a = jnp.where(use_bc, a_bc, a_sp)
                        actions_list.append(a)

                    actions_dict = {agent_names[k]: actions_list[k] for k in range(num_agents)}
                    step_rngs = jax.random.split(step_rng, n_env_l)
                    next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                        step_rngs, env_state, actions_dict
                    )

                    step_count = cumulative_steps + inner_step * n_env_total
                    done_all = dones["__all__"]

                    transitions = []
                    for k in range(num_ego):
                        obs_k, a_k, lp_k, val_k = ego_transitions[k]
                        reward_k = reward_shaper.compute(rewards, infos, agent_names[k], step_count)
                        transitions.append(Transition(obs_k, a_k, reward_k, done_all, lp_k, val_k))

                    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
                    sparse_r = rewards[agent_names[0]]
                    return (next_state, next_obs, rng), (stacked, sparse_r)

                rng, scan_rng = jax.random.split(rng)
                carry, (traj_stacked, sparse_r) = jax.lax.scan(
                    _env_step,
                    (all_env_states_s[layout_i], all_obs[layout_i], scan_rng),
                    jnp.arange(config["num_steps"]),
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                traj_flat = jax.tree.map(lambda x: x.reshape(x.shape[0], -1, *x.shape[3:]), traj_stacked)
                all_ego_traj.append(traj_flat)
                all_sparse.append(sparse_r)

            all_env_states = tuple(new_states)
            all_obs = tuple(new_obs_list)
            cumulative_steps = cumulative_steps + config["num_steps"] * n_env_total

            traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_ego_traj)

            all_last_vals = []
            for layout_i in range(len(envs)):
                for k in range(num_ego):
                    o = encoder.encode(all_obs[layout_i], all_env_states[layout_i], k, n_env_l)
                    _, lv = ego_network.apply(train_state.params, o)
                    all_last_vals.append(lv)
            last_val = jnp.concatenate(all_last_vals)

            advantages, returns = compute_gae(traj_batch, last_val, config["gamma"], config["gae_lambda"])
            flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj_batch)
            flat_adv, flat_ret = advantages.reshape(-1), returns.reshape(-1)

            rng, update_rng = jax.random.split(rng)
            train_state, loss_info = ppo_update(train_state, flat_batch, flat_adv, flat_ret, config, update_rng)

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
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config["num_updates"]))
        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    n_env_l = args.num_envs_per_layout
    n_layouts = len(BCP_JAXMARL_LAYOUTS)
    n_env_total = n_env_l * n_layouts
    num_ego = args.num_agents - args.num_unseen
    num_updates = args.total_timesteps // (args.num_steps * n_env_total)
    batch_size = args.num_steps * n_env_total * num_ego
    minibatch_size = min(args.minibatch_size, batch_size)

    return {
        "num_agents": args.num_agents,
        "num_unseen": args.num_unseen,
        "xplay_mode": args.xplay_mode,
        "num_envs_per_layout": n_env_l,
        "num_steps": args.num_steps,
        "num_updates": num_updates,
        "lr": args.lr,
        "gamma": args.gamma, "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps, "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef, "num_epochs": args.num_epochs,
        "minibatch_size": minibatch_size,
        "hidden_dims": tuple([args.hidden_dim] * args.num_layers),
        "reward_mode": args.reward_mode,
        "reward_shaping_factor": args.reward_shaping_factor,
        "reward_shaping_horizon": args.reward_shaping_horizon,
        "bc_anneal_horizon": args.bc_anneal_horizon,
    }


def main():
    parser = argparse.ArgumentParser(description="BCP with N-XPlay")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--total-timesteps", type=int, default=60_000_000)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--num-unseen", type=int, default=1)
    parser.add_argument("--xplay-mode", type=str, default="hetero", choices=["hetero", "homo"])
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
    parser.add_argument("--reward-mode", type=str, default="shaped")
    parser.add_argument("--reward-shaping-factor", type=float, default=1.0)
    parser.add_argument("--reward-shaping-horizon", type=float, default=float("inf"))
    parser.add_argument("--bc-anneal-horizon", type=int, default=8000)
    setup_wandb_args(parser)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    config = get_config(args)

    print(f"[BCP] N={args.num_agents}, X={args.num_unseen}, mode={args.xplay_mode}")

    # Load BC populations
    all_seed_bc_params = []
    bc_hidden_dims = None
    for seed in seeds:
        bc_path = f"checkpoints/bc_seed{seed}.pkl"
        print(f"  Loading {bc_path}")
        with open(bc_path, "rb") as f:
            bc_data = pickle.load(f)
        bc_hidden_dims = bc_data["bc_hidden_dims"]
        bc_list = []
        for layout in bc_data["layouts"]:
            bc_list.extend(bc_data["bc_params"][layout])
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *bc_list)
        all_seed_bc_params.append(stacked)

    total_bc = jax.tree.leaves(all_seed_bc_params[0])[0].shape[0]
    stacked_all_bc = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_bc_params)
    stacked_all_bc = jax.device_put(stacked_all_bc)

    encoder = ObsEncoder()
    reward_shaper = RewardShaper(mode=config["reward_mode"],
                                 reward_shaping_factor=config["reward_shaping_factor"],
                                 reward_shaping_horizon=config["reward_shaping_horizon"])
    ego_network = ActorCritic(action_dim=6, hidden_dims=config["hidden_dims"])
    bc_network = ActorCritic(action_dim=6, hidden_dims=bc_hidden_dims)

    bcp_train_fn = make_bcp_train(config, ego_network, bc_network, encoder, reward_shaper, total_bc)
    multi_bcp_fn = jax.jit(jax.vmap(bcp_train_fn))

    rngs = jnp.stack([jax.random.PRNGKey(s + 1000) for s in seeds])

    print(f"[BCP] Training {len(seeds)} seeds ({config['num_updates']} updates)")
    t0 = time.time()
    results = multi_bcp_fn(rngs, stacked_all_bc)
    jax.block_until_ready(results)
    elapsed = time.time() - t0

    os.makedirs("checkpoints", exist_ok=True)
    tags = args.wandb_tags.split(",") if args.wandb_tags else ["bcp"]
    for i, seed in enumerate(seeds):
        r = float(results["metrics"]["mean_reward"][i, -1])
        sr = float(results["metrics"]["mean_sparse_reward"][i, -1])
        print(f"  seed {seed}: reward={r:.4f}, sparse={sr:.4f}")
        metrics_i = jax.device_get(jax.tree.map(lambda x: x[i], results["metrics"]))
        ckpt = {
            "params": jax.device_get(jax.tree.map(lambda x: x[i], results["train_state"].params)),
            "config": config, "metrics": metrics_i,
        }
        with open(f"checkpoints/bcp_n{args.num_agents}x{args.num_unseen}_{args.xplay_mode}_seed{seed}.pkl", "wb") as f:
            pickle.dump(ckpt, f)
        logger = WandbLogger(project=args.wandb_project,
                             name=f"bcp_n{args.num_agents}x{args.num_unseen}_{args.xplay_mode}_seed{seed}",
                             config={**config, "seed": seed, "method": "BCP"},
                             enabled=args.wandb, group=args.wandb_group, tags=tags)
        logger.log_training_curves(metrics_i, prefix="train")
        logger.log_summary({"final_reward": r, "final_sparse": sr, "train_time_s": elapsed})
        logger.finish()

    print(f"[BCP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
