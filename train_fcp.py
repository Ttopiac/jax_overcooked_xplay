"""Fictitious Co-Play (FCP) training in JAX.

Loads a pre-trained SP population with checkpoints (from train_sp_population.py),
categorizes checkpoints into High/Medium/Low performance tiers, and trains a new
FCP agent against the diverse checkpoint pool using curriculum scheduling.

Curriculum (matching original multiHRI):
  - 50% of training: LOW tier partners
  - 12.5%: MEDIUM tier partners
  - 12.5%: HIGH tier partners
  - 25%: random mix (40% LOW, 30% MED, 30% HIGH)
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
from ppo import Transition, compute_gae, ppo_update, create_train_state


def flatten_obs(obs, num_envs):
    return obs.reshape(num_envs, -1).astype(jnp.float32) 


def categorize_checkpoints(ckpt_params, ckpt_rewards):
    """Categorize checkpoint params into High/Medium/Low tiers.

    Args:
        ckpt_params: tree of (pop_size, num_checkpoints, ...) - all checkpoint params
        ckpt_rewards: (pop_size, num_checkpoints) - reward at each checkpoint

    Returns:
        high_params, mid_params, low_params: each stacked as tree of (n_tier, ...)
    """
    # Flatten: (pop_size * num_checkpoints,)
    flat_rewards = ckpt_rewards.reshape(-1)
    n_total = flat_rewards.shape[0]
    third = max(1, n_total // 3)

    # Sort indices by reward (descending)
    sorted_idx = jnp.argsort(-flat_rewards)
    high_idx = sorted_idx[:third]
    mid_idx = sorted_idx[third : 2 * third]
    low_idx = sorted_idx[2 * third :]

    # Flatten params: tree of (pop_size * num_checkpoints, ...)
    flat_params = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]), ckpt_params
    )

    high_params = jax.tree.map(lambda x: x[high_idx], flat_params)
    mid_params = jax.tree.map(lambda x: x[mid_idx], flat_params)
    low_params = jax.tree.map(lambda x: x[low_idx], flat_params)

    return high_params, mid_params, low_params


def make_fcp_train(config, network, obs_flat_dim, env, agents):
    """Train an FCP agent against a pool of diverse checkpoints with curriculum."""

    total_updates = config["num_updates"]
    phase1_end = int(0.50 * total_updates)
    phase2_end = int(0.625 * total_updates)
    phase3_end = int(0.75 * total_updates)

    def train(rng, high_params, mid_params, low_params):
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        train_state = create_train_state(init_rng, network, (obs_flat_dim,), config["lr"])

        env_rngs = jax.random.split(env_rng, config["num_envs"])
        obs, env_state = jax.vmap(env.reset)(env_rngs)

        n_high = jax.tree.leaves(high_params)[0].shape[0]
        n_mid = jax.tree.leaves(mid_params)[0].shape[0]
        n_low = jax.tree.leaves(low_params)[0].shape[0]

        def _select_partner(step, rng):
            """Select partner params based on curriculum phase."""
            rng, tier_rng, idx_rng = jax.random.split(rng, 3)

            mix_tier = jax.random.choice(
                tier_rng, jnp.array([0, 1, 2]), p=jnp.array([0.4, 0.3, 0.3])
            )
            tier = jnp.where(
                step < phase1_end, 0,
                jnp.where(step < phase2_end, 1,
                    jnp.where(step < phase3_end, 2, mix_tier)),
            )

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

        def _update_step(runner_state, step):
            train_state, env_state, obs, rng = runner_state

            rng, partner_rng = jax.random.split(rng)
            partner_params = _select_partner(step, partner_rng)

            def _env_step(carry, unused):
                env_state, obs, rng = carry
                rng, a_rng0, a_rng1, s_rng = jax.random.split(rng, 4)

                obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
                obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])

                pi_0, val_0 = network.apply(train_state.params, obs_0)
                pi_1, _ = network.apply(partner_params, obs_1)

                a0 = pi_0.sample(seed=a_rng0)
                a1 = pi_1.sample(seed=a_rng1)
                lp0 = pi_0.log_prob(a0)

                actions = {agents[0]: a0, agents[1]: a1}
                step_rngs = jax.random.split(s_rng, config["num_envs"])
                next_obs, next_state, rewards, dones, infos = jax.vmap(env.step)(
                    step_rngs, env_state, actions
                )

                trans = Transition(
                    obs_0, a0, rewards[agents[0]], dones["__all__"], lp0, val_0
                )
                return (next_state, next_obs, rng), trans

            carry, traj_batch = jax.lax.scan(
                _env_step, (env_state, obs, rng), None, length=config["num_steps"]
            )
            env_state, obs, rng = carry

            obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
            _, last_val = network.apply(train_state.params, obs_0)

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
            return (train_state, env_state, obs, rng), metrics

        runner_state = (train_state, env_state, obs, rng)
        steps = jnp.arange(total_updates)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, steps)
        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    num_envs = args.num_envs
    num_steps = args.num_steps
    total = args.total_timesteps
    num_updates = total // (num_steps * num_envs)
    batch_size = num_steps * num_envs
    minibatch_size = min(args.minibatch_size, batch_size)

    return {
        "layout_name": args.layout,
        "num_envs": num_envs,
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
    }


def main():
    parser = argparse.ArgumentParser(description="FCP training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--total-timesteps", type=int, default=60_000_000,
                        help="Timesteps for FCP agent training (SP pop loaded separately)")
    parser.add_argument("--num-envs", type=int, default=32)
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
    args = parser.parse_args()

    config = get_config(args)

    # --- Load shared SP population ---
    pop_path = f"checkpoints/sp_pop_seed{args.seed}_{args.layout}.pkl"
    print(f"[FCP] Loading SP population from {pop_path}")
    with open(pop_path, "rb") as f:
        pop_data = pickle.load(f)

    ckpt_params = jax.device_put(pop_data["ckpt_params"])   # (pop, num_ckpts, ...)
    ckpt_rewards = jax.device_put(pop_data["ckpt_rewards"])  # (pop, num_ckpts)

    # Categorize into H/M/L
    high_params, mid_params, low_params = categorize_checkpoints(ckpt_params, ckpt_rewards)
    n_h = jax.tree.leaves(high_params)[0].shape[0]
    n_m = jax.tree.leaves(mid_params)[0].shape[0]
    n_l = jax.tree.leaves(low_params)[0].shape[0]
    print(f"[FCP] Checkpoint tiers: HIGH={n_h} MED={n_m} LOW={n_l}")

    # --- Setup env and network ---
    layout = overcooked_layouts[config["layout_name"]]
    env = make("overcooked", layout=layout)
    agent_names = env.agents

    sample_obs, _ = env.reset(jax.random.PRNGKey(0))
    obs_flat_dim = int(jnp.prod(jnp.array(sample_obs[agent_names[0]].shape)))
    action_dim = env.action_space(agent_names[0]).n
    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    # --- Train FCP agent ---
    print(f"[FCP] Training FCP agent ({config['num_updates']} updates, "
          f"{args.total_timesteps/1e6:.0f}M timesteps)")
    fcp_train_fn = make_fcp_train(config, network, obs_flat_dim, env, agent_names)

    rng = jax.random.PRNGKey(args.seed + 2000)
    t0 = time.time()
    result = fcp_train_fn(rng, high_params, mid_params, low_params)
    jax.block_until_ready(result)
    elapsed = time.time() - t0

    rewards = result["metrics"]["mean_reward"]
    print(f"[FCP] Done in {elapsed:.1f}s ({elapsed/60:.1f} min). "
          f"Final mean reward: {rewards[-1]:.4f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "params": jax.device_get(result["train_state"].params),
        "config": config,
        "metrics": jax.device_get(result["metrics"]),
    }
    ckpt_path = f"checkpoints/fcp_seed{args.seed}_{args.layout}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[FCP] Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
