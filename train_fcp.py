"""Fictitious Co-Play (FCP) training in JAX.

Two phases:
1. Train SP agents and save checkpoints at regular intervals.
   Categorize checkpoints into High/Medium/Low performance tiers.
2. Train a new FCP agent against the diverse checkpoint pool using
   curriculum scheduling:
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
import optax
from flax.training.train_state import TrainState
from jaxmarl import make
from jaxmarl.environments.overcooked import overcooked_layouts

from networks import ActorCritic
from ppo import Transition, compute_gae, ppo_update, create_train_state


def flatten_obs(obs, num_envs):
    return obs.reshape(num_envs, -1).astype(jnp.float32) / 255.0


def train_sp_with_checkpoints(config, network, env, agents, rng, num_checkpoints=15):
    """Train a single SP agent and save checkpoints at regular intervals.

    Returns:
        List of (params, eval_score) tuples.
    """
    obs_flat_dim = config["obs_flat_dim"]

    rng, init_rng, env_rng = jax.random.split(rng, 3)
    train_state = create_train_state(init_rng, network, (obs_flat_dim,), config["lr"])

    env_rngs = jax.random.split(env_rng, config["num_envs"])
    obs, env_state = jax.vmap(env.reset)(env_rngs)

    sp_updates = config["sp_num_updates"]
    ckpt_interval = max(1, sp_updates // num_checkpoints)

    def _make_train_chunk(chunk_len):
        """Create a JIT-compiled training function for a fixed chunk length."""
        @jax.jit
        def _train_chunk(runner_state):
            def _update_step(runner_state, unused):
                train_state, env_state, obs, rng = runner_state

                def _env_step(carry, unused):
                    env_state, obs, rng = carry
                    rng, a_rng0, a_rng1, s_rng = jax.random.split(rng, 4)

                    obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
                    obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])

                    pi_0, val_0 = network.apply(train_state.params, obs_0)
                    pi_1, val_1 = network.apply(train_state.params, obs_1)

                    a0 = pi_0.sample(seed=a_rng0)
                    a1 = pi_1.sample(seed=a_rng1)
                    lp0 = pi_0.log_prob(a0)
                    lp1 = pi_1.log_prob(a1)

                    actions_dict = {agents[0]: a0, agents[1]: a1}
                    step_rngs = jax.random.split(s_rng, config["num_envs"])
                    next_obs, next_state, rewards, dones, infos = jax.vmap(env.step)(
                        step_rngs, env_state, actions_dict
                    )

                    done_all = dones["__all__"]
                    reward = rewards[agents[0]]

                    t0 = Transition(obs_0, a0, reward, done_all, lp0, val_0)
                    t1 = Transition(obs_1, a1, reward, done_all, lp1, val_1)
                    return (next_state, next_obs, rng), (t0, t1)

                carry, (traj_0, traj_1) = jax.lax.scan(
                    _env_step, (env_state, obs, rng), None, length=config["num_steps"]
                )
                env_state, obs, rng = carry

                traj_batch = jax.tree.map(
                    lambda x, y: jnp.concatenate([x, y], axis=1), traj_0, traj_1
                )

                obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
                obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])
                _, lv0 = network.apply(train_state.params, obs_0)
                _, lv1 = network.apply(train_state.params, obs_1)
                last_val = jnp.concatenate([lv0, lv1])

                advantages, returns = compute_gae(
                    traj_batch, last_val, config["gamma"], config["gae_lambda"]
                )

                flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj_batch)
                flat_adv = advantages.reshape(-1)
                flat_ret = returns.reshape(-1)

                rng, update_rng = jax.random.split(rng)
                train_state_new, loss_info = ppo_update(
                    train_state, flat_batch, flat_adv, flat_ret, config, update_rng
                )

                metrics = {"mean_reward": jnp.mean(traj_batch.reward)}
                return (train_state_new, env_state, obs, rng), metrics

            runner_state, metrics = jax.lax.scan(
                _update_step, runner_state, None, length=chunk_len
            )
            return runner_state, metrics
        return _train_chunk

    # Train in chunks, saving checkpoints
    runner_state = (train_state, env_state, obs, rng)
    checkpoints = []
    updates_done = 0

    for ck_idx in range(num_checkpoints):
        chunk_len = min(ckpt_interval, sp_updates - updates_done)
        if chunk_len <= 0:
            break

        train_chunk_fn = _make_train_chunk(chunk_len)
        runner_state, metrics = train_chunk_fn(runner_state)
        jax.block_until_ready(runner_state[0].params)
        updates_done += chunk_len

        eval_score = float(metrics["mean_reward"].mean())
        ck_params = jax.device_get(runner_state[0].params)
        checkpoints.append((ck_params, eval_score))
        print(f"    SP checkpoint {ck_idx}: score={eval_score:.3f} "
              f"({updates_done}/{sp_updates} updates)")

    return checkpoints


def categorize_checkpoints(checkpoints):
    """Categorize checkpoints into High/Medium/Low tiers.

    Returns:
        (high_params_list, med_params_list, low_params_list)
    """
    sorted_cks = sorted(checkpoints, key=lambda x: x[1], reverse=True)
    n = len(sorted_cks)
    third = max(1, n // 3)

    high = [p for p, s in sorted_cks[:third]]
    mid = [p for p, s in sorted_cks[third : 2 * third]]
    low = [p for p, s in sorted_cks[2 * third :]]

    # Ensure each tier has at least one entry
    if not mid:
        mid = [sorted_cks[n // 2][0]]
    if not low:
        low = [sorted_cks[-1][0]]

    return high, mid, low


def stack_params(params_list):
    """Stack a list of params pytrees into a single pytree with leading axis."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *params_list)


def make_fcp_train(config, network, env, agents):
    """Train an FCP agent against a pool of diverse checkpoints with curriculum."""

    obs_flat_dim = config["obs_flat_dim"]

    def train(rng, high_params, mid_params, low_params):
        """
        Args:
            rng: JAX random key.
            high_params: stacked params, shape (n_high, ...).
            mid_params: stacked params, shape (n_mid, ...).
            low_params: stacked params, shape (n_low, ...).
        """
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        train_state = create_train_state(init_rng, network, (obs_flat_dim,), config["lr"])

        env_rngs = jax.random.split(env_rng, config["num_envs"])
        obs, env_state = jax.vmap(env.reset)(env_rngs)

        n_high = jax.tree.leaves(high_params)[0].shape[0]
        n_mid = jax.tree.leaves(mid_params)[0].shape[0]
        n_low = jax.tree.leaves(low_params)[0].shape[0]

        total_updates = config["fcp_num_updates"]
        # Curriculum boundaries (fraction of total updates)
        # Phase 1: 0-50% -> LOW
        # Phase 2: 50-62.5% -> MEDIUM
        # Phase 3: 62.5-75% -> HIGH
        # Phase 4: 75-100% -> mix (40% LOW, 30% MED, 30% HIGH)
        phase1_end = int(0.50 * total_updates)
        phase2_end = int(0.625 * total_updates)
        phase3_end = int(0.75 * total_updates)

        def _select_partner(step, rng):
            """Select partner params based on curriculum phase."""
            rng, tier_rng, idx_rng = jax.random.split(rng, 3)

            # Determine tier: 0=LOW, 1=MED, 2=HIGH
            # Phase 4: random mix
            mix_tier = jax.random.choice(
                tier_rng, jnp.array([0, 1, 2]), p=jnp.array([0.4, 0.3, 0.3])
            )
            tier = jnp.where(
                step < phase1_end,
                0,  # LOW
                jnp.where(
                    step < phase2_end,
                    1,  # MED
                    jnp.where(step < phase3_end, 2, mix_tier),  # HIGH or mix
                ),
            )

            # Select random params from the chosen tier
            low_idx = jax.random.randint(idx_rng, (), 0, n_low)
            mid_idx = jax.random.randint(idx_rng, (), 0, n_mid)
            high_idx = jax.random.randint(idx_rng, (), 0, n_high)

            partner = jax.tree.map(
                lambda lo, md, hi: jnp.where(
                    tier == 0,
                    lo[low_idx],
                    jnp.where(tier == 1, md[mid_idx], hi[high_idx]),
                ),
                low_params,
                mid_params,
                high_params,
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

                # Ego uses trainable params, partner uses selected checkpoint
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
            train_state_new, loss_info = ppo_update(
                train_state, flat_batch, flat_adv, flat_ret, config, update_rng
            )

            metrics = {
                "mean_reward": jnp.mean(traj_batch.reward),
                "total_loss": loss_info["total_loss"],
            }
            return (train_state_new, env_state, obs, rng), metrics

        runner_state = (train_state, env_state, obs, rng)
        steps = jnp.arange(total_updates)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, steps)

        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    num_envs = args.num_envs
    num_steps = args.num_steps

    sp_total = args.sp_total_timesteps
    sp_num_updates = sp_total // (num_steps * num_envs)
    sp_batch = num_steps * num_envs * 2

    fcp_total = args.fcp_total_timesteps
    fcp_num_updates = fcp_total // (num_steps * num_envs)
    fcp_batch = num_steps * num_envs

    return {
        "layout_name": args.layout,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_eps": args.clip_eps,
        "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef,
        "num_epochs": args.num_epochs,
        "hidden_dims": tuple([args.hidden_dim] * args.num_layers),
        "pop_size": args.pop_size,
        "num_checkpoints": args.num_checkpoints,
        # SP phase
        "sp_num_updates": sp_num_updates,
        "minibatch_size": min(args.minibatch_size, sp_batch),
        # FCP phase
        "fcp_num_updates": fcp_num_updates,
        "obs_flat_dim": None,  # set at runtime
    }


def main():
    parser = argparse.ArgumentParser(description="FCP training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--sp-total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--fcp-total-timesteps", type=int, default=5_000_000)
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
    parser.add_argument("--pop-size", type=int, default=4)
    parser.add_argument("--num-checkpoints", type=int, default=15)
    args = parser.parse_args()

    config = get_config(args)
    rng = jax.random.PRNGKey(args.seed)

    # Setup env and network
    layout = overcooked_layouts[config["layout_name"]]
    env = make("overcooked", layout=layout)
    agent_names = env.agents

    sample_obs, _ = env.reset(jax.random.PRNGKey(0))
    obs_shape = sample_obs[agent_names[0]].shape
    obs_flat_dim = int(jnp.prod(jnp.array(obs_shape)))
    action_dim = env.action_space(agent_names[0]).n
    config["obs_flat_dim"] = obs_flat_dim

    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    # --- Phase 1: Train SP population with checkpoint saving ---
    print(f"[FCP] Phase 1: Training {args.pop_size} SP agents with checkpoints")
    all_checkpoints = []
    for i in range(args.pop_size):
        rng, sp_rng = jax.random.split(rng)
        t0 = time.time()
        cks = train_sp_with_checkpoints(
            config, network, env, agent_names, sp_rng, args.num_checkpoints
        )
        elapsed = time.time() - t0
        print(f"  SP agent {i}: {len(cks)} checkpoints in {elapsed:.1f}s")
        all_checkpoints.extend(cks)

    print(f"[FCP] Total checkpoints: {len(all_checkpoints)}")

    # --- Phase 2: Categorize checkpoints ---
    high_list, mid_list, low_list = categorize_checkpoints(all_checkpoints)
    print(f"[FCP] Tiers: HIGH={len(high_list)} MED={len(mid_list)} LOW={len(low_list)}")

    high_params = jax.device_put(stack_params(high_list))
    mid_params = jax.device_put(stack_params(mid_list))
    low_params = jax.device_put(stack_params(low_list))

    # --- Phase 3: Train FCP agent ---
    fcp_batch = config["num_steps"] * config["num_envs"]
    config["minibatch_size"] = min(args.minibatch_size, fcp_batch)

    print(f"[FCP] Phase 3: Training FCP agent ({config['fcp_num_updates']} updates)")
    fcp_train_fn = make_fcp_train(config, network, env, agent_names)

    rng, fcp_rng = jax.random.split(rng)
    t0 = time.time()
    result = fcp_train_fn(fcp_rng, high_params, mid_params, low_params)
    jax.block_until_ready(result)
    elapsed = time.time() - t0

    rewards = result["metrics"]["mean_reward"]
    print(f"[FCP] Done in {elapsed:.1f}s. Final mean reward: {rewards[-1]:.3f}")

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
    print(f"[FCP] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
