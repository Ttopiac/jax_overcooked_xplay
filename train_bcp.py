"""Best Cross-Play (BCP) training in JAX.

Two phases:
1. Train a population of SP agents (each with a different seed).
2. Evaluate all SP agents, select the best one.
3. Train a new agent (best response) against the frozen best SP agent.
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


def make_sp_population_train(config):
    """Train a population of SP agents and return their params + evaluation scores."""
    layout = overcooked_layouts[config["layout_name"]]
    env = make("overcooked", layout=layout)
    agents = env.agents

    sample_key = jax.random.PRNGKey(0)
    sample_obs, _ = env.reset(sample_key)
    obs_shape = sample_obs[agents[0]].shape
    obs_flat_dim = int(jnp.prod(jnp.array(obs_shape)))
    action_dim = env.action_space(agents[0]).n

    network = ActorCritic(action_dim=action_dim, hidden_dims=config["hidden_dims"])

    def train_single_sp(rng):
        """Train one SP agent and return final params + mean reward."""
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        train_state = create_train_state(init_rng, network, (obs_flat_dim,), config["lr"])

        env_rngs = jax.random.split(env_rng, config["num_envs"])
        obs, env_state = jax.vmap(env.reset)(env_rngs)

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

                actions = {agents[0]: a0, agents[1]: a1}
                step_rngs = jax.random.split(s_rng, config["num_envs"])
                next_obs, next_state, rewards, dones, infos = jax.vmap(env.step)(
                    step_rngs, env_state, actions
                )

                done_all = dones["__all__"]
                reward = rewards[agents[0]]

                trans_0 = Transition(obs_0, a0, reward, done_all, lp0, val_0)
                trans_1 = Transition(obs_1, a1, reward, done_all, lp1, val_1)
                return (next_state, next_obs, rng), (trans_0, trans_1)

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

        runner_state = (train_state, env_state, obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=config["sp_num_updates"]
        )

        final_params = runner_state[0].params
        # Use last 10% of rewards as evaluation score
        n_eval = max(1, config["sp_num_updates"] // 10)
        eval_score = metrics["mean_reward"][-n_eval:].mean()
        return final_params, eval_score

    return jax.jit(train_single_sp), network, obs_flat_dim, action_dim, env, agents


def make_bcp_train(config, network, obs_flat_dim, env, agents):
    """Train a best-response agent against a frozen partner."""

    def train(rng, frozen_params):
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        train_state = create_train_state(init_rng, network, (obs_flat_dim,), config["lr"])

        env_rngs = jax.random.split(env_rng, config["num_envs"])
        obs, env_state = jax.vmap(env.reset)(env_rngs)

        def _update_step(runner_state, unused):
            train_state, env_state, obs, rng = runner_state

            def _env_step(carry, unused):
                env_state, obs, rng = carry
                rng, a_rng0, a_rng1, s_rng = jax.random.split(rng, 4)

                obs_0 = flatten_obs(obs[agents[0]], config["num_envs"])
                obs_1 = flatten_obs(obs[agents[1]], config["num_envs"])

                # Ego uses trainable params, partner uses frozen params
                pi_0, val_0 = network.apply(train_state.params, obs_0)
                pi_1, _ = network.apply(frozen_params, obs_1)

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
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=config["bcp_num_updates"]
        )
        return {"train_state": runner_state[0], "metrics": metrics}

    return jax.jit(train)


def get_config(args):
    num_envs = args.num_envs
    num_steps = args.num_steps

    sp_total = args.sp_total_timesteps
    sp_num_updates = sp_total // (num_steps * num_envs)
    sp_batch = num_steps * num_envs * 2
    sp_mb = min(args.minibatch_size, sp_batch)

    bcp_total = args.bcp_total_timesteps
    bcp_num_updates = bcp_total // (num_steps * num_envs)
    bcp_batch = num_steps * num_envs
    bcp_mb = min(args.minibatch_size, bcp_batch)

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
        # SP phase
        "sp_num_updates": sp_num_updates,
        "minibatch_size": sp_mb,
        # BCP phase (overridden later)
        "bcp_num_updates": bcp_num_updates,
    }


def main():
    parser = argparse.ArgumentParser(description="BCP training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--sp-total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--bcp-total-timesteps", type=int, default=5_000_000)
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
    args = parser.parse_args()

    config = get_config(args)
    rng = jax.random.PRNGKey(args.seed)

    # --- Phase 1: Train SP population ---
    print(f"[BCP] Phase 1: Training {args.pop_size} SP agents (seed={args.seed})")
    train_sp_fn, network, obs_flat_dim, action_dim, env, agent_names = (
        make_sp_population_train(config)
    )

    all_sp_params = []
    all_sp_scores = []
    for i in range(args.pop_size):
        rng, sp_rng = jax.random.split(rng)
        t0 = time.time()
        sp_params, sp_score = train_sp_fn(sp_rng)
        jax.block_until_ready(sp_params)
        elapsed = time.time() - t0
        sp_score_val = float(sp_score)
        print(f"  SP agent {i}: score={sp_score_val:.3f} ({elapsed:.1f}s)")
        all_sp_params.append(jax.device_get(sp_params))
        all_sp_scores.append(sp_score_val)

    # --- Phase 2: Select best SP agent ---
    best_idx = int(jnp.argmax(jnp.array(all_sp_scores)))
    frozen_params = jax.device_put(all_sp_params[best_idx])
    print(f"[BCP] Phase 2: Best SP agent idx={best_idx} score={all_sp_scores[best_idx]:.3f}")

    # --- Phase 3: Train best-response agent ---
    print(f"[BCP] Phase 3: Training best-response agent ({config['bcp_num_updates']} updates)")
    bcp_config = {**config}
    bcp_batch = config["num_steps"] * config["num_envs"]
    bcp_config["minibatch_size"] = min(args.minibatch_size, bcp_batch)

    bcp_train_fn = make_bcp_train(bcp_config, network, obs_flat_dim, env, agent_names)

    rng, bcp_rng = jax.random.split(rng)
    t0 = time.time()
    result = bcp_train_fn(bcp_rng, frozen_params)
    jax.block_until_ready(result)
    elapsed = time.time() - t0

    rewards = result["metrics"]["mean_reward"]
    print(f"[BCP] Done in {elapsed:.1f}s. Final mean reward: {rewards[-1]:.3f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "params": jax.device_get(result["train_state"].params),
        "frozen_params": all_sp_params[best_idx],
        "sp_scores": all_sp_scores,
        "config": config,
        "metrics": jax.device_get(result["metrics"]),
    }
    ckpt_path = f"checkpoints/bcp_seed{args.seed}_{args.layout}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[BCP] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
