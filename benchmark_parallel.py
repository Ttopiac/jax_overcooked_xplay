"""Benchmark: how many SP agents can train in parallel efficiently?

Tests 4, 8, 16 agents with all 5 layouts simultaneously.
Each agent trains across all 5 Overcooked layouts with padded observations.
"""

import time
import jax
import jax.numpy as jnp
from overcooked_env import Overcooked, overcooked_layouts

from networks import ActorCritic
from ppo import Transition, compute_gae, ppo_update, create_train_state

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]
NUM_UPDATES = 100  # fixed for benchmark
NUM_STEPS = 128
NUM_ENVS_PER_LAYOUT = 8  # 8 envs per layout × 5 layouts = 40 envs per agent


def make_benchmark(pop_size):
    """Build a JIT-compiled parallel training function for pop_size agents."""
    # Create all 5 envs
    envs = [Overcooked(layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents

    # Determine max obs dim
    obs_flat_dims = []
    for env in envs:
        obs, _ = env.reset(jax.random.PRNGKey(0))
        obs_flat_dims.append(int(jnp.prod(jnp.array(obs[agents[0]].shape))))
    max_obs_dim = max(obs_flat_dims)
    action_dim = envs[0].action_space(agents[0]).n

    network = ActorCritic(action_dim=action_dim, hidden_dims=(256, 256))
    n_envs = NUM_ENVS_PER_LAYOUT

    config = {
        "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
        "vf_coef": 0.3, "ent_coef": 0.01, "num_epochs": 4,
        "num_steps": NUM_STEPS,
        "minibatch_size": min(512, NUM_STEPS * n_envs * len(LAYOUT_NAMES) * 2),
    }

    def flatten_pad(obs, target_dim):
        flat = obs.reshape(n_envs, -1).astype(jnp.float32) / 255.0
        pad_width = target_dim - flat.shape[1]
        return jnp.pad(flat, ((0, 0), (0, pad_width)))

    def train_single_agent(rng):
        rng, init_rng = jax.random.split(rng)
        train_state = create_train_state(init_rng, network, (max_obs_dim,), config["lr"])

        # Init envs for all 5 layouts
        env_init_data = []
        for i, env in enumerate(envs):
            rng, env_rng = jax.random.split(rng)
            env_rngs = jax.random.split(env_rng, n_envs)
            obs, state = jax.vmap(env.reset)(env_rngs)
            env_init_data.append((obs, state))

        all_obs = tuple(d[0] for d in env_init_data)
        all_states = tuple(d[1] for d in env_init_data)

        def _update_step(runner_state, unused):
            train_state, all_states, all_obs, rng = runner_state

            # Collect transitions from ALL layouts
            all_trans_0 = []
            all_trans_1 = []
            new_states = []
            new_obs_list = []

            for i, env in enumerate(envs):
                rng, a0_rng, a1_rng, s_rng = jax.random.split(rng, 4)

                def _env_step_layout(carry, unused, _env=env, _obs_dim=obs_flat_dims[i]):
                    env_state, obs, rng = carry
                    rng, ar0, ar1, sr = jax.random.split(rng, 4)

                    obs_0 = flatten_pad(obs[agents[0]], max_obs_dim)
                    obs_1 = flatten_pad(obs[agents[1]], max_obs_dim)

                    pi_0, val_0 = network.apply(train_state.params, obs_0)
                    pi_1, val_1 = network.apply(train_state.params, obs_1)

                    a0 = pi_0.sample(seed=ar0)
                    a1 = pi_1.sample(seed=ar1)
                    lp0 = pi_0.log_prob(a0)
                    lp1 = pi_1.log_prob(a1)

                    actions = {agents[0]: a0, agents[1]: a1}
                    step_rngs = jax.random.split(sr, n_envs)
                    next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                        step_rngs, env_state, actions
                    )

                    reward = rewards[agents[0]]
                    done_all = dones["__all__"]
                    t0 = Transition(obs_0, a0, reward, done_all, lp0, val_0)
                    t1 = Transition(obs_1, a1, reward, done_all, lp1, val_1)
                    return (next_state, next_obs, rng), (t0, t1)

                carry, (traj_0, traj_1) = jax.lax.scan(
                    _env_step_layout,
                    (all_states[i], all_obs[i], rng),
                    None, length=NUM_STEPS,
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                rng = carry[2]
                all_trans_0.append(traj_0)
                all_trans_1.append(traj_1)

            all_states = tuple(new_states)
            all_obs = tuple(new_obs_list)

            # Concatenate transitions across layouts: (T, N_per_layout, ...) -> (T, N_total, ...)
            combined_0 = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_trans_0)
            combined_1 = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_trans_1)
            traj_batch = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=1), combined_0, combined_1
            )

            # Bootstrap values
            all_last_vals = []
            for i, env in enumerate(envs):
                o0 = flatten_pad(all_obs[i][agents[0]], max_obs_dim)
                o1 = flatten_pad(all_obs[i][agents[1]], max_obs_dim)
                _, lv0 = network.apply(train_state.params, o0)
                _, lv1 = network.apply(train_state.params, o1)
                all_last_vals.extend([lv0, lv1])
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

            return (train_state, all_states, all_obs, rng), {
                "mean_reward": jnp.mean(traj_batch.reward)
            }

        runner_state = (train_state, all_states, all_obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=NUM_UPDATES
        )
        return metrics["mean_reward"][-1]

    return jax.jit(jax.vmap(train_single_agent))


def main():
    import subprocess

    print(f"Benchmark: {NUM_UPDATES} updates, {NUM_STEPS} steps, "
          f"{NUM_ENVS_PER_LAYOUT} envs/layout, 5 layouts")
    print(f"Batch size per agent per update: {NUM_STEPS * NUM_ENVS_PER_LAYOUT * 5 * 2}")
    print()

    for pop_size in [4, 8, 16]:
        print(f"--- pop_size={pop_size} ---")

        # Clear GPU cache
        jax.clear_caches()

        train_fn = make_benchmark(pop_size)
        rngs = jax.random.split(jax.random.PRNGKey(0), pop_size)

        # Warmup (JIT compile)
        t_compile = time.time()
        result = train_fn(rngs)
        jax.block_until_ready(result)
        compile_time = time.time() - t_compile

        # Actual run (already compiled)
        t_run = time.time()
        result = train_fn(rngs)
        jax.block_until_ready(result)
        run_time = time.time() - t_run

        # VRAM
        vram = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
            capture_output=True, text=True
        ).stdout.strip()

        timesteps_per_update = NUM_STEPS * NUM_ENVS_PER_LAYOUT * 5  # per agent
        total_timesteps = NUM_UPDATES * timesteps_per_update * pop_size
        throughput = total_timesteps / run_time

        print(f"  Compile time:  {compile_time:.1f}s")
        print(f"  Run time:      {run_time:.1f}s")
        print(f"  VRAM used:     {vram}")
        print(f"  Throughput:    {throughput/1e6:.1f}M timesteps/s")
        print(f"  Time for 60M per agent: {60e6 / (timesteps_per_update * NUM_UPDATES / run_time) / 60:.1f} min")
        print()


if __name__ == "__main__":
    main()
