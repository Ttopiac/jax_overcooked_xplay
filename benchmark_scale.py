"""Benchmark: SP Pop training with 3×64 MLP at different agent counts.
Tests 16, 32, 48, 64, 80 agents with all 5 layouts, egocentric encoding.
"""

import time
import subprocess
import jax
import jax.numpy as jnp
from overcooked_env import Overcooked, overcooked_layouts
from networks import ActorCritic
from obs_encoding import ObsEncoder
from ppo import Transition, compute_gae, ppo_update, create_train_state
from reward_shaping import RewardShaper

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]
NUM_UPDATES = 50
NUM_STEPS = 128
NUM_ENVS_PER_LAYOUT = 8
HIDDEN_DIMS = (64, 64, 64)  # 3×64


def make_benchmark(pop_size):
    envs = [Overcooked(layout=overcooked_layouts[name]) for name in LAYOUT_NAMES]
    agents = envs[0].agents
    action_dim = envs[0].action_space(agents[0]).n
    n_env_l = NUM_ENVS_PER_LAYOUT
    n_env_total = n_env_l * len(LAYOUT_NAMES)

    encoder = ObsEncoder(mode="egocentric")
    reward_shaper = RewardShaper(mode="shaped")
    network = ActorCritic(action_dim=action_dim, hidden_dims=HIDDEN_DIMS)

    config = {
        "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
        "vf_coef": 0.3, "ent_coef": 0.01, "num_epochs": 4,
        "num_steps": NUM_STEPS,
        "minibatch_size": min(512, NUM_STEPS * n_env_total * 2),
    }

    def train_single_agent(rng):
        rng, init_rng = jax.random.split(rng)
        train_state = create_train_state(init_rng, network, (encoder.obs_flat_dim,), config["lr"])

        all_env_states, all_obs = [], []
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

            all_traj_0, all_traj_1 = [], []
            new_states, new_obs_list = [], []

            for i, env in enumerate(envs):
                def _env_step(carry, inner_step, _env=env):
                    env_state, obs, rng = carry
                    rng, a0r, a1r, sr = jax.random.split(rng, 4)
                    obs_0 = encoder.encode(obs, env_state, 0, n_env_l)
                    obs_1 = encoder.encode(obs, env_state, 1, n_env_l)
                    pi_0, val_0 = network.apply(train_state.params, obs_0)
                    pi_1, val_1 = network.apply(train_state.params, obs_1)
                    a0 = pi_0.sample(seed=a0r)
                    a1 = pi_1.sample(seed=a1r)
                    lp0, lp1 = pi_0.log_prob(a0), pi_1.log_prob(a1)
                    actions = {agents[0]: a0, agents[1]: a1}
                    step_rngs = jax.random.split(sr, n_env_l)
                    next_obs, next_state, rewards, dones, infos = jax.vmap(_env.step)(
                        step_rngs, env_state, actions
                    )
                    step_count = (update_idx * NUM_STEPS + inner_step) * n_env_total
                    r0 = reward_shaper.compute(rewards, infos, agents[0], step_count)
                    r1 = reward_shaper.compute(rewards, infos, agents[1], step_count)
                    done_all = dones["__all__"]
                    t0 = Transition(obs_0, a0, r0, done_all, lp0, val_0)
                    t1 = Transition(obs_1, a1, r1, done_all, lp1, val_1)
                    return (next_state, next_obs, rng), (t0, t1)

                rng, scan_rng = jax.random.split(rng)
                carry, (traj_0, traj_1) = jax.lax.scan(
                    _env_step, (all_env_states[i], all_obs[i], scan_rng),
                    jnp.arange(NUM_STEPS),
                )
                new_states.append(carry[0])
                new_obs_list.append(carry[1])
                all_traj_0.append(traj_0)
                all_traj_1.append(traj_1)

            all_env_states = tuple(new_states)
            all_obs = tuple(new_obs_list)
            combined_0 = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_traj_0)
            combined_1 = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_traj_1)
            traj_batch = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), combined_0, combined_1)

            all_last_vals = []
            for i in range(len(envs)):
                o0 = encoder.encode(all_obs[i], all_env_states[i], 0, n_env_l)
                o1 = encoder.encode(all_obs[i], all_env_states[i], 1, n_env_l)
                _, lv0 = network.apply(train_state.params, o0)
                _, lv1 = network.apply(train_state.params, o1)
                all_last_vals.extend([lv0, lv1])
            last_val = jnp.concatenate(all_last_vals)

            advantages, returns = compute_gae(traj_batch, last_val, config["gamma"], config["gae_lambda"])
            flat_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), traj_batch)
            flat_adv, flat_ret = advantages.reshape(-1), returns.reshape(-1)
            rng, update_rng = jax.random.split(rng)
            train_state, loss_info = ppo_update(train_state, flat_batch, flat_adv, flat_ret, config, update_rng)
            return (train_state, all_env_states, all_obs, rng), jnp.mean(traj_batch.reward)

        runner_state = (train_state, all_env_states, all_obs, rng)
        runner_state, rewards = jax.lax.scan(_update_step, runner_state, jnp.arange(NUM_UPDATES))
        return rewards[-1]

    return jax.jit(jax.vmap(train_single_agent))


def main():
    print(f"Benchmark: {NUM_UPDATES} updates, {NUM_STEPS} steps, "
          f"{NUM_ENVS_PER_LAYOUT} envs/layout, 5 layouts, 3×64 MLP")
    print()

    for pop_size in [16, 32, 48, 64, 80]:
        jax.clear_caches()
        train_fn = make_benchmark(pop_size)
        rngs = jax.random.split(jax.random.PRNGKey(0), pop_size)

        t_compile = time.time()
        result = train_fn(rngs)
        jax.block_until_ready(result)
        compile_time = time.time() - t_compile

        t_run = time.time()
        result = train_fn(rngs)
        jax.block_until_ready(result)
        run_time = time.time() - t_run

        vram = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
            capture_output=True, text=True
        ).stdout.strip()

        steps_per_update = NUM_STEPS * NUM_ENVS_PER_LAYOUT * 5
        total_steps = NUM_UPDATES * steps_per_update * pop_size
        throughput = total_steps / run_time
        time_60m = 60e6 * pop_size / throughput / 60  # minutes for all agents to do 60M each

        print(f"--- {pop_size} agents ---")
        print(f"  Compile: {compile_time:.1f}s")
        print(f"  Run:     {run_time:.1f}s")
        print(f"  VRAM:    {vram}")
        print(f"  Throughput: {throughput/1e6:.1f}M steps/s")
        print(f"  Time for 60M/agent (all {pop_size}): {time_60m:.0f} min")
        print()


if __name__ == "__main__":
    main()
