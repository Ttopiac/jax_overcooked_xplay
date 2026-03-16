"""Evaluation script for Overcooked agents.

Two evaluation modes:
  1. Self-play: Agent plays with a copy of itself. Measures individual competence.
  2. Cross-play: Ego agent plays with an unseen partner. Measures zero-shot coordination.

Usage:
  # Self-play evaluation of SP population
  python evaluate.py --mode self-play --agent-path checkpoints/sp_pop_seed0.pkl --agent-type pop

  # Cross-play evaluation of ego vs unseen partners
  python evaluate.py --mode cross-play --ego-path checkpoints/sp_n2_seed0.pkl \
      --partner-path checkpoints/sp_pop_seed3.pkl --partner-type pop
"""

import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from overcooked_env import Overcooked, overcooked_layouts
from networks import ActorCritic
from obs_encoding import ObsEncoder

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]
BCP_LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring"]
# JaxMARL layout name mapping for BC (original names -> JaxMARL names)
BC_TO_JAXMARL = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymm_advantages",
    "coordination_ring": "coord_ring",
}


def evaluate_episode(env, network_a, params_a, network_b, params_b, encoder, key, num_envs=32):
    """Run one batch of episodes and return mean sparse reward.

    Both agents use deterministic actions (argmax).
    Supports different network architectures for each agent (e.g. ego 3×64 vs BC 2×64).

    Args:
        env: Overcooked env (2 agents)
        network_a: ActorCritic network for agent_0
        params_a: agent_0 params
        network_b: ActorCritic network for agent_1
        params_b: agent_1 params
        encoder: ObsEncoder
        key: random key
        num_envs: parallel episodes

    Returns:
        mean_sparse_reward: jax array scalar
    """
    keys = jax.random.split(key, num_envs)
    obs, state = jax.vmap(env.reset)(keys)

    total_sparse = jnp.zeros(num_envs)

    def _step(carry, unused):
        state, obs, total_sparse, key = carry
        key, step_key = jax.random.split(key)

        obs_0 = encoder.encode(obs, state, 0, num_envs)
        obs_1 = encoder.encode(obs, state, 1, num_envs)

        pi_0, _ = network_a.apply(params_a, obs_0)
        pi_1, _ = network_b.apply(params_b, obs_1)

        # Deterministic: argmax
        a0 = pi_0.logits.argmax(axis=-1)
        a1 = pi_1.logits.argmax(axis=-1)

        actions = {"agent_0": a0, "agent_1": a1}
        step_keys = jax.random.split(step_key, num_envs)
        obs, state, rewards, dones, infos = jax.vmap(env.step)(step_keys, state, actions)

        total_sparse = total_sparse + rewards["agent_0"]
        return (state, obs, total_sparse, key), None

    (_, _, total_sparse, _), _ = jax.lax.scan(
        _step, (state, obs, total_sparse, key), None, length=400  # max_steps
    )

    return total_sparse.mean()


def evaluate_self_play(network, params, encoder, layouts, num_episodes=100, num_envs=32):
    """Evaluate agent in self-play across layouts.

    Args:
        network: ActorCritic
        params: single agent params (plays both sides)
        encoder: ObsEncoder
        layouts: list of layout names
        num_episodes: total episodes per layout
        num_envs: parallel episodes per batch

    Returns:
        dict: {layout_name: mean_sparse_reward}
    """
    results = {}
    num_batches = max(1, num_episodes // num_envs)

    for layout_name in layouts:
        env = Overcooked(layout=overcooked_layouts[layout_name], num_agents=2)

        @jax.jit
        def eval_fn(key, _env=env):
            return evaluate_episode(_env, network, params, network, params, encoder, key, num_envs)

        batch_rewards = []
        for b in range(num_batches):
            key = jax.random.PRNGKey(b * 1000)
            r = float(eval_fn(key))
            batch_rewards.append(r)

        results[layout_name] = float(np.mean(batch_rewards))

    return results


def evaluate_cross_play(network_ego, params_ego, network_partner, params_partner,
                        encoder, layouts, num_episodes=100, num_envs=32):
    """Evaluate ego agent paired with a different partner. Supports different architectures."""
    results = {}
    num_batches = max(1, num_episodes // num_envs)

    for layout_name in layouts:
        env = Overcooked(layout=overcooked_layouts[layout_name], num_agents=2)

        @jax.jit
        def eval_fn(key, _env=env):
            return evaluate_episode(_env, network_ego, params_ego, network_partner, params_partner, encoder, key, num_envs)

        batch_rewards = []
        for b in range(num_batches):
            key = jax.random.PRNGKey(b * 1000 + 42)
            r = float(eval_fn(key))
            batch_rewards.append(r)

        results[layout_name] = float(np.mean(batch_rewards))

    return results


def load_pop_agents(path):
    """Load population agents from checkpoint. Returns list of param dicts."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if "final_params" in data:
        # SP Pop format
        params = data["final_params"]
        n = jax.tree.leaves(params)[0].shape[0]
        return [jax.tree.map(lambda x: x[i], params) for i in range(n)]
    elif "all_params" in data:
        # MEP Pop format
        params = data["all_params"]
        n = jax.tree.leaves(params)[0].shape[0]
        return [jax.tree.map(lambda x: x[i], params) for i in range(n)]
    elif "bc_params" in data:
        # BC format
        agents = []
        for layout in data["bc_params"]:
            for p in data["bc_params"][layout]:
                agents.append(p)
        return agents
    else:
        raise ValueError(f"Unknown checkpoint format: {list(data.keys())}")


def load_ego_agent(path):
    """Load ego agent params from checkpoint."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "params" in data:
        return data["params"]
    elif "ego_params" in data:
        return data["ego_params"]
    else:
        raise ValueError(f"Unknown checkpoint format: {list(data.keys())}")


def categorize_by_skill(agents_results, n_tiers=3):
    """Categorize agents into H/M/L by mean self-play performance.

    Args:
        agents_results: list of (agent_params, {layout: reward}) tuples

    Returns:
        dict: {"HIGH": [...], "MED": [...], "LOW": [...]}
    """
    # Compute mean reward across layouts for each agent
    scored = []
    for params, results in agents_results:
        mean_r = np.mean(list(results.values()))
        scored.append((params, mean_r, results))

    scored.sort(key=lambda x: x[1], reverse=True)
    n = len(scored)
    third = max(1, n // 3)

    return {
        "HIGH": scored[:third],
        "MED": scored[third:2*third],
        "LOW": scored[2*third:],
    }


def print_results_table(title, results_dict):
    """Print results as a formatted table.

    Args:
        results_dict: {row_name: {col_name: value}}
    """
    if not results_dict:
        return

    # Get all columns
    all_cols = set()
    for row in results_dict.values():
        all_cols.update(row.keys())
    cols = sorted(all_cols)

    # Print header
    col_width = 12
    header = f"{'':>20}" + "".join(f"{c:>{col_width}}" for c in cols)
    if "Mean" not in cols:
        header += f"{'Mean':>{col_width}}"
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row_name, row_data in results_dict.items():
        values = [f"{row_data.get(c, 0):.1f}" for c in cols]
        mean_val = np.mean([row_data.get(c, 0) for c in cols])
        line = f"{row_name:>20}" + "".join(f"{v:>{col_width}}" for v in values)
        if "Mean" not in cols:
            line += f"{mean_val:>{col_width}.1f}"
        print(line)
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Overcooked agents")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["self-play-pop", "self-play-ego", "cross-play"],
                        help="Evaluation mode")
    parser.add_argument("--agent-path", type=str, help="Path to agent checkpoint")
    parser.add_argument("--agent-type", type=str, default="pop",
                        choices=["pop", "ego", "bc"],
                        help="Type of agent checkpoint")
    parser.add_argument("--ego-path", type=str, help="Path to ego agent (cross-play)")
    parser.add_argument("--partner-path", type=str, help="Path to partner (cross-play)")
    parser.add_argument("--partner-type", type=str, default="pop",
                        choices=["pop", "ego", "bc"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bc-hidden-dim", type=int, default=64)
    parser.add_argument("--bc-num-layers", type=int, default=2)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--layouts", type=str, default=None,
                        help="Comma-separated layout names (default: all 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    encoder = ObsEncoder()
    rl_network = ActorCritic(action_dim=6, hidden_dims=tuple([args.hidden_dim] * args.num_layers))
    bc_network = ActorCritic(action_dim=6, hidden_dims=tuple([args.bc_hidden_dim] * args.bc_num_layers))
    layouts = args.layouts.split(",") if args.layouts else LAYOUT_NAMES

    if args.mode == "self-play-pop":
        print(f"[EVAL] Self-play evaluation: {args.agent_path}")
        agents = load_pop_agents(args.agent_path)
        network = bc_network if args.agent_type == "bc" else rl_network
        eval_layouts = BCP_LAYOUTS if args.agent_type == "bc" else layouts

        results_table = {}
        for i, params in enumerate(agents):
            params = jax.device_put(params)
            results = evaluate_self_play(network, params, encoder, eval_layouts,
                                         args.num_episodes, args.num_envs)
            label = f"Agent_{i}"
            results_table[label] = results
            mean_r = np.mean(list(results.values()))
            print(f"  {label}: mean={mean_r:.1f}, {results}")

        print_results_table("Self-Play Performance", results_table)

    elif args.mode == "self-play-ego":
        print(f"[EVAL] Self-play evaluation (ego): {args.agent_path}")
        params = jax.device_put(load_ego_agent(args.agent_path))
        results = evaluate_self_play(rl_network, params, encoder, layouts,
                                      args.num_episodes, args.num_envs)
        mean_r = np.mean(list(results.values()))
        print(f"  mean={mean_r:.1f}, {results}")

    elif args.mode == "cross-play":
        print(f"[EVAL] Cross-play: ego={args.ego_path} vs partner={args.partner_path}")
        ego_params = jax.device_put(load_ego_agent(args.ego_path))
        partners = load_pop_agents(args.partner_path)
        partner_network = bc_network if args.partner_type == "bc" else rl_network
        eval_layouts = BCP_LAYOUTS if args.partner_type == "bc" else layouts

        all_results = []
        for i, p_params in enumerate(partners):
            p_params = jax.device_put(p_params)
            results = evaluate_cross_play(rl_network, ego_params, partner_network, p_params,
                                           encoder, eval_layouts, args.num_episodes, args.num_envs)
            mean_r = np.mean(list(results.values()))
            all_results.append((results, mean_r))
            if i < 5 or i == len(partners) - 1:
                print(f"  Partner_{i}: mean={mean_r:.1f}")

        overall_mean = np.mean([r[1] for r in all_results])
        print(f"  Overall mean: {overall_mean:.1f}")

    if args.output:
        # Save results
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"mode": args.mode, "results": results_table if args.mode == "self-play-pop" else results}, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
