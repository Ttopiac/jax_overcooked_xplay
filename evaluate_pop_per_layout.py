"""Evaluate SP population checkpoints per layout.

Evaluates all SP pop checkpoints (16 agents × 35 ckpts = 560) in self-play
on each of the 5 layouts independently. Saves per-layout scores for use by
FCP's per-layout H/M/L pool construction.

Usage:
    python evaluate_pop_per_layout.py --seeds 0,1,2,3,4
"""

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from overcooked_env import Overcooked, overcooked_layouts
from networks import ActorCritic, CNNActorCritic
from obs_encoding import ObsEncoder
from evaluate import evaluate_episode

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]


def evaluate_pop_per_layout(all_flat_params, network, encoder, num_envs=32, batch_size=16):
    """Evaluate all flat population members in self-play per layout.

    Args:
        all_flat_params: tree of (n_total, ...) — all checkpoint params flattened
        network: ActorCritic
        encoder: ObsEncoder
        num_envs: parallel episodes per eval
        batch_size: how many agents to eval per JIT call

    Returns:
        layout_scores: (n_total, n_layouts) array
    """
    n_total = jax.tree.leaves(all_flat_params)[0].shape[0]
    n_layouts = len(LAYOUT_NAMES)
    n_batches = (n_total + batch_size - 1) // batch_size

    layout_scores = np.zeros((n_total, n_layouts))

    for li, layout_name in enumerate(LAYOUT_NAMES):
        env = Overcooked(layout=overcooked_layouts[layout_name], num_agents=2)

        @jax.jit
        def eval_one(params, key):
            return evaluate_episode(env, network, params, network, params, encoder, key, num_envs)

        eval_batch = jax.jit(jax.vmap(eval_one))

        print(f"  Layout {li+1}/5: {layout_name}")
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, n_total)
            actual_bs = end - start

            batch_params = jax.tree.map(lambda x: x[start:end], all_flat_params)

            # Pad if last batch is smaller
            if actual_bs < batch_size:
                batch_params = jax.tree.map(
                    lambda x: jnp.concatenate([x, jnp.zeros((batch_size - actual_bs, *x.shape[1:]))]),
                    batch_params
                )

            keys = jax.random.split(jax.random.PRNGKey(li * 1000 + bi), batch_size)
            scores = eval_batch(batch_params, keys)
            layout_scores[start:end, li] = np.array(scores[:actual_bs])

        print(f"    mean={layout_scores[:, li].mean():.1f}, "
              f"min={layout_scores[:, li].min():.1f}, max={layout_scores[:, li].max():.1f}")

    return layout_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate SP pop checkpoints per layout")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel episodes per eval")
    parser.add_argument("--batch-size", type=int, default=16, help="Agents per JIT batch")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    for seed in seeds:
        pop_path = f"checkpoints/sp_pop_seed{seed}.pkl"
        out_path = f"checkpoints/sp_pop_seed{seed}_layout_scores.pkl"

        print(f"[EVAL-PER-LAYOUT] seed {seed}: loading {pop_path}")
        with open(pop_path, "rb") as f:
            pop_data = pickle.load(f)

        ckpt_params = pop_data["ckpt_params"]
        config = pop_data["config"]
        obs_enc = config.get("obs_encoding", "egocentric")
        encoder = ObsEncoder(mode=obs_enc)
        if obs_enc == "full":
            network = CNNActorCritic(action_dim=6)
        else:
            hidden_dims = config["hidden_dims"] if "hidden_dims" in config else (256, 256)
            network = ActorCritic(action_dim=6, hidden_dims=hidden_dims)

        # Flatten: (pop_size, num_ckpts, ...) -> (pop_size * num_ckpts, ...)
        flat_params = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), ckpt_params)
        n_total = jax.tree.leaves(flat_params)[0].shape[0]
        pop_size = jax.tree.leaves(ckpt_params)[0].shape[0]
        num_ckpts = jax.tree.leaves(ckpt_params)[0].shape[1]
        print(f"  {pop_size} agents × {num_ckpts} ckpts = {n_total} total")

        t0 = time.time()
        layout_scores = evaluate_pop_per_layout(
            jax.device_put(flat_params), network, encoder,
            num_envs=args.num_envs, batch_size=args.batch_size
        )
        elapsed = time.time() - t0

        print(f"  layout_scores shape: {layout_scores.shape}")
        print(f"  Done in {elapsed:.1f}s")

        # Save
        os.makedirs("checkpoints", exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump({"layout_scores": layout_scores, "seed": seed}, f)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
