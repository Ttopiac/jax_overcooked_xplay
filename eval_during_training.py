"""Post-training evaluation at multiple checkpoints.

Instead of only evaluating the final params, evaluate intermediate checkpoints
saved during training. This shows the learning curve without needing to break
the JIT-compiled scan.

Works with any method that saves intermediate checkpoints (ckpt_params).
For methods that only save final params (FCP/BCP/MEP ego), a simple
modification saves params at eval_intervals inside the scan.
"""

import numpy as np
import jax
from evaluate import evaluate_self_play

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]


def eval_checkpoints(ckpt_params, network, encoder, layouts=None,
                     num_episodes=32, num_envs=32, labels=None):
    """Evaluate a sequence of checkpoint params in self-play.

    Args:
        ckpt_params: tree of (num_ckpts, ...) — stacked params at different training points
        network: ActorCritic or CNN network
        encoder: ObsEncoder
        layouts: layout names (default: all 5)
        num_episodes: episodes per layout
        num_envs: parallel envs
        labels: list of strings for each checkpoint (e.g., ["20M", "40M", ...])

    Returns:
        list of dicts with eval results per checkpoint
    """
    if layouts is None:
        layouts = LAYOUT_NAMES

    n_ckpts = jax.tree.leaves(ckpt_params)[0].shape[0]
    if labels is None:
        labels = [f"ckpt_{i}" for i in range(n_ckpts)]

    results = []
    for i in range(n_ckpts):
        params = jax.device_put(jax.tree.map(lambda x: x[i], ckpt_params))
        eval_result = evaluate_self_play(
            network, params, encoder, layouts, num_episodes, num_envs
        )
        mean_r = np.mean(list(eval_result.values()))
        results.append({"label": labels[i], "mean": mean_r, "per_layout": eval_result})
        print(f"    {labels[i]}: mean={mean_r:.1f} {eval_result}")

    return results
