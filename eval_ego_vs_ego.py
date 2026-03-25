"""Ego vs Ego cross-play evaluation.

Evaluates all Run 3 ego methods against each other and against Run 4 pixel FCP.
Fills the "vs Egos" column in Table D and expands Table C to a full matrix.
"""

import os
import json
import pickle
import time
import numpy as np

os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import jax
import jax.numpy as jnp
from overcooked_env import Overcooked, overcooked_layouts
from networks import ActorCritic, IMPALACNNActorCritic
from obs_encoding import ObsEncoder
from evaluate import evaluate_episode

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]
BCP_LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring"]
NUM_EVAL_ENVS = 32
NUM_BATCHES = 3


def eval_one_layout(env, net_a, params_a, enc_a, net_b, params_b, enc_b):
    @jax.jit
    def _eval(key):
        return evaluate_episode(env, net_a, params_a, net_b, params_b, enc_a, key, NUM_EVAL_ENVS, encoder_b=enc_b)
    rewards = []
    for b in range(NUM_BATCHES):
        rewards.append(float(_eval(jax.random.PRNGKey(b * 1000 + 42))))
    return float(np.mean(rewards))


def load_ego_params(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "params" in data:
        return data["params"]
    elif "ego_params" in data:
        return data["ego_params"]
    raise ValueError(f"Unknown format: {list(data.keys())}")


def main():
    t_start = time.time()

    # Setup
    ego_encoder = ObsEncoder(mode="egocentric")
    pixel_encoder = ObsEncoder(mode="pixel")

    with open("checkpoints/sp_pop_seed0.pkl", "rb") as f:
        sp_data = pickle.load(f)
    mlp_dims = sp_data["config"].get("hidden_dims", (256, 256))
    mlp_network = ActorCritic(action_dim=6, hidden_dims=mlp_dims)
    pixel_network = IMPALACNNActorCritic(action_dim=6, palette=pixel_encoder.atlas.palette)

    envs = {name: Overcooked(layout=overcooked_layouts[name], num_agents=2) for name in LAYOUT_NAMES}

    # Load all ego agents
    agents = {}  # {(method, seed): (params, network, encoder)}

    r3_methods = {
        "SP": ("checkpoints/sp_n2_seed{}.pkl", [0, 1, 2]),
        "MEP-16": ("checkpoints/mep16_seed{}.pkl", [0, 1, 2]),
        "MEP-8": ("checkpoints/mep8_seed{}.pkl", [0, 1, 2]),
        "BCP": ("checkpoints/run3_backup/bcp_n2x1_hetero_seed{}.pkl", [0, 1, 2]),
        "FCP": ("checkpoints/run3_backup/fcp_n2x1_hetero_seed{}.pkl", [0, 1, 2]),
    }

    for method, (path_tmpl, seeds) in r3_methods.items():
        for s in seeds:
            path = path_tmpl.format(s)
            if os.path.exists(path):
                agents[(method, s)] = (jax.device_put(load_ego_params(path)), mlp_network, ego_encoder)

    for s in range(6):
        path = f"checkpoints/fcp_n2x1_hetero_seed{s}.pkl"
        if os.path.exists(path):
            agents[("R4_pixel", s)] = (jax.device_put(load_ego_params(path)), pixel_network, pixel_encoder)

    all_methods = ["SP", "MEP-16", "MEP-8", "BCP", "FCP", "R4_pixel"]
    method_seeds = {
        "SP": [0, 1, 2], "MEP-16": [0, 1, 2], "MEP-8": [0, 1, 2],
        "BCP": [0, 1, 2], "FCP": [0, 1, 2], "R4_pixel": [0, 1, 2, 3, 4, 5],
    }

    # results[ego_method][partner_method][layout] = score (averaged across seeds)
    results = {}
    n_evals = 0

    for ego_m in all_methods:
        results[ego_m] = {}
        for partner_m in all_methods:
            if ego_m == partner_m:
                continue  # skip self-play (already in Table A)

            # Determine layouts
            if ego_m == "BCP" or partner_m == "BCP":
                layouts = BCP_LAYOUTS
            else:
                layouts = LAYOUT_NAMES

            layout_scores = {l: [] for l in layouts}

            for es in method_seeds.get(ego_m, []):
                if (ego_m, es) not in agents:
                    continue
                ego_params, ego_net, ego_enc = agents[(ego_m, es)]

                for ps in method_seeds.get(partner_m, []):
                    if (partner_m, ps) not in agents:
                        continue
                    p_params, p_net, p_enc = agents[(partner_m, ps)]

                    for layout in layouts:
                        score = eval_one_layout(envs[layout], ego_net, ego_params, ego_enc,
                                                 p_net, p_params, p_enc)
                        layout_scores[layout].append(score)
                        n_evals += 1

            # Average across all seed pairs
            avg_per_layout = {}
            for l in layouts:
                if layout_scores[l]:
                    avg_per_layout[l] = float(np.mean(layout_scores[l]))
            results[ego_m][partner_m] = avg_per_layout

            mean_r = np.mean(list(avg_per_layout.values())) if avg_per_layout else 0
            print(f"  {ego_m:>10s} vs {partner_m:<10s}: mean={mean_r:.1f}")

    elapsed = time.time() - t_start
    print(f"\nCompleted {n_evals} evals in {elapsed/60:.0f} min")

    # === Print full ego-vs-ego matrix (Table C expanded) ===
    print(f"\n{'='*60}")
    print("TABLE C (EXPANDED): EGO vs EGO CROSS-PLAY")
    print(f"{'='*60}")

    header = f"{'Ego/Partner':>12s}" + "".join(f" {m:>10s}" for m in all_methods)
    print(header)

    for ego_m in all_methods:
        row = f"{ego_m:>12s}"
        for partner_m in all_methods:
            if ego_m == partner_m:
                row += f" {'(self)':>10s}"
            elif partner_m in results.get(ego_m, {}):
                vals = list(results[ego_m][partner_m].values())
                mean = np.mean(vals) if vals else 0
                row += f" {mean:>10.1f}"
            else:
                row += f" {'—':>10s}"
        print(row)

    # === Compute "vs Egos" for each method ===
    print(f"\n{'='*60}")
    print("VS EGOS SUMMARY (for Table D)")
    print(f"{'='*60}")

    for ego_m in all_methods:
        all_scores = []
        for partner_m in all_methods:
            if ego_m == partner_m:
                continue
            if partner_m in results.get(ego_m, {}):
                vals = list(results[ego_m][partner_m].values())
                all_scores.extend(vals)
        ego_mean = np.mean(all_scores) if all_scores else 0
        print(f"  {ego_m:>12s} vs Egos = {ego_mean:.1f}")

    # === Per-layout breakdown ===
    print(f"\n{'='*60}")
    print("PER-LAYOUT EGO vs EGO")
    print(f"{'='*60}")

    for layout in LAYOUT_NAMES:
        print(f"\n  {layout}:")
        header = f"{'':>12s}" + "".join(f" {m:>10s}" for m in all_methods)
        print(header)
        for ego_m in all_methods:
            row = f"  {ego_m:>10s}"
            for partner_m in all_methods:
                if ego_m == partner_m:
                    row += f" {'(self)':>10s}"
                elif partner_m in results.get(ego_m, {}):
                    s = results[ego_m][partner_m].get(layout)
                    row += f" {s:>10.1f}" if s is not None else f" {'—':>10s}"
                else:
                    row += f" {'—':>10s}"
            print(row)

    # Save raw results
    def to_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): to_json(v) for k, v in obj.items()}
        return obj

    with open("reports/eval_results/ego_vs_ego_raw.json", "w") as f:
        json.dump(to_json(results), f, indent=2)
    print(f"\nRaw results saved to reports/eval_results/ego_vs_ego_raw.json")


if __name__ == "__main__":
    main()
