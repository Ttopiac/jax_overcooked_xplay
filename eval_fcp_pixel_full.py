"""Full evaluation of FCP pixel ego agents against all Run 3 partners.

Evaluates:
1. FCP pixel ego (seeds 0-5) self-play
2. FCP pixel ego vs SP Pop (H/M/L tiers per layout)
3. FCP pixel ego vs MEP-16 Pop (H/M/L tiers)
4. FCP pixel ego vs MEP-8 Pop (H/M/L tiers)
5. FCP pixel ego vs BC
6. FCP pixel ego vs Run 3 ego agents (FCP, MEP-16, MEP-8, SP)

All cross-play uses dual encoder: pixel CNN for ego, egocentric MLP for partners.
"""

import os
import json
import pickle
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
NUM_EVAL_EPISODES = 96  # 3 batches of 32


def eval_ego_vs_partner(ego_network, ego_params, ego_encoder,
                         partner_network, partner_params, partner_encoder,
                         layouts, num_episodes=NUM_EVAL_EPISODES, num_envs=NUM_EVAL_ENVS):
    """Evaluate ego vs one partner across layouts."""
    results = {}
    num_batches = max(1, num_episodes // num_envs)
    for layout in layouts:
        env = Overcooked(layout=overcooked_layouts[layout], num_agents=2)

        @jax.jit
        def eval_fn(key, _env=env):
            return evaluate_episode(
                _env, ego_network, ego_params, partner_network, partner_params,
                ego_encoder, key, num_envs, encoder_b=partner_encoder
            )

        batch_rewards = []
        for b in range(num_batches):
            key = jax.random.PRNGKey(b * 1000 + 42)
            r = float(eval_fn(key))
            batch_rewards.append(r)
        results[layout] = float(np.mean(batch_rewards))
    return results


def load_and_setup():
    """Load all networks, encoders, and checkpoints."""
    # Pixel ego setup
    ego_encoder = ObsEncoder(mode="pixel")
    ego_network = IMPALACNNActorCritic(action_dim=6, palette=ego_encoder.atlas.palette)

    # Egocentric partner setup
    partner_encoder = ObsEncoder(mode="egocentric")

    # Load SP Pop config for partner hidden dims
    with open("checkpoints/sp_pop_seed0.pkl", "rb") as f:
        sp_data = pickle.load(f)
    partner_dims = sp_data["config"].get("hidden_dims", (256, 256))
    partner_network = ActorCritic(action_dim=6, hidden_dims=partner_dims)

    # BC network
    with open("checkpoints/bc_seed0.pkl", "rb") as f:
        bc_data = pickle.load(f)
    bc_dims = bc_data.get("bc_hidden_dims", (64, 64))
    bc_network = ActorCritic(action_dim=6, hidden_dims=bc_dims)

    return ego_encoder, ego_network, partner_encoder, partner_network, bc_network


def load_ego_params(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "params" in data:
        return jax.device_put(data["params"])
    elif "ego_params" in data:
        return jax.device_put(data["ego_params"])
    else:
        raise ValueError(f"Unknown format: {list(data.keys())}")


def load_pop_with_tiers(pop_path, scores_path=None):
    """Load population and split into H/M/L tiers using layout scores.

    If scores_path doesn't exist, tries to find it automatically.
    Falls back to all-same tiers if no scores available.
    """
    with open(pop_path, "rb") as f:
        pop_data = pickle.load(f)

    # Get flat params
    if "ckpt_params" in pop_data:
        flat_params = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), pop_data["ckpt_params"])
    elif "all_params" in pop_data:
        flat_params = pop_data["all_params"]
    else:
        raise ValueError(f"Unknown pop format: {list(pop_data.keys())}")

    n_total = jax.tree.leaves(flat_params)[0].shape[0]

    # Try to load layout scores
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, "rb") as f:
            scores_data = pickle.load(f)
        layout_scores = scores_data["layout_scores"]  # (n_total, 5)

        # Handle size mismatch (scores may cover different number of agents)
        if layout_scores.shape[0] != n_total:
            print(f"    WARNING: scores ({layout_scores.shape[0]}) != params ({n_total}), using min")
            n_use = min(layout_scores.shape[0], n_total)
            layout_scores = layout_scores[:n_use]
            flat_params = jax.tree.map(lambda x: x[:n_use], flat_params)
            n_total = n_use

        overall_scores = layout_scores.mean(axis=1)
        sorted_idx = np.argsort(-overall_scores)
        third = max(1, n_total // 3)

        high_idx = sorted_idx[:third]
        mid_idx = sorted_idx[third:2*third]
        low_idx = sorted_idx[2*third:]

        return {
            "high": (jax.tree.map(lambda x: x[high_idx], flat_params), float(overall_scores[high_idx].mean())),
            "mid": (jax.tree.map(lambda x: x[mid_idx], flat_params), float(overall_scores[mid_idx].mean())),
            "low": (jax.tree.map(lambda x: x[low_idx], flat_params), float(overall_scores[low_idx].mean())),
        }
    else:
        # No scores — return all params as single tier
        print(f"    No layout_scores found, returning all {n_total} agents as one tier")
        return {
            "high": (flat_params, 0),
            "mid": (flat_params, 0),
            "low": (flat_params, 0),
        }


def eval_ego_vs_pop_tiered(ego_network, ego_params, ego_encoder,
                            partner_network, partner_encoder,
                            pop_tiers, layouts, label=""):
    """Evaluate ego against H/M/L population tiers."""
    results = {}
    for tier_name in ["high", "mid", "low"]:
        tier_params, tier_score = pop_tiers[tier_name]
        n_agents = jax.tree.leaves(tier_params)[0].shape[0]
        # Sample up to 8 agents from each tier for efficiency
        n_eval = min(8, n_agents)
        tier_results = []
        for i in range(n_eval):
            p_params = jax.device_put(jax.tree.map(lambda x: x[i], tier_params))
            r = eval_ego_vs_partner(
                ego_network, ego_params, ego_encoder,
                partner_network, p_params, partner_encoder,
                layouts
            )
            tier_results.append(r)

        # Average across partners
        avg = {}
        for layout in layouts:
            avg[layout] = float(np.mean([r[layout] for r in tier_results]))
        mean_r = float(np.mean(list(avg.values())))
        results[tier_name] = {"per_layout": avg, "mean": mean_r}
        print(f"    {label} {tier_name.upper():>4} (pop_score={tier_score:.0f}): "
              f"mean={mean_r:.1f} {avg}")

    return results


def main():
    ego_encoder, ego_network, partner_encoder, partner_network, bc_network = load_and_setup()

    os.makedirs("reports/eval_results", exist_ok=True)
    all_results = {}

    # Load FCP pixel ego seeds
    pixel_ego_seeds = []
    for seed in range(6):
        path = f"checkpoints/fcp_n2x1_hetero_seed{seed}.pkl"
        if os.path.exists(path):
            pixel_ego_seeds.append(seed)

    print(f"\nFCP pixel ego seeds: {pixel_ego_seeds}")

    # ============================================================
    # 1. Self-play evaluation
    # ============================================================
    print("\n" + "="*60)
    print("1. FCP PIXEL EGO SELF-PLAY")
    print("="*60)
    selfplay_results = {}
    for seed in pixel_ego_seeds:
        params = load_ego_params(f"checkpoints/fcp_n2x1_hetero_seed{seed}.pkl")
        r = eval_ego_vs_partner(
            ego_network, params, ego_encoder,
            ego_network, params, ego_encoder,
            LAYOUT_NAMES
        )
        mean_r = float(np.mean(list(r.values())))
        selfplay_results[f"seed{seed}"] = r
        print(f"  seed {seed}: mean={mean_r:.1f} {r}")
    all_results["selfplay"] = selfplay_results

    # ============================================================
    # 2. Cross-play vs SP Pop (H/M/L tiers)
    # ============================================================
    print("\n" + "="*60)
    print("2. FCP PIXEL EGO vs SP POP (H/M/L)")
    print("="*60)
    sp_cross = {}
    for p_seed in [3, 4]:
        pop_path = f"checkpoints/sp_pop_seed{p_seed}.pkl"
        scores_path = f"checkpoints/sp_pop_seed{p_seed}_layout_scores.pkl"
        if not os.path.exists(scores_path):
            print(f"  Skipping SP Pop seed {p_seed} (no layout scores)")
            continue
        pop_tiers = load_pop_with_tiers(pop_path, scores_path)
        for ego_seed in pixel_ego_seeds:
            print(f"  Ego seed {ego_seed} vs SP Pop seed {p_seed}:")
            ego_params = load_ego_params(f"checkpoints/fcp_n2x1_hetero_seed{ego_seed}.pkl")
            tier_results = eval_ego_vs_pop_tiered(
                ego_network, ego_params, ego_encoder,
                partner_network, partner_encoder,
                pop_tiers, LAYOUT_NAMES, f"sp_s{p_seed}"
            )
            sp_cross[f"ego{ego_seed}_vs_sp{p_seed}"] = tier_results
    all_results["vs_sp_pop"] = sp_cross

    # ============================================================
    # 3. Cross-play vs MEP-16 Pop (H/M/L tiers)
    # ============================================================
    print("\n" + "="*60)
    print("3. FCP PIXEL EGO vs MEP-16 POP (H/M/L)")
    print("="*60)
    mep16_cross = {}
    for p_seed in [3, 4]:
        pop_path = f"checkpoints/mep16_pop_seed{p_seed}.pkl"
        scores_path = f"checkpoints/mep16_pop_seed{p_seed}_layout_scores.pkl"
        if not os.path.exists(pop_path):
            continue
        pop_tiers = load_pop_with_tiers(pop_path, scores_path)

        for ego_seed in pixel_ego_seeds:
            print(f"  Ego seed {ego_seed} vs MEP-16 Pop seed {p_seed}:")
            ego_params = load_ego_params(f"checkpoints/fcp_n2x1_hetero_seed{ego_seed}.pkl")
            tier_results = eval_ego_vs_pop_tiered(
                ego_network, ego_params, ego_encoder,
                partner_network, partner_encoder,
                pop_tiers, LAYOUT_NAMES, f"mep16_s{p_seed}"
            )
            mep16_cross[f"ego{ego_seed}_vs_mep16_{p_seed}"] = tier_results
    all_results["vs_mep16_pop"] = mep16_cross

    # ============================================================
    # 4. Cross-play vs MEP-8 Pop (H/M/L tiers)
    # ============================================================
    print("\n" + "="*60)
    print("4. FCP PIXEL EGO vs MEP-8 POP (H/M/L)")
    print("="*60)
    mep8_cross = {}
    for p_seed in [3, 4]:
        pop_path = f"checkpoints/mep8_pop_seed{p_seed}.pkl"
        scores_path = f"checkpoints/mep8_pop_seed{p_seed}_layout_scores.pkl"
        if not os.path.exists(pop_path):
            continue
        pop_tiers = load_pop_with_tiers(pop_path, scores_path)

        for ego_seed in pixel_ego_seeds:
            print(f"  Ego seed {ego_seed} vs MEP-8 Pop seed {p_seed}:")
            ego_params = load_ego_params(f"checkpoints/fcp_n2x1_hetero_seed{ego_seed}.pkl")
            tier_results = eval_ego_vs_pop_tiered(
                ego_network, ego_params, ego_encoder,
                partner_network, partner_encoder,
                pop_tiers, LAYOUT_NAMES, f"mep8_s{p_seed}"
            )
            mep8_cross[f"ego{ego_seed}_vs_mep8_{p_seed}"] = tier_results
    all_results["vs_mep8_pop"] = mep8_cross

    # ============================================================
    # 5. Cross-play vs BC
    # ============================================================
    print("\n" + "="*60)
    print("5. FCP PIXEL EGO vs BC")
    print("="*60)
    bc_cross = {}
    for p_seed in [3, 4]:
        bc_path = f"checkpoints/bc_seed{p_seed}.pkl"
        if not os.path.exists(bc_path):
            continue
        with open(bc_path, "rb") as f:
            bc_data = pickle.load(f)
        bc_list = []
        for layout in bc_data["bc_params"]:
            bc_list.extend(bc_data["bc_params"][layout])

        for ego_seed in pixel_ego_seeds:
            ego_params = load_ego_params(f"checkpoints/fcp_n2x1_hetero_seed{ego_seed}.pkl")
            bc_results = []
            for i, bp in enumerate(bc_list[:6]):  # Sample 6 BC agents
                bp = jax.device_put(bp)
                r = eval_ego_vs_partner(
                    ego_network, ego_params, ego_encoder,
                    bc_network, bp, partner_encoder,
                    BCP_LAYOUTS
                )
                bc_results.append(r)
            avg = {}
            for layout in BCP_LAYOUTS:
                avg[layout] = float(np.mean([r[layout] for r in bc_results]))
            mean_r = float(np.mean(list(avg.values())))
            bc_cross[f"ego{ego_seed}_vs_bc{p_seed}"] = {"per_layout": avg, "mean": mean_r}
            print(f"  Ego seed {ego_seed} vs BC seed {p_seed}: mean={mean_r:.1f} {avg}")
    all_results["vs_bc"] = bc_cross

    # ============================================================
    # 6. Cross-play vs Run 3 ego agents (dual encoder)
    # ============================================================
    print("\n" + "="*60)
    print("6. FCP PIXEL EGO vs RUN 3 EGO AGENTS")
    print("="*60)
    ego_cross = {}

    run3_egos = {
        "sp": ("checkpoints/sp_n2_seed{}.pkl", [0, 1, 2]),
        "fcp_mlp": ("checkpoints/fcp_n2x1_hetero_seed{}.pkl", [0, 1, 2]),
        "mep16": ("checkpoints/mep16_seed{}.pkl", [0, 1, 2]),
        "mep8": ("checkpoints/mep8_seed{}.pkl", [0, 1, 2]),
    }

    # Note: Run 3 FCP/MEP/SP egos use egocentric encoding
    # But current fcp_n2x1_hetero_seed{0-5} are pixel — skip those for "fcp_mlp"
    # The Run 3 FCP egos were overwritten by Run 4! Check if they're MLP or pixel:
    with open("checkpoints/fcp_n2x1_hetero_seed0.pkl", "rb") as f:
        check = pickle.load(f)
    if check["config"].get("obs_encoding") == "pixel":
        # Run 4 overwrote Run 3 FCP. Use run3_backup if available
        if os.path.exists("checkpoints/run3_backup/fcp_n2x1_hetero_seed0.pkl"):
            run3_egos["fcp_mlp"] = ("checkpoints/run3_backup/fcp_n2x1_hetero_seed{}.pkl", [0, 1, 2])
            print("  (Using run3_backup for FCP MLP egos)")
        else:
            print("  WARNING: Run 3 FCP egos not available (overwritten by Run 4)")
            del run3_egos["fcp_mlp"]

    for method, (path_template, seeds) in run3_egos.items():
        for p_seed in seeds:
            path = path_template.format(p_seed)
            if not os.path.exists(path):
                continue
            p_params = load_ego_params(path)
            layouts = BCP_LAYOUTS if method == "bcp" else LAYOUT_NAMES

            for ego_seed in pixel_ego_seeds:
                ego_params = load_ego_params(f"checkpoints/fcp_n2x1_hetero_seed{ego_seed}.pkl")
                r = eval_ego_vs_partner(
                    ego_network, ego_params, ego_encoder,
                    partner_network, p_params, partner_encoder,
                    layouts
                )
                mean_r = float(np.mean(list(r.values())))
                key = f"pixel_fcp{ego_seed}_vs_{method}{p_seed}"
                ego_cross[key] = {"per_layout": r, "mean": mean_r}
                print(f"  Pixel FCP seed {ego_seed} vs {method} seed {p_seed}: mean={mean_r:.1f}")
    all_results["vs_run3_egos"] = ego_cross

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY: FCP PIXEL EGO CROSS-PLAY")
    print("="*60)

    # Self-play
    sp_means = [np.mean(list(v.values())) for v in selfplay_results.values()]
    print(f"\n  Self-play mean: {np.mean(sp_means):.1f} (across {len(sp_means)} seeds)")

    # vs populations (H/M/L)
    for pop_name, pop_results in [("SP Pop", sp_cross), ("MEP-16 Pop", mep16_cross), ("MEP-8 Pop", mep8_cross)]:
        if not pop_results:
            continue
        for tier in ["high", "mid", "low"]:
            tier_means = [v[tier]["mean"] for v in pop_results.values() if tier in v]
            if tier_means:
                print(f"  vs {pop_name} {tier.upper()}: {np.mean(tier_means):.1f}")

    # vs BC
    bc_means = [v["mean"] for v in bc_cross.values()]
    if bc_means:
        print(f"  vs BC: {np.mean(bc_means):.1f}")

    # vs Run 3 egos
    for method in ["sp", "fcp_mlp", "mep16", "mep8"]:
        method_means = [v["mean"] for k, v in ego_cross.items() if method in k]
        if method_means:
            print(f"  vs Run 3 {method}: {np.mean(method_means):.1f}")

    # Save all results
    save_path = "reports/eval_results/fcp_pixel_full_eval.json"
    # Convert numpy types for JSON
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(save_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\n  Full results saved to {save_path}")


if __name__ == "__main__":
    main()
