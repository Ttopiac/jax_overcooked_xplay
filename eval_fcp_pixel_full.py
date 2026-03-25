"""Full cross-play evaluation for Run 3 (MLP) and Run 4 (Pixel CNN).

Uses categorize_by_skill_per_layout for H/M/L partner selection:
  H = 1 best agent per layout
  M = 1 agent closest to 50% of H per layout
  L = 1 worst agent per layout

Different agents are H/M/L for different layouts.
Applied consistently to SP Pop, MEP-16 Pop, MEP-8 Pop.
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
from evaluate import evaluate_episode, categorize_by_skill_per_layout

LAYOUT_NAMES = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]
BCP_LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring"]
NUM_EVAL_ENVS = 32
NUM_BATCHES = 3  # 3 × 32 = 96 episodes


def eval_one_layout(env, net_a, params_a, enc_a, net_b, params_b, enc_b, num_envs=NUM_EVAL_ENVS):
    """Evaluate one pairing on one layout. Returns mean sparse reward."""
    @jax.jit
    def _eval(key):
        return evaluate_episode(env, net_a, params_a, net_b, params_b, enc_a, key, num_envs, encoder_b=enc_b)

    rewards = []
    for b in range(NUM_BATCHES):
        key = jax.random.PRNGKey(b * 1000 + 42)
        rewards.append(float(_eval(key)))
    return float(np.mean(rewards))


def load_ego_params(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "params" in data:
        return data["params"]
    elif "ego_params" in data:
        return data["ego_params"]
    raise ValueError(f"Unknown format: {list(data.keys())}")


def build_per_layout_hml(pop_path, scores_path):
    """Build per-layout H/M/L using categorize_by_skill_per_layout.

    Returns:
        per_layout_pool: tree of (5, 3, ...) where dim1 = [H, M, L]
        hml_info: dict with H/M/L scores per layout for logging
    """
    with open(pop_path, "rb") as f:
        pop_data = pickle.load(f)
    with open(scores_path, "rb") as f:
        scores_data = pickle.load(f)

    if "ckpt_params" in pop_data:
        flat_params = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), pop_data["ckpt_params"])
    elif "all_params" in pop_data:
        flat_params = pop_data["all_params"]
    else:
        raise ValueError(f"Unknown format: {list(pop_data.keys())}")

    layout_scores = scores_data["layout_scores"]
    n_total = jax.tree.leaves(flat_params)[0].shape[0]

    # Handle size mismatch
    if layout_scores.shape[0] != n_total:
        n_use = min(layout_scores.shape[0], n_total)
        layout_scores = layout_scores[:n_use]
        flat_params = jax.tree.map(lambda x: x[:n_use], flat_params)

    per_layout_pool = categorize_by_skill_per_layout(
        jax.device_put(flat_params), jnp.array(layout_scores), LAYOUT_NAMES
    )

    # Log H/M/L scores
    hml_info = {}
    for li, name in enumerate(LAYOUT_NAMES):
        scores = layout_scores[:, li]
        h_score = float(scores.max())
        l_score = float(scores.min())
        target_mid = h_score * 0.5
        m_score = float(scores[np.argmin(np.abs(scores - target_mid))])
        hml_info[name] = {"H": h_score, "M": m_score, "L": l_score}

    return per_layout_pool, hml_info


def main():
    t_start = time.time()

    # === Setup encoders and networks ===
    pixel_encoder = ObsEncoder(mode="pixel")
    pixel_network = IMPALACNNActorCritic(action_dim=6, palette=pixel_encoder.atlas.palette)

    ego_encoder = ObsEncoder(mode="egocentric")
    with open("checkpoints/sp_pop_seed0.pkl", "rb") as f:
        sp_data = pickle.load(f)
    mlp_dims = sp_data["config"].get("hidden_dims", (256, 256))
    mlp_network = ActorCritic(action_dim=6, hidden_dims=mlp_dims)

    with open("checkpoints/bc_seed0.pkl", "rb") as f:
        bc_data = pickle.load(f)
    bc_dims = bc_data.get("bc_hidden_dims", (64, 64))
    bc_network = ActorCritic(action_dim=6, hidden_dims=bc_dims)

    envs = {name: Overcooked(layout=overcooked_layouts[name], num_agents=2) for name in LAYOUT_NAMES}

    # === Define ego agents ===
    run3_egos = {}
    for method, path_tmpl, seeds in [
        ("SP", "checkpoints/sp_n2_seed{}.pkl", [0, 1, 2]),
        ("MEP-16", "checkpoints/mep16_seed{}.pkl", [0, 1, 2]),
        ("MEP-8", "checkpoints/mep8_seed{}.pkl", [0, 1, 2]),
        ("BCP", "checkpoints/run3_backup/bcp_n2x1_hetero_seed{}.pkl", [0, 1, 2]),
        ("FCP", "checkpoints/run3_backup/fcp_n2x1_hetero_seed{}.pkl", [0, 1, 2]),
    ]:
        agent_list = []
        for s in seeds:
            path = path_tmpl.format(s)
            if os.path.exists(path):
                agent_list.append((s, jax.device_put(load_ego_params(path))))
        if agent_list:
            run3_egos[method] = agent_list

    run4_egos = []
    for s in range(6):
        path = f"checkpoints/fcp_n2x1_hetero_seed{s}.pkl"
        if os.path.exists(path):
            run4_egos.append((s, jax.device_put(load_ego_params(path))))

    print(f"Run 3 egos: {', '.join(f'{k}({len(v)})' for k, v in run3_egos.items())}")
    print(f"Run 4 egos: FCP pixel({len(run4_egos)})")

    # === Build per-layout H/M/L pools for all populations ===
    pop_pools = {}  # {(pop_type, pop_seed): per_layout_pool}
    for pop_type in ["sp_pop", "mep16_pop", "mep8_pop"]:
        for p_seed in [3, 4]:
            pop_path = f"checkpoints/{pop_type}_seed{p_seed}.pkl"
            scores_path = f"checkpoints/{pop_type}_seed{p_seed}_layout_scores.pkl"
            if not os.path.exists(pop_path) or not os.path.exists(scores_path):
                print(f"  SKIP {pop_type} seed {p_seed} (missing files)")
                continue
            pool, hml_info = build_per_layout_hml(pop_path, scores_path)
            pop_pools[(pop_type, p_seed)] = pool
            print(f"  {pop_type} seed {p_seed}: loaded H/M/L per layout")
            for name in LAYOUT_NAMES:
                info = hml_info[name]
                print(f"    {name}: H={info['H']:.0f} M={info['M']:.0f} L={info['L']:.0f}")

    # === Load BC agents ===
    bc_agents = {}  # {bc_seed: [params_list]}
    for p_seed in [3, 4]:
        bc_path = f"checkpoints/bc_seed{p_seed}.pkl"
        if not os.path.exists(bc_path):
            continue
        with open(bc_path, "rb") as f:
            bd = pickle.load(f)
        agents = []
        for layout in bd["bc_params"]:
            for p in bd["bc_params"][layout]:
                agents.append(jax.device_put(p))
        bc_agents[p_seed] = agents

    # === Storage for all results ===
    # results[ego_method][ego_seed][pop_type][pop_seed][layout][tier] = score
    results = {}

    def record(ego_method, ego_seed, pop_type, pop_seed, layout, tier, score):
        results.setdefault(ego_method, {})
        results[ego_method].setdefault(ego_seed, {})
        results[ego_method][ego_seed].setdefault(pop_type, {})
        results[ego_method][ego_seed][pop_type].setdefault(pop_seed, {})
        results[ego_method][ego_seed][pop_type][pop_seed].setdefault(layout, {})
        results[ego_method][ego_seed][pop_type][pop_seed][layout][tier] = score

    # ============================================================
    # TABLE A: Self-play
    # ============================================================
    print("\n" + "=" * 60)
    print("TABLE A: SELF-PLAY")
    print("=" * 60)

    for method, agents in run3_egos.items():
        for seed, params in agents:
            layouts = BCP_LAYOUTS if method == "BCP" else LAYOUT_NAMES
            for layout in layouts:
                score = eval_one_layout(envs[layout], mlp_network, params, ego_encoder,
                                         mlp_network, params, ego_encoder)
                record(f"R3_{method}", seed, "selfplay", 0, layout, "self", score)
            scores = [results[f"R3_{method}"][seed]["selfplay"][0][l]["self"] for l in layouts]
            print(f"  R3 {method} s{seed}: mean={np.mean(scores):.1f}")

    for seed, params in run4_egos:
        for layout in LAYOUT_NAMES:
            score = eval_one_layout(envs[layout], pixel_network, params, pixel_encoder,
                                     pixel_network, params, pixel_encoder)
            record("R4_FCP_pixel", seed, "selfplay", 0, layout, "self", score)
        scores = [results["R4_FCP_pixel"][seed]["selfplay"][0][l]["self"] for l in LAYOUT_NAMES]
        print(f"  R4 FCP pixel s{seed}: mean={np.mean(scores):.1f}")

    # ============================================================
    # TABLES B: Cross-play vs populations (H/M/L per layout)
    # ============================================================
    print("\n" + "=" * 60)
    print("TABLES B: CROSS-PLAY vs POPULATIONS (H/M/L per layout)")
    print("=" * 60)

    tier_names = ["H", "M", "L"]

    def eval_ego_vs_pop_hml(ego_method, ego_seed, ego_params, ego_net, ego_enc,
                             partner_net, partner_enc):
        """Evaluate one ego against all pop H/M/L per layout."""
        for pop_type in ["sp_pop", "mep16_pop", "mep8_pop"]:
            for p_seed in [3, 4]:
                key = (pop_type, p_seed)
                if key not in pop_pools:
                    continue
                pool = pop_pools[key]  # tree of (5, 3, ...)
                for li, layout in enumerate(LAYOUT_NAMES):
                    for ti, tier in enumerate(tier_names):
                        partner_params = jax.tree.map(lambda x: x[li, ti], pool)
                        score = eval_one_layout(envs[layout], ego_net, ego_params, ego_enc,
                                                 partner_net, partner_params, partner_enc)
                        record(ego_method, ego_seed, pop_type, p_seed, layout, tier, score)

    # Run 3 egos vs populations
    for method, agents in run3_egos.items():
        for seed, params in agents:
            print(f"  R3 {method} s{seed} vs populations...")
            eval_ego_vs_pop_hml(f"R3_{method}", seed, params, mlp_network, ego_encoder,
                                 mlp_network, ego_encoder)

    # Run 4 pixel ego vs populations (dual encoder)
    for seed, params in run4_egos:
        print(f"  R4 FCP pixel s{seed} vs populations...")
        eval_ego_vs_pop_hml("R4_FCP_pixel", seed, params, pixel_network, pixel_encoder,
                             mlp_network, ego_encoder)

    # === BC cross-play ===
    print("\n  BC cross-play...")
    for ego_method_key, agents, ego_net, ego_enc in [
        *[(f"R3_{m}", a, mlp_network, ego_encoder) for m, a in run3_egos.items()],
        ("R4_FCP_pixel", run4_egos, pixel_network, pixel_encoder),
    ]:
        for seed, params in agents:
            for p_seed, bc_list in bc_agents.items():
                bc_scores = {l: [] for l in BCP_LAYOUTS}
                for bp in bc_list:
                    for layout in BCP_LAYOUTS:
                        s = eval_one_layout(envs[layout], ego_net, params, ego_enc,
                                             bc_network, bp, ego_encoder)
                        bc_scores[layout].append(s)
                for layout in BCP_LAYOUTS:
                    avg = float(np.mean(bc_scores[layout]))
                    record(ego_method_key, seed, "bc", p_seed, layout, "all", avg)

    # ============================================================
    # TABLE C: Run 4 pixel ego vs Run 3 ego agents
    # ============================================================
    print("\n" + "=" * 60)
    print("TABLE C: RUN 4 PIXEL FCP vs RUN 3 EGO AGENTS")
    print("=" * 60)

    for seed, params in run4_egos:
        for method, agents in run3_egos.items():
            for p_seed, p_params in agents:
                layouts = BCP_LAYOUTS if method == "BCP" else LAYOUT_NAMES
                for layout in layouts:
                    score = eval_one_layout(envs[layout], pixel_network, params, pixel_encoder,
                                             mlp_network, p_params, ego_encoder)
                    record("R4_FCP_pixel", seed, f"r3ego_{method}", p_seed, layout, "ego", score)
        print(f"  R4 pixel s{seed} vs all R3 egos done")

    # ============================================================
    # COMPILE AND PRINT TABLES
    # ============================================================
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE ({elapsed/60:.0f} min)")
    print(f"{'=' * 60}")

    # Helper to average across seeds and pop_seeds
    def avg_score(ego_method, pop_type, layout, tier):
        """Average across all ego seeds and pop seeds."""
        if ego_method not in results:
            return None
        scores = []
        for ego_seed in results[ego_method]:
            if pop_type not in results[ego_method][ego_seed]:
                continue
            for p_seed in results[ego_method][ego_seed][pop_type]:
                r = results[ego_method][ego_seed][pop_type][p_seed]
                if layout in r and tier in r[layout]:
                    scores.append(r[layout][tier])
        return float(np.mean(scores)) if scores else None

    # Print Table A
    print("\n--- TABLE A: Self-Play ---")
    all_methods = [f"R3_{m}" for m in run3_egos] + ["R4_FCP_pixel"]
    for method in all_methods:
        layouts = BCP_LAYOUTS if "BCP" in method else LAYOUT_NAMES
        scores = [avg_score(method, "selfplay", l, "self") for l in layouts]
        scores = [s for s in scores if s is not None]
        mean = np.mean(scores) if scores else 0
        score_str = " ".join(f"{s:>8.1f}" for s in scores)
        print(f"  {method:>20s}: {score_str}  mean={mean:.1f}")

    # Print Tables B (per layout)
    pop_types = ["sp_pop", "mep16_pop", "mep8_pop"]
    pop_short = {"sp_pop": "SP", "mep16_pop": "M16", "mep8_pop": "M8"}

    for layout in LAYOUT_NAMES:
        print(f"\n--- TABLE B: {layout} ---")
        header = "                    "
        for pt in pop_types:
            for t in tier_names:
                header += f" {pop_short[pt]}{t:>2s}"
        header += "    BC"
        print(header)
        for method in all_methods:
            if "BCP" in method and layout not in BCP_LAYOUTS:
                continue
            row = f"  {method:>20s}"
            for pt in pop_types:
                for t in tier_names:
                    s = avg_score(method, pt, layout, t)
                    row += f" {s:>5.0f}" if s is not None else "     —"
            bc_s = avg_score(method, "bc", layout, "all")
            row += f" {bc_s:>5.0f}" if bc_s is not None else "     —"
            print(row)

    # Print Table C
    print(f"\n--- TABLE C: R4 Pixel FCP vs R3 Egos ---")
    r3_methods = list(run3_egos.keys())
    header = "              " + "".join(f" {m:>8s}" for m in r3_methods)
    print(header)
    for layout in LAYOUT_NAMES:
        row = f"  {layout:>14s}"
        for method in r3_methods:
            s = avg_score("R4_FCP_pixel", f"r3ego_{method}", layout, "ego")
            row += f" {s:>8.1f}" if s is not None else "        —"
        print(row)
    # Mean row
    row = f"  {'Mean':>14s}"
    for method in r3_methods:
        layouts = BCP_LAYOUTS if method == "BCP" else LAYOUT_NAMES
        scores = [avg_score("R4_FCP_pixel", f"r3ego_{method}", l, "ego") for l in layouts]
        scores = [s for s in scores if s is not None]
        row += f" {np.mean(scores):>8.1f}" if scores else "        —"
    print(row)

    # Print Table D summary
    print(f"\n--- TABLE D: Overall Summary ---")
    header = "                     Self  SP_H  SP_M  SP_L M16_H M16_M M16_L  M8_H  M8_M  M8_L    BC  Egos   XP"
    print(header)
    for method in all_methods:
        layouts = BCP_LAYOUTS if "BCP" in method else LAYOUT_NAMES
        self_scores = [avg_score(method, "selfplay", l, "self") for l in layouts]
        self_mean = np.mean([s for s in self_scores if s is not None]) if any(s is not None for s in self_scores) else 0

        xp_all = []
        row = f"  {method:>20s} {self_mean:>5.1f}"
        for pt in pop_types:
            for t in tier_names:
                scores = [avg_score(method, pt, l, t) for l in layouts]
                scores = [s for s in scores if s is not None]
                m = np.mean(scores) if scores else 0
                xp_all.extend(scores)
                row += f" {m:>5.0f}"

        bc_scores = [avg_score(method, "bc", l, "all") for l in BCP_LAYOUTS]
        bc_scores = [s for s in bc_scores if s is not None]
        bc_mean = np.mean(bc_scores) if bc_scores else 0
        xp_all.extend(bc_scores)
        row += f" {bc_mean:>5.0f}"

        # Ego cross-play (only for R4)
        if "R4" in method:
            ego_scores = []
            for rm in run3_egos:
                ls = BCP_LAYOUTS if rm == "BCP" else LAYOUT_NAMES
                for l in ls:
                    s = avg_score(method, f"r3ego_{rm}", l, "ego")
                    if s is not None:
                        ego_scores.append(s)
            ego_mean = np.mean(ego_scores) if ego_scores else 0
            xp_all.extend(ego_scores)
            row += f" {ego_mean:>5.0f}"
        else:
            row += "     —"

        xp_mean = np.mean(xp_all) if xp_all else 0
        row += f" {xp_mean:>5.0f}"
        print(row)

    # Save raw results
    def to_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json(v) for v in obj]
        return obj

    with open("reports/eval_results/run3_run4_crossplay_raw.json", "w") as f:
        json.dump(to_json(results), f, indent=2)
    print(f"\nRaw results saved to reports/eval_results/run3_run4_crossplay_raw.json")


if __name__ == "__main__":
    main()
