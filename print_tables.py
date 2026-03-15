"""Print evaluation tables from JSON results in reports/eval_results/."""

import json
import os
import glob
import numpy as np

EVAL_DIR = "reports/eval_results"
LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"]
BCP_LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring"]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, width=10):
    if val is None:
        return f"{'—':>{width}}"
    return f"{val:>{width}.1f}"


def print_separator(width):
    print("-" * width)


def print_table1_section(title, pattern, layouts, agent_prefix="Agent"):
    """Print self-play results for a population."""
    files = sorted(glob.glob(os.path.join(EVAL_DIR, pattern)))
    if not files:
        print(f"\n  {title}: No results yet")
        return

    print(f"\n  {title}")
    header = f"  {'Agent':>16}" + "".join(f"{l[:12]:>12}" for l in layouts) + f"{'Mean':>10}"
    print(header)
    print_separator(len(header))

    all_means = []
    for fpath in files:
        data = load_json(fpath)
        if not data or "results" not in data:
            continue
        results = data["results"]
        for agent_name, layout_rewards in results.items():
            vals = [layout_rewards.get(l) for l in layouts]
            valid = [v for v in vals if v is not None]
            mean = np.mean(valid) if valid else None
            all_means.append(mean)
            row = f"  {agent_name:>16}" + "".join(fmt(v, 12) for v in vals) + fmt(mean)
            print(row)

    if all_means:
        overall = np.mean([m for m in all_means if m is not None])
        print(f"  {'OVERALL':>16}" + " " * (12 * len(layouts)) + fmt(overall))


def print_table2():
    """Print ego self-play results."""
    methods = {
        "SP": ("sp_ego_selfplay_seed{}.json", LAYOUTS),
        "MEP-16": ("mep16_ego_selfplay_seed{}.json", LAYOUTS),
        "MEP-8": ("mep8_ego_selfplay_seed{}.json", LAYOUTS),
        "BCP": ("bcp_ego_selfplay_seed{}.json", BCP_LAYOUTS),
        "FCP": ("fcp_ego_selfplay_seed{}.json", LAYOUTS),
    }

    print("\n" + "=" * 90)
    print("TABLE 2: Ego Self-Play Performance")
    print("=" * 90)

    header = f"  {'Method':>12}" + "".join(f"{l[:12]:>12}" for l in LAYOUTS) + f"{'Mean':>10}"
    print(header)
    print_separator(len(header))

    for method, (pattern, layouts) in methods.items():
        seed_results = []
        for seed in [0, 1, 2]:
            data = load_json(os.path.join(EVAL_DIR, pattern.format(seed)))
            if data and "results" in data:
                seed_results.append(data["results"])

        if not seed_results:
            vals = [None] * len(LAYOUTS)
            mean = None
        else:
            vals = []
            for l in LAYOUTS:
                l_vals = [r.get(l) for r in seed_results if r.get(l) is not None]
                vals.append(np.mean(l_vals) if l_vals else None)
            valid = [v for v in vals if v is not None]
            mean = np.mean(valid) if valid else None

        row = f"  {method:>12}" + "".join(fmt(v, 12) for v in vals) + fmt(mean)
        print(row)


def print_cross_play_summary():
    """Print cross-play summary table."""
    ego_methods = ["sp", "mep16", "mep8", "bcp", "fcp"]
    partner_types = ["sp", "mep16", "mep8", "bc"]

    print("\n" + "=" * 80)
    print("TABLE 4: Cross-Play Summary (mean sparse reward)")
    print("=" * 80)

    header = f"  {'Ego':>10}" + "".join(f"{'vs '+p:>14}" for p in partner_types) + f"{'Overall':>12}"
    print(header)
    print_separator(len(header))

    for ego in ego_methods:
        partner_means = []
        for ptype in partner_types:
            all_vals = []
            for ego_seed in [0, 1, 2]:
                for p_seed in [3, 4]:
                    path = os.path.join(EVAL_DIR, f"cross_{ego}_s{ego_seed}_vs_{ptype}_s{p_seed}.json")
                    data = load_json(path)
                    if data and "results" in data:
                        for layout_r in data["results"].values() if isinstance(data["results"], dict) else []:
                            if isinstance(layout_r, dict):
                                all_vals.extend(layout_r.values())
                            else:
                                all_vals.append(layout_r)
            partner_means.append(np.mean(all_vals) if all_vals else None)

        valid = [v for v in partner_means if v is not None]
        overall = np.mean(valid) if valid else None
        row = f"  {ego.upper():>10}" + "".join(fmt(v, 14) for v in partner_means) + fmt(overall, 12)
        print(row)


def print_all():
    """Print all available tables."""
    print("\n" + "=" * 90)
    print("TABLE 1: Self-Play Performance (Populations & BC)")
    print("=" * 90)

    print_table1_section("SP Population", "sp_pop_selfplay_seed*.json", LAYOUTS)
    print_table1_section("MEP-16 Population", "mep16_pop_selfplay_seed*.json", LAYOUTS)
    print_table1_section("MEP-8 Population", "mep8_pop_selfplay_seed*.json", LAYOUTS)
    print_table1_section("BC Agents", "bc_selfplay_seed*.json", BCP_LAYOUTS)

    print_table2()
    print_cross_play_summary()
    print()


if __name__ == "__main__":
    print_all()
