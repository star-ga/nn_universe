"""V12 aggregator — pool all V12 result JSONs into one master JSON.

Reads every *.json file under --results-dir and produces:
  v12_aggregate.json with per-substrate, per-item summaries

Usage:
    python3 experiments/v12_partition_invariant/aggregate.py \
        --results-dir /data/checkpoints/v12_cluster_followup/results \
        --out v12_aggregate.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_substrate = defaultdict(list)
    for f in sorted(Path(args.results_dir).glob("*.json")):
        try:
            d = json.load(open(f))
            by_substrate[d.get("substrate", "unknown")].append(d)
        except Exception as e:
            print(f"  [warn] skip {f.name}: {e}")

    aggregate = {}
    for substrate, runs in by_substrate.items():
        t1t3_values = [r["t1_t3"] for r in runs if "t1_t3" in r and np.isfinite(r["t1_t3"])]
        if not t1t3_values:
            continue
        t1t3_values = np.array(t1t3_values, dtype=np.float64)
        log10_values = np.log10(t1t3_values)
        ginis = [r.get("gini", float("nan")) for r in runs if "gini" in r]
        rranks = [r.get("effective_rank_n", float("nan")) for r in runs if "effective_rank_n" in r]
        topks = [r.get("top_1pct_mass", float("nan")) for r in runs if "top_1pct_mass" in r]

        aggregate[substrate] = {
            "n_seeds":             len(runs),
            "t1_t3_mean":          float(np.mean(t1t3_values)),
            "t1_t3_std":           float(np.std(t1t3_values)),
            "t1_t3_min":           float(np.min(t1t3_values)),
            "t1_t3_max":           float(np.max(t1t3_values)),
            "log10_t1t3_mean":     float(np.mean(log10_values)),
            "log10_t1t3_std":      float(np.std(log10_values)),
            "log10_t1t3_ci95":     [
                float(np.percentile(log10_values, 2.5)),
                float(np.percentile(log10_values, 97.5)),
            ] if len(log10_values) >= 3 else None,
            "gini_mean":           float(np.nanmean(ginis)) if ginis else None,
            "effective_rank_n_mean": float(np.nanmean(rranks)) if rranks else None,
            "top_1pct_mass_mean":  float(np.nanmean(topks)) if topks else None,
            "crosses_threshold_100": bool(np.min(t1t3_values) < 100 < np.max(t1t3_values)),
            "all_above_100":       bool(np.min(t1t3_values) >= 100),
            "all_below_100":       bool(np.max(t1t3_values) <= 100),
        }

    with open(args.out, "w") as f:
        json.dump({
            "n_substrates":  len(aggregate),
            "substrates":    aggregate,
            "schema_version": "v12.1",
        }, f, indent=2)

    # Pretty-print headline
    print(f"\n== V12 aggregate ({len(aggregate)} substrates) ==")
    print(f"{'substrate':<35} {'n':>3} {'T1/T3 mean':>12} {'95%CI':>22} {'classification':>16}")
    for s, d in sorted(aggregate.items(), key=lambda x: -x[1]["t1_t3_mean"]):
        ci = d["log10_t1t3_ci95"]
        ci_str = f"[{10**ci[0]:.2g}, {10**ci[1]:.2g}]" if ci else "(single-seed)"
        verdict = ("DEEP" if d["all_above_100"] else
                   "REST" if d["all_below_100"] else
                   "AMBIGUOUS")
        print(f"{s:<35} {d['n_seeds']:>3} {d['t1_t3_mean']:>12.3g} {ci_str:>22} {verdict:>16}")


if __name__ == "__main__":
    main()
