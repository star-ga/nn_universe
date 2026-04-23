"""V4.0 uniqueness experiment — run cross-substrate FIM-tier analysis.

For each baseline system (neural network + 5 alternatives) computes the
tier-1 / tier-3 importance ratio across ``n_seeds`` random initializations.

Output: ``v4_0_uniqueness_results.json`` with per-baseline per-seed tier
ratios and summary statistics (mean, CV).

Usage::

    python run_uniqueness.py --seeds 6 --probes 32

Keep --probes small when debugging (8-16); production is 64-256.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from baselines import REGISTRY, make


# ---------------------------------------------------------------------------
# Tier analysis (same definition as V1.2 scaling_experiment_extended)
# ---------------------------------------------------------------------------


RATIO_CAP = 1e10  # numerical cap; above this the ratio is not physical


def tier_ratio(importance: np.ndarray) -> tuple[float, float, float]:
    """Return (tier1_mean, tier3_mean, tier1/tier3 ratio).

    Tier-1 = top 1% of parameters by importance.
    Tier-3 = bottom 50%.

    Matches the definition used throughout nn_universe.

    For ill-conditioned systems where tier-3 collapses to ~0 (e.g. softmax
    circuits with dormant branches), the ratio is not physically meaningful.
    We floor tier-3 to the smallest positive importance value seen in the
    system, capped at ``RATIO_CAP``, and return a finite number rather than
    inf. Analysis code treats ``>= RATIO_CAP`` as "degenerate."
    """
    x = np.sort(importance.astype(np.float64))[::-1]
    n = len(x)
    if n < 20:
        # Small system: fall back to top 10% / bottom 50%
        t1_cut = max(1, n // 10)
    else:
        t1_cut = max(1, n // 100)
    t3_cut = max(1, n // 2)
    t1_mean = float(np.mean(x[:t1_cut]))
    t3_mean = float(np.mean(x[-t3_cut:]))
    # If tier-3 is exactly zero (many zeros in the distribution), floor it
    # to the smallest positive value observed, or to the global mean / 1e8
    # as an outer safety bound. This keeps the ratio finite and bounded.
    if t3_mean <= 1e-20:
        positive = x[x > 0]
        if positive.size > 0:
            t3_mean_safe = float(np.min(positive))
        else:
            t3_mean_safe = 1e-20
    else:
        t3_mean_safe = t3_mean
    ratio = t1_mean / t3_mean_safe
    if not np.isfinite(ratio):
        ratio = RATIO_CAP
    ratio = float(min(ratio, RATIO_CAP))
    return t1_mean, t3_mean, ratio


def tier_mass(importance: np.ndarray) -> float:
    """Fraction of total importance concentrated in top 1%."""
    x = np.sort(importance.astype(np.float64))[::-1]
    t1_cut = max(1, len(x) // 100)
    total = float(np.sum(x)) + 1e-20
    return float(np.sum(x[:t1_cut]) / total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(seeds: int, probes: int, out_path: Path) -> dict:
    results: dict = {"baselines": {}, "meta": {"seeds": seeds, "probes": probes}}
    for spec in REGISTRY:
        print(f"\n=== {spec.name} (n_params={spec.n_params}) ===")
        per_seed: list[dict] = []
        for s in range(seeds):
            t0 = time.time()
            sys = make(spec.name, seed=s)
            importance = sys.parameter_importance(probes)
            t1_mean, t3_mean, ratio = tier_ratio(importance)
            mass = tier_mass(importance)
            elapsed = time.time() - t0
            print(
                f"  seed={s}  t1/t3={ratio:>10.2f}  top1%_mass={mass:.4f}"
                f"  ({elapsed:.1f}s)"
            )
            per_seed.append(
                {
                    "seed": s,
                    "tier1_mean": t1_mean,
                    "tier3_mean": t3_mean,
                    "tier_ratio": ratio,
                    "top1pct_mass": mass,
                    "n_params_actual": int(importance.size),
                    "elapsed_s": elapsed,
                }
            )
        ratios = np.array([r["tier_ratio"] for r in per_seed])
        masses = np.array([r["top1pct_mass"] for r in per_seed])
        ratio_cv = float(np.std(ratios) / np.mean(ratios)) if np.mean(ratios) > 0 else float("inf")
        mass_cv = float(np.std(masses) / np.mean(masses)) if np.mean(masses) > 0 else float("inf")
        results["baselines"][spec.name] = {
            "n_params": spec.n_params,
            "per_seed": per_seed,
            "tier_ratio_mean": float(np.mean(ratios)),
            "tier_ratio_std": float(np.std(ratios)),
            "tier_ratio_cv": ratio_cv,
            "top1pct_mass_mean": float(np.mean(masses)),
            "top1pct_mass_cv": mass_cv,
        }
        print(
            f"  → mean tier_ratio = {np.mean(ratios):.2f}"
            f"  (CV {100*ratio_cv:.1f}%)   mean top1%_mass = {np.mean(masses):.4f}"
        )

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--probes", type=int, default=32)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "v4_0_uniqueness_results.json",
    )
    args = p.parse_args()
    run(args.seeds, args.probes, args.out)


if __name__ == "__main__":
    main()
