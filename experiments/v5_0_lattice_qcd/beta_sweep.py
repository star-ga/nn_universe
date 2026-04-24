"""V7.1 — U(1) lattice β-sweep.

V5.0 measured T1/T3 = 1.62 ± 0.005 at β = 1.0 (L=8, d=4). The question
V7.1 asks: is the "non-deep, T1/T3 ≈ O(1)" characterisation stable
across the full range of gauge couplings — including through the
deconfinement crossover in 4D compact U(1)?

Prediction: YES. Lattice U(1) at every β is a spatially-parallel QFT
with no sequential composition chain, so T1/T3 stays O(1) regardless
of β. If T1/T3 varied with β (e.g., spiked near a phase boundary), the
dichotomy's "deep-layered-sequential only" claim would need hedging.

Sweep: β ∈ {0.1, 0.5, 1.0, 2.0, 5.0}, L=6 (smaller than V5.0's L=8 to
fit β=5.0 thermalisation budget), d=4, 3 seeds each. Uses the same
gradient-fixed action code as V5.0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lattice_u1 import (  # noqa: E402
    init_links, action, grad_action, metropolis_sweep_local, tier_ratio,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--thermalise", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--decorr", type=int, default=20)
    ap.add_argument("--step-size", type=float, default=0.6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=str,
                    default=str(HERE / "v7_1_beta_sweep_results.json"))
    args = ap.parse_args()

    all_runs: list[dict] = []
    per_beta: dict[float, list[float]] = {}

    for beta in args.betas:
        per_beta[beta] = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            theta = init_links(args.L, args.d, rng)
            n_params = theta.size

            print(
                f"β={beta:<4}  seed={seed}  L={args.L}  d={args.d}  "
                f"n_params={n_params:,}",
                flush=True,
            )

            t0 = time.time()
            for _ in range(args.thermalise):
                metropolis_sweep_local(theta, beta, args.d, args.step_size, rng)
            t_therm = time.time() - t0
            print(f"  thermalised in {t_therm:.1f}s", flush=True)

            t0 = time.time()
            fim = np.zeros(n_params, dtype=np.float64)
            for _ in range(args.n_samples):
                for _ in range(args.decorr):
                    metropolis_sweep_local(theta, beta, args.d, args.step_size, rng)
                g = grad_action(theta, beta, args.d).ravel()
                fim += g * g
            fim /= args.n_samples
            t_meas = time.time() - t0

            t1m, t3m, ratio = tier_ratio(fim)
            print(
                f"  measured in {t_meas:.1f}s  T1={t1m:.3e}  T3={t3m:.3e}  "
                f"T1/T3={ratio:.3f}",
                flush=True,
            )

            row = {
                "beta": beta, "seed": seed, "L": args.L, "d": args.d,
                "n_params": int(n_params),
                "fim_tier1_mean": t1m, "fim_tier3_mean": t3m, "fim_tier1_tier3": ratio,
                "thermalise_s": t_therm, "measure_s": t_meas,
            }
            all_runs.append(row)
            per_beta[beta].append(ratio)

    print("\n==== β-sweep summary ====")
    summary: dict[float, dict] = {}
    for beta in args.betas:
        ratios = np.array(per_beta[beta])
        mean_r = float(ratios.mean())
        std_r = float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0
        cv = std_r / mean_r if mean_r > 0 else 0.0
        summary[str(beta)] = {
            "mean": mean_r, "std": std_r, "cv": cv, "n_seeds": len(ratios),
            "per_seed": ratios.tolist(),
        }
        print(f"  β={beta:<5}  T1/T3 mean={mean_r:.3f}  CV={cv*100:.2f}%")

    payload = {
        "config": vars(args),
        "per_run": all_runs,
        "per_beta": summary,
        "prediction": (
            "T1/T3 should stay O(1) across all β, because lattice U(1) is "
            "spatially parallel at every coupling. Variation within a factor "
            "of ~3 is expected from physical-observable β-dependence; >10× "
            "variation would challenge the 'non-deep only' characterisation."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
