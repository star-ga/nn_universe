"""V6.1 — width sweep at fixed depth.

Hanin-Nica's log-normal theorem is asymptotic in BOTH depth and width.
V6.0 tested depth at fixed width 64. This tests width at fixed depth 8:
does log-normality sharpen (skew, excess kurtosis → 0) as width grows?

Predictions:
  (H4)  At fixed L, Var[log F_ii] is approximately width-independent.
        The per-layer log-variance σ² depends only on the activation,
        not on width n.
  (H5)  |skew|, |excess kurtosis| of log F_ii decrease monotonically
        with width. Log-normality is a large-n asymptotic.
  (H6)  T1/T3 is approximately width-independent at fixed depth (the
        shape of the log-normal is set by depth; width only tightens
        the Gaussian approximation to log F).

If H4+H5+H6 all pass, the Hanin-Nica mechanism is confirmed empirically
along the width axis too, completing the depth × width picture.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from depth_sweep import (  # noqa: E402
    make_net, fim_diagonal, tier_ratio, log_stats, fit_linear,
)


def run_one(width: int, seed: int, depth: int, dim: int, n_probes: int) -> dict:
    torch.manual_seed(seed)
    net = make_net(depth, width, dim)
    n_params = sum(p.numel() for p in net.parameters())
    fim = fim_diagonal(net, dim, n_probes)
    t1, t3, ratio = tier_ratio(fim)
    stats = log_stats(fim)
    return {
        "width": width, "seed": seed, "depth": depth, "dim": dim, "n_probes": n_probes,
        "n_params": int(n_params),
        "tier1_mean": t1, "tier3_mean": t3, "tier_ratio": ratio,
        "log_mean": stats["mean"], "log_var": stats["var"],
        "log_skew": stats["skew"], "log_excess_kurtosis": stats["excess_kurtosis"],
        "fim_nonzero_count": stats["n"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n-probes", type=int, default=1000)
    ap.add_argument("--omp-threads", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v6_1_width_sweep.json"))
    args = ap.parse_args()

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.omp_threads)

    per_run: list[dict] = []
    per_width: dict[int, list[float]] = {}
    per_width_var: dict[int, list[float]] = {}
    per_width_skew: dict[int, list[float]] = {}
    per_width_kurt: dict[int, list[float]] = {}

    for W in args.widths:
        per_width[W] = []
        per_width_var[W] = []
        per_width_skew[W] = []
        per_width_kurt[W] = []
        for seed in args.seeds:
            t0 = time.time()
            row = run_one(W, seed, args.depth, args.dim, args.n_probes)
            row["elapsed_s"] = time.time() - t0
            per_run.append(row)
            lr = math.log(row["tier_ratio"]) if row["tier_ratio"] > 0 else float("nan")
            per_width[W].append(lr)
            per_width_var[W].append(row["log_var"])
            per_width_skew[W].append(row["log_skew"])
            per_width_kurt[W].append(row["log_excess_kurtosis"])
            print(
                f"  width={W:>3}  seed={seed}  N={row['n_params']:>7,}  "
                f"T1/T3={row['tier_ratio']:>.4e}  Var[log F]={row['log_var']:.3f}  "
                f"skew={row['log_skew']:+.2f}  kurt={row['log_excess_kurtosis']:+.2f}  "
                f"({row['elapsed_s']:.1f}s)",
                flush=True,
            )

    widths = sorted(per_width.keys())
    mean_log_ratio = [float(np.mean(per_width[W])) for W in widths]
    mean_log_var = [float(np.mean(per_width_var[W])) for W in widths]
    mean_skew = [float(np.mean(per_width_skew[W])) for W in widths]
    mean_kurt = [float(np.mean(per_width_kurt[W])) for W in widths]

    # H4: Var[log F] is width-independent → slope should be small
    h4_slope, h4_intercept, h4_r2 = fit_linear([float(W) for W in widths], mean_log_var)
    # H5: |skew| and |kurt| decrease with width → slope vs log(width) should be negative
    log_widths = [math.log(W) for W in widths]
    h5_skew_slope, _, h5_skew_r2 = fit_linear(log_widths, [abs(s) for s in mean_skew])
    h5_kurt_slope, _, h5_kurt_r2 = fit_linear(log_widths, [abs(k) for k in mean_kurt])
    # H6: T1/T3 width-independent
    h6_slope, _, h6_r2 = fit_linear([float(W) for W in widths], mean_log_ratio)

    summary = {
        "widths": widths,
        "mean_log_ratio": mean_log_ratio,
        "mean_log_var": mean_log_var,
        "mean_abs_skew": [abs(s) for s in mean_skew],
        "mean_abs_excess_kurtosis": [abs(k) for k in mean_kurt],
        "H4_logvar_width_independent": {
            "slope_per_width_unit": h4_slope, "R2": h4_r2,
            "pass": bool(abs(h4_slope) < 0.02),
        },
        "H5_kurt_skew_decrease_with_width": {
            "skew_slope_per_log_width": h5_skew_slope, "skew_R2": h5_skew_r2,
            "kurt_slope_per_log_width": h5_kurt_slope, "kurt_R2": h5_kurt_r2,
            "pass": bool(h5_skew_slope < 0 and h5_kurt_slope < 0),
        },
        "H6_T1_T3_width_independent": {
            "slope_per_width_unit": h6_slope, "R2": h6_r2,
            "pass": bool(abs(h6_slope) < 0.02),
        },
    }

    print("\n==== hypothesis tests ====")
    print(f"H4 Var[log F] ~ 0·W        slope={h4_slope:.5f}  {'PASS' if summary['H4_logvar_width_independent']['pass'] else 'WEAK'}")
    print(f"H5 |skew| decreases w/ W   slope={h5_skew_slope:+.3f}  R²={h5_skew_r2:.3f}  {'PASS' if summary['H5_kurt_skew_decrease_with_width']['pass'] else 'FAIL'}")
    print(f"   |kurt| decreases w/ W   slope={h5_kurt_slope:+.3f}  R²={h5_kurt_r2:.3f}")
    print(f"H6 log(T1/T3) ~ 0·W        slope={h6_slope:+.5f}  {'PASS' if summary['H6_T1_T3_width_independent']['pass'] else 'WEAK'}")

    payload = {"config": vars(args), "per_run": per_run, "summary": summary}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
