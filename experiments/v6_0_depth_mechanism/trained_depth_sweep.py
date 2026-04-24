"""V6.2 — trained-NN depth sweep: does training preserve the log-normal?

V4.1 found training dissipates the FIM tier hierarchy by 4-24×. V6.0
shows untrained MLPs have Var[log F] ∝ L and log(T1/T3) ∝ √L. Does
training (a) preserve the linear-in-L scaling with a reduced slope, or
(b) flatten it, or (c) introduce a qualitatively different form?

Predictions:
  (P1)  Trained Var[log F] is LOWER than untrained at each L but still
        approximately linear in L (with a reduced slope).
  (P2)  Trained log(T1/T3) ∝ √L holds with a reduced slope (consistent
        with P1 + log-normal quantile analysis).
  (P3)  Trained log F is MORE Gaussian than untrained — training
        regularises the tails (skew, kurtosis closer to 0).

If P1+P2 pass, the mechanism is confirmed post-training; V4.1's "training
dissipates" becomes "training reduces the PoRM variance coefficient."
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


def train(net, dim, steps, lr=1e-3, batch=128):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(steps):
        x = torch.randn(batch, dim)
        y = net(x)
        loss = 0.5 * (y - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def run_one(depth: int, seed: int, width: int, dim: int, n_probes: int, train_steps: int) -> dict:
    torch.manual_seed(seed)
    net = make_net(depth, width, dim)
    n_params = sum(p.numel() for p in net.parameters())
    # Untrained measurement first (V4.1 convention)
    fim_un = fim_diagonal(net, dim, n_probes)
    t1u, t3u, ru = tier_ratio(fim_un)
    su = log_stats(fim_un)
    # Train
    train(net, dim, train_steps)
    fim_tr = fim_diagonal(net, dim, n_probes)
    t1t, t3t, rt = tier_ratio(fim_tr)
    st = log_stats(fim_tr)
    return {
        "depth": depth, "seed": seed, "width": width, "dim": dim,
        "n_probes": n_probes, "train_steps": train_steps,
        "n_params": int(n_params),
        "untrained": {
            "tier1_mean": t1u, "tier3_mean": t3u, "tier_ratio": ru,
            "log_mean": su["mean"], "log_var": su["var"],
            "log_skew": su["skew"], "log_excess_kurtosis": su["excess_kurtosis"],
        },
        "trained": {
            "tier1_mean": t1t, "tier3_mean": t3t, "tier_ratio": rt,
            "log_mean": st["mean"], "log_var": st["var"],
            "log_skew": st["skew"], "log_excess_kurtosis": st["excess_kurtosis"],
        },
        "training_dissipation_factor": ru / rt if rt > 0 else float("inf"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n-probes", type=int, default=1000)
    ap.add_argument("--train-steps", type=int, default=10000)
    ap.add_argument("--omp-threads", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v6_2_trained_depth_sweep.json"))
    args = ap.parse_args()

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.omp_threads)

    per_run: list[dict] = []
    per_depth_untr_var: dict[int, list[float]] = {}
    per_depth_tr_var: dict[int, list[float]] = {}
    per_depth_untr_lr: dict[int, list[float]] = {}
    per_depth_tr_lr: dict[int, list[float]] = {}
    per_depth_skew_untr: dict[int, list[float]] = {}
    per_depth_skew_tr: dict[int, list[float]] = {}
    per_depth_kurt_untr: dict[int, list[float]] = {}
    per_depth_kurt_tr: dict[int, list[float]] = {}

    for L in args.depths:
        per_depth_untr_var[L] = []
        per_depth_tr_var[L] = []
        per_depth_untr_lr[L] = []
        per_depth_tr_lr[L] = []
        per_depth_skew_untr[L] = []
        per_depth_skew_tr[L] = []
        per_depth_kurt_untr[L] = []
        per_depth_kurt_tr[L] = []
        for seed in args.seeds:
            t0 = time.time()
            row = run_one(L, seed, args.width, args.dim, args.n_probes, args.train_steps)
            row["elapsed_s"] = time.time() - t0
            per_run.append(row)
            per_depth_untr_var[L].append(row["untrained"]["log_var"])
            per_depth_tr_var[L].append(row["trained"]["log_var"])
            if row["untrained"]["tier_ratio"] > 0:
                per_depth_untr_lr[L].append(math.log(row["untrained"]["tier_ratio"]))
            if row["trained"]["tier_ratio"] > 0:
                per_depth_tr_lr[L].append(math.log(row["trained"]["tier_ratio"]))
            per_depth_skew_untr[L].append(row["untrained"]["log_skew"])
            per_depth_skew_tr[L].append(row["trained"]["log_skew"])
            per_depth_kurt_untr[L].append(row["untrained"]["log_excess_kurtosis"])
            per_depth_kurt_tr[L].append(row["trained"]["log_excess_kurtosis"])
            print(
                f"  L={L:>2}  seed={seed}  N={row['n_params']:>7,}  "
                f"untrained T1/T3={row['untrained']['tier_ratio']:>.3e}  "
                f"trained T1/T3={row['trained']['tier_ratio']:>.3e}  "
                f"dissip={row['training_dissipation_factor']:.1f}x  "
                f"({row['elapsed_s']:.1f}s)",
                flush=True,
            )

    depths = sorted(per_depth_untr_var.keys())
    # Predictions
    mean_untr_var = [float(np.mean(per_depth_untr_var[L])) for L in depths]
    mean_tr_var = [float(np.mean(per_depth_tr_var[L])) for L in depths]
    mean_untr_lr = [float(np.mean(per_depth_untr_lr[L])) if per_depth_untr_lr[L] else float("nan") for L in depths]
    mean_tr_lr = [float(np.mean(per_depth_tr_lr[L])) if per_depth_tr_lr[L] else float("nan") for L in depths]

    untr_var_slope, _, untr_var_r2 = fit_linear([float(L) for L in depths], mean_untr_var)
    tr_var_slope, _, tr_var_r2 = fit_linear([float(L) for L in depths], mean_tr_var)
    untr_lr_slope_sqrt, _, untr_lr_r2 = fit_linear([math.sqrt(L) for L in depths], mean_untr_lr)
    tr_lr_slope_sqrt, _, tr_lr_r2 = fit_linear([math.sqrt(L) for L in depths], mean_tr_lr)

    summary = {
        "depths": depths,
        "untrained_mean_log_var": mean_untr_var,
        "trained_mean_log_var": mean_tr_var,
        "untrained_mean_log_T1T3": mean_untr_lr,
        "trained_mean_log_T1T3": mean_tr_lr,
        "P1_training_reduces_variance_slope": {
            "untrained_slope": untr_var_slope, "untrained_R2": untr_var_r2,
            "trained_slope": tr_var_slope, "trained_R2": tr_var_r2,
            "ratio_slope_untrained_over_trained": untr_var_slope / tr_var_slope if tr_var_slope > 0 else float("inf"),
            "pass": bool(tr_var_slope < untr_var_slope and tr_var_r2 > 0.6),
        },
        "P2_trained_sqrt_L_scaling": {
            "untrained_slope": untr_lr_slope_sqrt, "untrained_R2": untr_lr_r2,
            "trained_slope": tr_lr_slope_sqrt, "trained_R2": tr_lr_r2,
            "pass": bool(tr_lr_r2 > 0.8),
        },
    }

    print("\n==== hypothesis tests ====")
    print(f"P1 trained Var[log F] slope < untrained: trained={tr_var_slope:.3f}  untrained={untr_var_slope:.3f}  "
          f"{'PASS' if summary['P1_training_reduces_variance_slope']['pass'] else 'FAIL'}")
    print(f"P2 trained log(T1/T3) ~ sqrt(L) R²={tr_lr_r2:.3f}  "
          f"{'PASS' if summary['P2_trained_sqrt_L_scaling']['pass'] else 'FAIL'}")

    payload = {"config": vars(args), "per_run": per_run, "summary": summary}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
