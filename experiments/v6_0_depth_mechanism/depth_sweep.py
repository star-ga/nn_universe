"""V6.0 — depth-sweep test of the chain-rule / log-normal mechanism.

Theoretical hypothesis. For an L-layer ReLU MLP with i.i.d. Kaiming-init
weights, the gradient of a scalar loss with respect to any weight θ in
layer l is

    ∂L/∂θ  =  (product of L-l Jacobian factors) × (direct-gradient term)

The Jacobian factor for each downstream layer is a projection through a
random weight matrix composed with a ReLU mask. Taking the squared
magnitude and then the logarithm turns the product into a SUM of
L−l + 1 terms, each with bounded variance. By the Lindeberg CLT, for
large L the log of (∂L/∂θ)² is approximately Gaussian with variance
growing linearly in L:

    Var[log F_ii]  ≈  σ² · L

Heavy-tail in a log-normal distribution implies a tier ratio

    T1/T3  ∼  exp(c · σ · √L)

so T1/T3 is expected to grow **exponentially in √L**. This is the
quantitative prediction that the V4.1 init-induced finding (hierarchy
already at Kaiming init) implies but does not quantify.

This experiment measures, at each depth L ∈ {2, 3, 4, 6, 8, 12, 20},
the untrained-MLP FIM diagonal on the V1.0 self-prediction task
(width=64, dim=16, 5 seeds, 1000 FIM probes). For each (L, seed) we
report:

  - T1/T3 tier ratio (canonical 1% / 50% partition)
  - Var[log F_ii]  (the direct theoretical quantity)
  - Skewness + kurtosis of log F  (normality diagnostic)

Predictions that this experiment falsifies or confirms:

  (H1)  Var[log F_ii]  grows linearly in L.
  (H2)  log-T1/T3      grows as √L  (i.e. log-log slope 0.5).
  (H3)  At fixed L, log F_ii is approximately Gaussian
        (|skew|, |excess kurtosis| < 0.5 for L >= 6).

Confirmation of (H1) + (H2) + (H3) upgrades the "deep-layered-sequential"
empirical signature from pure phenomenology to a concrete mechanism:
product-of-random-Jacobians gives log-normal, log-normal gives heavy tails,
heavy tails give the tier hierarchy. That's the missing theorem behind
V5.0's 10-system dichotomy.
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
import torch.nn as nn


def make_net(depth: int, width: int, dim: int) -> nn.Module:
    """L-layer ReLU MLP: stem + (depth-1) hidden + head. 'depth' = number
    of Linear layers in total. depth=2 gives a single hidden layer."""
    if depth < 2:
        raise ValueError("depth must be >= 2")
    layers: list[nn.Module] = [nn.Linear(dim, width), nn.ReLU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers.append(nn.Linear(width, dim))
    return nn.Sequential(*layers)


def fim_diagonal(net: nn.Module, dim: int, n_probes: int) -> np.ndarray:
    """Empirical FIM diagonal in float64 accumulation."""
    fim = {
        n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()
    }
    net.eval()
    for _ in range(n_probes):
        x = torch.randn(1, dim)
        y = net(x)
        loss = 0.5 * (y - x).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data.double() ** 2
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def tier_ratio(values: np.ndarray, top_pct: float = 1.0, bot_pct: float = 50.0) -> tuple[float, float, float]:
    sorted_desc = np.sort(values)[::-1]
    n = len(sorted_desc)
    k1 = max(1, int(n * top_pct / 100))
    k3 = max(1, int(n * bot_pct / 100))
    t1 = float(sorted_desc[:k1].mean())
    t3 = float(sorted_desc[-k3:].mean())
    if t3 <= 0:
        nz = sorted_desc[sorted_desc > 0]
        t3 = float(nz[-max(len(nz) // 10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1 / t3 if t3 > 0 else float("inf"))


def log_stats(values: np.ndarray) -> dict:
    """Mean, variance, skewness, excess kurtosis of log(F_ii) over nonzero
    entries. Returns NaN for moments that can't be computed."""
    nonzero = values[values > 0]
    if len(nonzero) < 10:
        return {"n": int(len(nonzero)), "mean": float("nan"), "var": float("nan"),
                "skew": float("nan"), "excess_kurtosis": float("nan")}
    lg = np.log(nonzero)
    m = lg.mean()
    var = lg.var()
    sd = math.sqrt(var) if var > 0 else 1.0
    std = (lg - m) / sd
    skew = float((std ** 3).mean())
    excess_kurtosis = float((std ** 4).mean() - 3.0)
    return {"n": int(len(nonzero)), "mean": float(m), "var": float(var),
            "skew": skew, "excess_kurtosis": excess_kurtosis}


def run_one(depth: int, seed: int, width: int, dim: int, n_probes: int) -> dict:
    torch.manual_seed(seed)
    net = make_net(depth, width, dim)
    n_params = sum(p.numel() for p in net.parameters())
    fim = fim_diagonal(net, dim, n_probes)
    t1, t3, ratio = tier_ratio(fim)
    stats = log_stats(fim)
    return {
        "depth": depth, "seed": seed,
        "width": width, "dim": dim, "n_probes": n_probes,
        "n_params": int(n_params),
        "tier1_mean": t1, "tier3_mean": t3, "tier_ratio": ratio,
        "log_mean": stats["mean"], "log_var": stats["var"],
        "log_skew": stats["skew"], "log_excess_kurtosis": stats["excess_kurtosis"],
        "fim_nonzero_count": stats["n"],
    }


def fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """OLS fit y = a + b x; returns (slope, intercept, R²)."""
    X = np.array(xs, dtype=np.float64)
    Y = np.array(ys, dtype=np.float64)
    n = len(X)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    x_mean, y_mean = X.mean(), Y.mean()
    Sxy = ((X - x_mean) * (Y - y_mean)).sum()
    Sxx = ((X - x_mean) ** 2).sum()
    if Sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * X
    ss_res = ((Y - y_pred) ** 2).sum()
    ss_tot = ((Y - y_mean) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n-probes", type=int, default=1000)
    ap.add_argument("--omp-threads", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v6_0_depth_sweep.json"))
    args = ap.parse_args()

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.omp_threads)

    per_run: list[dict] = []
    per_depth: dict[int, list[float]] = {}       # log T1/T3
    per_depth_var: dict[int, list[float]] = {}   # Var[log F]

    for L in args.depths:
        per_depth[L] = []
        per_depth_var[L] = []
        for seed in args.seeds:
            t0 = time.time()
            row = run_one(L, seed, args.width, args.dim, args.n_probes)
            row["elapsed_s"] = time.time() - t0
            per_run.append(row)
            per_depth[L].append(math.log(row["tier_ratio"]) if row["tier_ratio"] > 0 else float("nan"))
            per_depth_var[L].append(row["log_var"])
            print(
                f"  depth={L:>2}  seed={seed}  N={row['n_params']:>7,}  "
                f"T1/T3={row['tier_ratio']:>.4e}  Var[log F]={row['log_var']:.3f}  "
                f"skew={row['log_skew']:+.2f}  kurt={row['log_excess_kurtosis']:+.2f}  "
                f"({row['elapsed_s']:.1f}s)",
                flush=True,
            )

    # Fit hypotheses
    depths = sorted(per_depth.keys())
    mean_log_ratio = [float(np.mean(per_depth[L])) for L in depths]
    mean_log_var = [float(np.mean(per_depth_var[L])) for L in depths]
    sqrt_depths = [math.sqrt(L) for L in depths]
    lin_depths = [float(L) for L in depths]

    # H1: Var[log F] is linear in L  →  fit (L, mean_log_var)
    h1_slope, h1_intercept, h1_r2 = fit_linear(lin_depths, mean_log_var)

    # H2: log(T1/T3) is linear in sqrt(L)  →  fit (sqrt(L), mean_log_ratio)
    h2_slope, h2_intercept, h2_r2 = fit_linear(sqrt_depths, mean_log_ratio)

    # Auxiliary: log(T1/T3) vs L linear fit (log-normal-Gaussian-CDF would be sublinear)
    aux_slope_L, aux_intercept_L, aux_r2_L = fit_linear(lin_depths, mean_log_ratio)

    mean_skew = [float(np.mean([r["log_skew"] for r in per_run if r["depth"] == L])) for L in depths]
    mean_kurt = [float(np.mean([r["log_excess_kurtosis"] for r in per_run if r["depth"] == L])) for L in depths]

    summary = {
        "depths": depths,
        "mean_log_ratio": mean_log_ratio,
        "mean_log_var": mean_log_var,
        "mean_log_skew": mean_skew,
        "mean_log_excess_kurtosis": mean_kurt,
        "H1_Var_logF_linear_in_L": {
            "slope": h1_slope, "intercept": h1_intercept, "R2": h1_r2,
            "prediction": "slope > 0 AND R^2 > 0.9",
            "pass": bool(h1_r2 is not float("nan") and h1_r2 > 0.9 and h1_slope > 0),
        },
        "H2_log_ratio_linear_in_sqrt_L": {
            "slope": h2_slope, "intercept": h2_intercept, "R2": h2_r2,
            "prediction": "R^2 > 0.9",
            "pass": bool(h2_r2 is not float("nan") and h2_r2 > 0.9),
        },
        "aux_log_ratio_linear_in_L": {
            "slope": aux_slope_L, "intercept": aux_intercept_L, "R2": aux_r2_L,
        },
    }

    print("\n==== hypothesis tests ====")
    print(f"H1 Var[log F_ii] ~ L          slope={h1_slope:.4f}  R²={h1_r2:.4f}  "
          f"{'PASS' if summary['H1_Var_logF_linear_in_L']['pass'] else 'FAIL'}")
    print(f"H2 log(T1/T3)   ~ sqrt(L)     slope={h2_slope:.4f}  R²={h2_r2:.4f}  "
          f"{'PASS' if summary['H2_log_ratio_linear_in_sqrt_L']['pass'] else 'FAIL'}")
    print(f"aux log(T1/T3) ~ L            slope={aux_slope_L:.4f}  R²={aux_r2_L:.4f}")
    print(f"mean |skew| across depths     {float(np.mean(np.abs(mean_skew))):.3f}")
    print(f"mean |excess kurtosis|        {float(np.mean(np.abs(mean_kurt))):.3f}")

    payload = {
        "config": vars(args),
        "per_run": per_run,
        "summary": summary,
        "interpretation": (
            "If H1 (Var[log F] ~ L, slope>0, R²>0.9) and H2 (log T1/T3 ~ √L, R²>0.9) both pass,"
            " the chain-rule / log-normal hypothesis for the FIM tier hierarchy is confirmed."
            " This upgrades V5.0's empirical dichotomy to a mechanistic theorem: depth -> product"
            " of random Jacobians -> log-normal FIM_ii -> heavy-tailed tier ratio."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
