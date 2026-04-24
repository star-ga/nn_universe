"""V6.5 — activation-function depth sweep.

Hanin & Nica's log-normal theorem has an activation-dependent variance
coefficient σ². V6.0 confirmed the √L scaling for ReLU. V6.4 confirmed
it for GELU inside transformer blocks. V6.5 tests three activations in
the same vanilla-MLP setup as V6.0, to separate the "√L scaling works"
claim from the "σ² is activation-dependent" claim:

  - ReLU  : the V6.0 baseline (expected slope 11.5).
  - GELU  : smooth, near-ReLU. Expected slope ≈ ReLU's.
  - tanh  : saturates, smaller Jacobian variance. Expected smaller slope.
  - Swish : near-GELU.

For each activation × depth × seed: train untrained MLP, measure FIM,
fit slope of log(T1/T3) vs sqrt(L). The prediction is that all four
activations produce √L scaling but with different prefactors σ.
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

from depth_sweep import fim_diagonal, tier_ratio, log_stats, fit_linear  # noqa: E402


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "swish": nn.SiLU,  # SiLU ≡ Swish
}


def make_net(depth: int, width: int, dim: int, act_name: str) -> nn.Module:
    if depth < 2:
        raise ValueError("depth must be >= 2")
    act_cls = ACTIVATIONS[act_name]
    layers: list[nn.Module] = [nn.Linear(dim, width), act_cls()]
    for _ in range(depth - 2):
        layers += [nn.Linear(width, width), act_cls()]
    layers.append(nn.Linear(width, dim))
    return nn.Sequential(*layers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", type=str, nargs="+",
                    default=["relu", "gelu", "tanh", "swish"])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8, 12])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n-probes", type=int, default=500)
    ap.add_argument("--omp-threads", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v6_5_activation_sweep.json"))
    args = ap.parse_args()

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.omp_threads)

    per_run: list[dict] = []
    summaries: dict[str, dict] = {}

    for act in args.activations:
        per_depth_lr: dict[int, list[float]] = {}
        per_depth_var: dict[int, list[float]] = {}
        for L in args.depths:
            per_depth_lr[L] = []
            per_depth_var[L] = []
            for seed in args.seeds:
                torch.manual_seed(seed)
                net = make_net(L, args.width, args.dim, act)
                n_params = sum(p.numel() for p in net.parameters())
                t0 = time.time()
                fim = fim_diagonal(net, args.dim, args.n_probes)
                t1m, t3m, ratio = tier_ratio(fim)
                stats = log_stats(fim)
                dt = time.time() - t0
                row = {
                    "activation": act, "depth": L, "seed": seed,
                    "width": args.width, "dim": args.dim, "n_probes": args.n_probes,
                    "n_params": int(n_params),
                    "tier1_mean": t1m, "tier3_mean": t3m, "tier_ratio": ratio,
                    "log_var": stats["var"], "log_skew": stats["skew"],
                    "log_excess_kurtosis": stats["excess_kurtosis"],
                    "elapsed_s": dt,
                }
                per_run.append(row)
                if ratio > 0:
                    per_depth_lr[L].append(math.log(ratio))
                per_depth_var[L].append(stats["var"])
                print(
                    f"  {act:<5} L={L:>2}  seed={seed}  "
                    f"T1/T3={ratio:>.3e}  Var[log F]={stats['var']:.3f}  "
                    f"({dt:.1f}s)",
                    flush=True,
                )

        depths = sorted(per_depth_lr.keys())
        mean_lr = [float(np.mean(per_depth_lr[L])) if per_depth_lr[L] else float("nan")
                   for L in depths]
        mean_var = [float(np.mean(per_depth_var[L])) for L in depths]
        slope_var_L, _, r2_var_L = fit_linear([float(L) for L in depths], mean_var)
        slope_lr_sqrt, _, r2_lr_sqrt = fit_linear([math.sqrt(L) for L in depths], mean_lr)
        summaries[act] = {
            "depths": depths,
            "mean_log_var": mean_var,
            "mean_log_T1T3": mean_lr,
            "Var_logF_vs_L": {"slope": slope_var_L, "R2": r2_var_L,
                               "pass": bool(slope_var_L > 0 and r2_var_L > 0.8)},
            "logT1T3_vs_sqrtL": {"slope": slope_lr_sqrt, "R2": r2_lr_sqrt,
                                  "pass": bool(r2_lr_sqrt > 0.8)},
        }
        print(
            f"\n  {act:<5}  Var~L slope={slope_var_L:.2f} R²={r2_var_L:.3f}  "
            f"logT1T3~√L slope={slope_lr_sqrt:.2f} R²={r2_lr_sqrt:.3f}",
            flush=True,
        )

    print("\n==== cross-activation summary ====")
    for act, s in summaries.items():
        print(
            f"  {act:<5}  Var~L slope={s['Var_logF_vs_L']['slope']:>6.2f}  "
            f"R²={s['Var_logF_vs_L']['R2']:.3f}  "
            f"|  logT1T3~√L slope={s['logT1T3_vs_sqrtL']['slope']:>6.2f}  "
            f"R²={s['logT1T3_vs_sqrtL']['R2']:.3f}"
        )

    payload = {
        "config": vars(args),
        "per_run": per_run,
        "per_activation": summaries,
        "interpretation": (
            "If all activations give R²>0.8 on log(T1/T3) ~ sqrt(L), the Hanin-Nica "
            "mechanism is confirmed substrate-agnostic with respect to the activation "
            "choice, and only the σ prefactor depends on the nonlinearity. The slope "
            "ordering (slope_relu ≈ slope_gelu ≈ slope_swish > slope_tanh) should "
            "reflect the per-layer log-Jacobian-variance ordering."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
