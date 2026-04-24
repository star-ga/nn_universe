"""V6.4 — transformer depth sweep.

Tests whether log(T1/T3) ∝ √L holds for ATTENTION-composed Jacobians.

Attention differs from MLP in two ways:
  (a) The Jacobian at each block passes through a SOFTMAX over the
      token-token similarity matrix, producing correlated gradient
      contributions across tokens rather than independent ones.
  (b) There is a residual stream: each block's input has contributions
      from all upstream blocks directly (pre-norm addition).

Both (a) and (b) could break the i.i.d. assumption of Hanin-Nica. If the
√L scaling holds anyway, the universality class of the mechanism is
even broader than V6.0/V6.3 establish.

Architecture: depth L transformer blocks, each with:
  - LayerNorm
  - Multi-head self-attention (d_model, n_heads=4)
  - Residual add
  - LayerNorm
  - MLP (d_model -> 4*d_model -> d_model)
  - Residual add

Task: self-prediction of Gaussian token sequences (seq_len=8, d_model=32).
FIM diagonal computed same as V6.0 — per-sample squared gradient
averaged over 500 probes (fewer than V6.0's 1000 because transformer
forward pass is slower).

Predictions:
  (T1)  Var[log F_ii] ∝ L  with R² > 0.8.
  (T2)  log(T1/T3) ∝ √L  with R² > 0.8.

If both pass, the Hanin-Nica mechanism extends to transformers.
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
import torch.nn.functional as F

from depth_sweep import fim_diagonal, tier_ratio, log_stats, fit_linear  # noqa: E402


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, depth: int, d_model: int, seq_len: int, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(depth)])
        self.unembed = nn.Linear(d_model, d_model)
        self.depth = depth
        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.unembed(x)


def fim_diagonal_transformer(net: Transformer, seq_len: int, d_model: int, n_probes: int) -> np.ndarray:
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for _ in range(n_probes):
        x = torch.randn(1, seq_len, d_model)
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


def run_one(depth: int, seed: int, d_model: int, seq_len: int, n_probes: int) -> dict:
    torch.manual_seed(seed)
    net = Transformer(depth, d_model, seq_len)
    n_params = sum(p.numel() for p in net.parameters())
    fim = fim_diagonal_transformer(net, seq_len, d_model, n_probes)
    t1, t3, ratio = tier_ratio(fim)
    stats = log_stats(fim)
    return {
        "depth": depth, "seed": seed, "d_model": d_model, "seq_len": seq_len,
        "n_probes": n_probes, "n_params": int(n_params),
        "tier1_mean": t1, "tier3_mean": t3, "tier_ratio": ratio,
        "log_var": stats["var"], "log_skew": stats["skew"],
        "log_excess_kurtosis": stats["excess_kurtosis"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--n-probes", type=int, default=500)
    ap.add_argument("--omp-threads", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v6_4_transformer_depth_sweep.json"))
    args = ap.parse_args()

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.omp_threads)

    per_run: list[dict] = []
    per_L_lr: dict[int, list[float]] = {}
    per_L_var: dict[int, list[float]] = {}

    for L in args.depths:
        per_L_lr[L] = []
        per_L_var[L] = []
        for seed in args.seeds:
            t0 = time.time()
            row = run_one(L, seed, args.d_model, args.seq_len, args.n_probes)
            row["elapsed_s"] = time.time() - t0
            per_run.append(row)
            if row["tier_ratio"] > 0:
                per_L_lr[L].append(math.log(row["tier_ratio"]))
            per_L_var[L].append(row["log_var"])
            print(
                f"  L={L:>2}  seed={seed}  N={row['n_params']:>7,}  "
                f"T1/T3={row['tier_ratio']:>.3e}  Var[log F]={row['log_var']:.3f}  "
                f"skew={row['log_skew']:+.2f}  ({row['elapsed_s']:.1f}s)",
                flush=True,
            )

    depths = sorted(per_L_lr.keys())
    mean_lr = [float(np.mean(per_L_lr[L])) if per_L_lr[L] else float("nan") for L in depths]
    mean_var = [float(np.mean(per_L_var[L])) for L in depths]

    t1_slope, _, t1_r2 = fit_linear([float(L) for L in depths], mean_var)
    t2_slope, _, t2_r2 = fit_linear([math.sqrt(L) for L in depths], mean_lr)

    summary = {
        "depths": depths,
        "mean_log_var": mean_var,
        "mean_log_T1T3": mean_lr,
        "T1_var_linear_in_L": {"slope": t1_slope, "R2": t1_r2,
                                "pass": bool(t1_slope > 0 and t1_r2 > 0.8)},
        "T2_logT1T3_linear_in_sqrt_L": {"slope": t2_slope, "R2": t2_r2,
                                         "pass": bool(t2_r2 > 0.8)},
    }

    print("\n==== transformer hypothesis tests ====")
    print(f"T1 Var[log F_ii] ~ L      slope={t1_slope:.3f}  R²={t1_r2:.3f}  {'PASS' if summary['T1_var_linear_in_L']['pass'] else 'FAIL'}")
    print(f"T2 log(T1/T3)   ~ sqrt(L) slope={t2_slope:.3f}  R²={t2_r2:.3f}  {'PASS' if summary['T2_logT1T3_linear_in_sqrt_L']['pass'] else 'FAIL'}")

    payload = {"config": vars(args), "per_run": per_run, "summary": summary,
               "interpretation": (
                   "If T1 and T2 both pass, the Hanin-Nica mechanism extends to "
                   "attention-composed architectures. The universality class then "
                   "includes MLPs (V6.0), boolean circuits (V6.3), AND transformers."
               )}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
