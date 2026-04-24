"""V8.0 — Binary-tree tensor-network (BTTN) depth sweep.

A MERA / holographic-code-inspired architecture: the tensor network is
a balanced binary tree that maps $2^L$ leaf inputs to a single root
output. Each internal node is a parameterised tensor (here: a small
dense $2 \\times 2 \\to 1$ map with a soft-threshold nonlinearity).
The network is **explicitly layered** by construction (Vidal 2007,
Swingle 2012, Pastawski-Yoshida-Harlow-Preskill 2015) and has depth
exactly $L$ between leaves and root.

Cosmological motivation: tensor-network constructions like MERA +
HaPPY codes are the canonical toy models for emergent holographic
spacetime. If tensor networks exhibit the FIM tier hierarchy via the
same Hanin-Nica-style log-normal mechanism as V6.0 MLPs, this
closes the empirical loop between the paper's abstract "deep layered
sequential computation" class and the physics literature's tensor-
network cosmology constructions.

Prediction: $\\log(T_1/T_3) \\propto \\sqrt{L}$ should hold for BTTNs
with $R^2 > 0.8$, with a prefactor determined by the per-node
Jacobian variance of the soft-threshold nonlinearity.
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


class BinaryTreeTensorNetwork(nn.Module):
    """Balanced binary tree: 2^depth leaves → 1 scalar output.

    Each internal node at height k (counting from leaves, k=0) applies a
    parameterised map from two height-(k-1) values to one height-k value:
        out = tanh(W @ [a, b] + b0)
    where W is a learnable 1x2 matrix per node and b0 a scalar bias.
    The tree has 2^depth - 1 internal nodes → ~2^depth parameters.

    The depth of the composition chain from any leaf to the output is
    exactly `depth`, giving a clean tunable-depth substrate.
    """

    def __init__(self, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.n_leaves = 2 ** depth
        # Store per-node parameters layer by layer.
        # Layer k has 2^(depth-k) nodes, k = 1..depth.
        self.W_layers = nn.ParameterList()
        self.b_layers = nn.ParameterList()
        for k in range(1, depth + 1):
            n_nodes = 2 ** (depth - k)
            # Each node: 1x2 weight, scalar bias.
            # Kaiming-like scaled init (variance 2/fan_in with fan_in=2).
            W = torch.randn(n_nodes, 1, 2) * math.sqrt(2.0 / 2.0)
            b = torch.zeros(n_nodes, 1)
            self.W_layers.append(nn.Parameter(W))
            self.b_layers.append(nn.Parameter(b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, 2^depth). Returns (batch, 1)."""
        # Current layer values: (batch, n_current_nodes)
        cur = x
        for k in range(self.depth):
            W = self.W_layers[k]  # (n_nodes, 1, 2)
            b = self.b_layers[k]  # (n_nodes, 1)
            n_nodes = W.shape[0]
            # Reshape cur to pair up: (batch, n_nodes, 2)
            paired = cur.view(cur.shape[0], n_nodes, 2)
            # Apply W per node: (batch, n_nodes, 1)
            # einsum: sum over last axis of paired and last axis of W
            out = torch.einsum("bno,noi->bni", paired, W.transpose(1, 2)) + b.unsqueeze(0)
            out = torch.tanh(out)
            # Flatten back to (batch, n_nodes)
            cur = out.squeeze(-1)
        return cur  # (batch, 1)


def fim_diagonal_bttn(net: BinaryTreeTensorNetwork, n_probes: int) -> np.ndarray:
    fim_parts = {n: torch.zeros_like(p, dtype=torch.float64)
                 for n, p in net.named_parameters()}
    net.eval()
    for _ in range(n_probes):
        x = torch.randn(1, net.n_leaves)
        y = net(x)
        # Define a "loss" that has non-trivial gradient structure:
        # 0.5 * y^2 is convex in y, gives grad = y per sample.
        loss = 0.5 * y.pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim_parts[n] += p.grad.data.double() ** 2
    for n in fim_parts:
        fim_parts[n] /= n_probes
    return torch.cat([v.flatten() for v in fim_parts.values()]).cpu().numpy()


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
    nonzero = values[values > 0]
    if len(nonzero) < 10:
        return {"n": int(len(nonzero)), "var": float("nan"),
                "skew": float("nan"), "excess_kurtosis": float("nan")}
    lg = np.log(nonzero)
    m = lg.mean()
    var = lg.var()
    sd = math.sqrt(var) if var > 0 else 1.0
    std = (lg - m) / sd
    return {"n": int(len(nonzero)), "var": float(var),
            "skew": float((std ** 3).mean()),
            "excess_kurtosis": float((std ** 4).mean() - 3.0)}


def fit_linear(xs, ys) -> tuple[float, float, float]:
    X = np.array(xs, dtype=np.float64); Y = np.array(ys, dtype=np.float64)
    n = len(X)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    xm, ym = X.mean(), Y.mean()
    Sxy = ((X - xm) * (Y - ym)).sum(); Sxx = ((X - xm) ** 2).sum()
    if Sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = Sxy / Sxx; intercept = ym - slope * xm
    yp = intercept + slope * X
    ssr = ((Y - yp) ** 2).sum(); sst = ((Y - ym) ** 2).sum()
    return float(slope), float(intercept), float(1 - ssr/sst if sst > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-probes", type=int, default=500)
    ap.add_argument("--omp-threads", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v8_0_bttn_results.json"))
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
            torch.manual_seed(seed)
            net = BinaryTreeTensorNetwork(L)
            n_params = sum(p.numel() for p in net.parameters())
            t0 = time.time()
            fim = fim_diagonal_bttn(net, args.n_probes)
            t1m, t3m, ratio = tier_ratio(fim)
            stats = log_stats(fim)
            dt = time.time() - t0
            row = {
                "depth": L, "seed": seed, "n_leaves": 2 ** L,
                "n_probes": args.n_probes, "n_params": int(n_params),
                "tier1_mean": t1m, "tier3_mean": t3m, "tier_ratio": ratio,
                "log_var": stats["var"], "log_skew": stats["skew"],
                "log_excess_kurtosis": stats["excess_kurtosis"],
                "elapsed_s": dt,
            }
            per_run.append(row)
            if ratio > 0:
                per_L_lr[L].append(math.log(ratio))
            per_L_var[L].append(stats["var"])
            print(
                f"  L={L:>2}  seed={seed}  leaves={2**L:>4}  N={n_params:>5}  "
                f"T1/T3={ratio:>.3e}  Var[log F]={stats['var']:.3f}  "
                f"skew={stats['skew']:+.2f}  kurt={stats['excess_kurtosis']:+.2f}  "
                f"({dt:.1f}s)",
                flush=True,
            )

    depths = sorted(per_L_lr.keys())
    mean_lr = [float(np.mean(per_L_lr[L])) if per_L_lr[L] else float("nan") for L in depths]
    mean_var = [float(np.mean(per_L_var[L])) for L in depths]

    slope_var, _, r2_var = fit_linear([float(L) for L in depths], mean_var)
    slope_lr, _, r2_lr = fit_linear([math.sqrt(L) for L in depths], mean_lr)

    summary = {
        "depths": depths,
        "mean_log_var": mean_var,
        "mean_log_T1T3": mean_lr,
        "T1_var_linear_in_L": {"slope": slope_var, "R2": r2_var,
                                "pass": bool(slope_var > 0 and r2_var > 0.8)},
        "T2_logT1T3_linear_in_sqrt_L": {"slope": slope_lr, "R2": r2_lr,
                                          "pass": bool(r2_lr > 0.8)},
    }

    print("\n==== BTTN hypothesis tests ====")
    print(f"T1 Var[log F] ~ L         slope={slope_var:.3f}  R²={r2_var:.3f}  "
          f"{'PASS' if summary['T1_var_linear_in_L']['pass'] else 'FAIL'}")
    print(f"T2 log(T1/T3) ~ sqrt(L)   slope={slope_lr:.3f}  R²={r2_lr:.3f}  "
          f"{'PASS' if summary['T2_logT1T3_linear_in_sqrt_L']['pass'] else 'FAIL'}")

    payload = {
        "config": vars(args),
        "per_run": per_run,
        "summary": summary,
        "interpretation": (
            "If T1 and T2 both pass, the Hanin-Nica log-normal mechanism "
            "extends to balanced binary tensor networks — closing the "
            "empirical bridge between the 'deep layered sequential computation' "
            "universality class identified in the paper (V5.0 + V6.0) and "
            "the physics literature's tensor-network/holographic-code cosmology "
            "constructions (Swingle 2012, Pastawski-Yoshida-Harlow-Preskill 2015). "
            "The FIM tier hierarchy is then a predicted structural consequence "
            "of any MERA-like emergent-spacetime model."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
