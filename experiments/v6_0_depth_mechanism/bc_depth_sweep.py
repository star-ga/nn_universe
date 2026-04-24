"""V6.3 — layered Boolean-circuit depth sweep.

Builds a STRICTLY LAYERED random boolean circuit:
  layer 0 = input bits (n_inputs)
  layer k = `gates_per_layer` gates, each drawing two wires from layer k-1 only
  output = softmax-mixture of final-layer gates (scalar)

Each gate is a softmax mixture of {AND, OR, XOR} with logits w ∈ ℝ³
(exactly the baselines.py BooleanCircuit soft-gate convention), so the
output is a smooth function of all gate logits and we can compute the
FIM diagonal via finite differences on the gate logits.

Sweep depths L ∈ {2, 4, 8, 16, 32} with gates_per_layer = 16
(small enough to stay fast), 3 seeds each, 200 probe inputs.

Predictions:
  (B1)  Var[log F_ii] is approximately linear in L (same Hanin-Nica-style
        scaling as MLPs, because the gate logit's gradient passes through
        L-k layers' softmax-weighted mixture Jacobians).
  (B2)  log(T1/T3) is linear in sqrt(L).

If B1+B2 hold, the log-normal / PoRM mechanism is substrate-independent
within the deep-sequential class — applies to neurons with real weights
AND to boolean gates with softmax logits. That's the maximum-strength
universality claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


class LayeredBooleanCircuit:
    """Strictly layered random boolean circuit with softmax-mixture gates."""

    def __init__(self, depth: int, gates_per_layer: int, n_inputs: int, seed: int) -> None:
        if depth < 2:
            raise ValueError("depth >= 2")
        self.depth = depth
        self.gates_per_layer = gates_per_layer
        self.n_inputs = n_inputs
        self.rng = np.random.default_rng(seed)
        # Each gate has 3 softmax logits → n_params = depth * gates_per_layer * 3
        self.W = 0.3 * self.rng.standard_normal((depth, gates_per_layer, 3)).astype(np.float64)
        # Wiring: for each layer k and gate g, two wires point into layer k-1.
        # Layer 0 wires point into input bits (indices 0..n_inputs-1).
        self.wires = np.zeros((depth, gates_per_layer, 2), dtype=np.int32)
        for k in range(depth):
            in_size = n_inputs if k == 0 else gates_per_layer
            for g in range(gates_per_layer):
                self.wires[k, g, 0] = self.rng.integers(0, in_size)
                self.wires[k, g, 1] = self.rng.integers(0, in_size)
        # Output = softmax mixture of final-layer gates (fixed readout — no new params).
        # To avoid adding more params, we take a fixed uniform average of the final layer.

    def n_params(self) -> int:
        return self.depth * self.gates_per_layer * 3

    def _forward_batch(self, inputs: np.ndarray, W: np.ndarray) -> np.ndarray:
        """inputs: (B, n_inputs) in [0,1]. W: (depth, gpl, 3). Returns (B,) scalar output."""
        B = inputs.shape[0]
        prev = inputs  # (B, n_inputs)
        for k in range(self.depth):
            curr = np.zeros((B, self.gates_per_layer), dtype=np.float64)
            for g in range(self.gates_per_layer):
                a = prev[:, self.wires[k, g, 0]]
                b = prev[:, self.wires[k, g, 1]]
                logits = W[k, g]
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                andv = a * b
                orv = a + b - a * b
                xorv = a + b - 2 * a * b
                curr[:, g] = probs[0] * andv + probs[1] * orv + probs[2] * xorv
            prev = curr
        return prev.mean(axis=1)  # (B,) uniform average of final layer

    def fim_diagonal(self, n_probes: int) -> np.ndarray:
        """Finite-difference FIM diagonal on the gate logits."""
        eps = 1e-3
        xs = self.rng.uniform(0.0, 1.0, size=(n_probes, self.n_inputs)).astype(np.float64)
        base_y = self._forward_batch(xs, self.W)  # (B,)
        base_loss_grad = base_y  # "loss" = y itself → gradient of 0.5 y² is y (scalar regression)
        fim = np.zeros_like(self.W)
        for k in range(self.depth):
            for g in range(self.gates_per_layer):
                for i in range(3):
                    W_pert = self.W.copy()
                    W_pert[k, g, i] += eps
                    y_pert = self._forward_batch(xs, W_pert)  # (B,)
                    dy = (y_pert - base_y) / eps  # (B,) ~ ∂y/∂θ
                    # per-sample gradient of 0.5 y² wrt θ = y · ∂y/∂θ
                    grad = base_loss_grad * dy
                    fim[k, g, i] = float(np.mean(grad * grad))
        return fim.flatten()


def tier_ratio(values: np.ndarray, top_pct: float = 1.0, bot_pct: float = 50.0) -> tuple[float, float, float]:
    sorted_desc = np.sort(values)[::-1]
    n = len(sorted_desc)
    k1 = max(1, int(n * top_pct / 100))
    k3 = max(1, int(n * bot_pct / 100))
    t1 = float(sorted_desc[:k1].mean())
    t3 = float(sorted_desc[-k3:].mean())
    if t3 <= 0:
        nz = sorted_desc[sorted_desc > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1/t3 if t3 > 0 else float("inf"))


def log_stats(values: np.ndarray) -> dict:
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
    excess = float((std ** 4).mean() - 3.0)
    return {"n": int(len(nonzero)), "mean": float(m), "var": float(var),
            "skew": skew, "excess_kurtosis": excess}


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
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gates-per-layer", type=int, default=16)
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v6_3_bc_depth_sweep.json"))
    args = ap.parse_args()

    per_run: list[dict] = []
    per_L_lr: dict[int, list[float]] = {}
    per_L_var: dict[int, list[float]] = {}

    for L in args.depths:
        per_L_lr[L] = []
        per_L_var[L] = []
        for seed in args.seeds:
            t0 = time.time()
            bc = LayeredBooleanCircuit(L, args.gates_per_layer, args.n_inputs, seed)
            fim = bc.fim_diagonal(args.n_probes)
            t1, t3, ratio = tier_ratio(fim)
            stats = log_stats(fim)
            dt = time.time() - t0
            row = {
                "depth": L, "seed": seed,
                "gates_per_layer": args.gates_per_layer, "n_inputs": args.n_inputs,
                "n_probes": args.n_probes, "n_params": int(bc.n_params()),
                "tier1_mean": t1, "tier3_mean": t3, "tier_ratio": ratio,
                "log_mean": stats["mean"], "log_var": stats["var"],
                "log_skew": stats["skew"], "log_excess_kurtosis": stats["excess_kurtosis"],
                "elapsed_s": dt,
            }
            per_run.append(row)
            if ratio > 0:
                per_L_lr[L].append(math.log(ratio))
            per_L_var[L].append(stats["var"])
            print(
                f"  L={L:>2}  seed={seed}  N={row['n_params']:>4}  "
                f"T1/T3={ratio:>.3e}  Var[log F]={stats['var']:.3f}  "
                f"skew={stats['skew']:+.2f}  kurt={stats['excess_kurtosis']:+.2f}  "
                f"({dt:.1f}s)",
                flush=True,
            )

    depths = sorted(per_L_lr.keys())
    mean_lr = [float(np.mean(per_L_lr[L])) if per_L_lr[L] else float("nan") for L in depths]
    mean_var = [float(np.mean(per_L_var[L])) for L in depths]

    b1_slope, _, b1_r2 = fit_linear([float(L) for L in depths], mean_var)
    b2_slope, _, b2_r2 = fit_linear([math.sqrt(L) for L in depths], mean_lr)

    summary = {
        "depths": depths,
        "mean_log_var": mean_var,
        "mean_log_T1T3": mean_lr,
        "B1_var_linear_in_L": {"slope": b1_slope, "R2": b1_r2,
                                "pass": bool(b1_slope > 0 and b1_r2 > 0.8)},
        "B2_logT1T3_linear_in_sqrt_L": {"slope": b2_slope, "R2": b2_r2,
                                         "pass": bool(b2_r2 > 0.8)},
    }

    print("\n==== BC depth hypothesis tests ====")
    print(f"B1 Var[log F_ii] ~ L         slope={b1_slope:.4f}  R²={b1_r2:.4f}  {'PASS' if summary['B1_var_linear_in_L']['pass'] else 'FAIL'}")
    print(f"B2 log(T1/T3)   ~ sqrt(L)   slope={b2_slope:.4f}  R²={b2_r2:.4f}  {'PASS' if summary['B2_logT1T3_linear_in_sqrt_L']['pass'] else 'FAIL'}")

    payload = {"config": vars(args), "per_run": per_run, "summary": summary,
               "interpretation": (
                   "If B1 and B2 both pass for boolean circuits, the Hanin-Nica log-normal "
                   "mechanism is substrate-independent within the deep-sequential class. "
                   "That's the strongest possible universality claim: the signature tracks "
                   "'deep layered sequential computation' as a computational primitive, "
                   "independent of whether the substrate uses neurons, weights, gradients, "
                   "or probabilities."
               )}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
