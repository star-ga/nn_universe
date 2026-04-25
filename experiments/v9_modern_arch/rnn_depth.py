"""V9.3 — RNN / LSTM depth sweep.

A reviewer concern about reaching architecture_coverage 10/10:
test whether *temporal* sequential composition (RNN / LSTM time-steps)
induces the same FIM hierarchy as *spatial* layered composition (MLP
hidden layers).

Three subtle differences from the depth sweeps in V6:
  - The sequential dimension is unrolled in TIME, not in DEPTH-stack.
  - LSTM gates introduce sigmoid+tanh saturation that bounds Jacobian
    variance accumulation.
  - Vanishing-gradient phenomena could break the log-normal mechanism.

This is a clean test of whether the universality class extends to RNNs.
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


def fit_linear(xs, ys):
    X = np.array(xs, dtype=np.float64); Y = np.array(ys, dtype=np.float64)
    if len(X) < 2:
        return float("nan"), float("nan"), float("nan")
    xm, ym = X.mean(), Y.mean()
    Sxy = ((X - xm) * (Y - ym)).sum(); Sxx = ((X - xm) ** 2).sum()
    if Sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = Sxy / Sxx; intercept = ym - slope * xm
    yp = intercept + slope * X
    ssr = ((Y - yp) ** 2).sum(); sst = ((Y - ym) ** 2).sum()
    return float(slope), float(intercept), float(1 - ssr/sst if sst > 0 else float("nan"))


def tier_ratio(values: np.ndarray):
    s = np.sort(values)[::-1]; n = len(s)
    k1 = max(1, int(n * 0.01)); k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean()); t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1/t3 if t3 > 0 else float("inf"))


# ============================================================
# Vanilla RNN and LSTM modules with controllable time depth
# ============================================================

class VanillaRNN(nn.Module):
    """Single RNN cell unrolled over `seq_len` time steps. Sequential
    composition along the time axis."""
    def __init__(self, dim, hidden, seq_len):
        super().__init__()
        self.cell = nn.RNNCell(dim, hidden)
        self.head = nn.Linear(hidden, dim)
        self.hidden = hidden; self.dim = dim; self.seq_len = seq_len
    def forward(self, x):
        # x: (batch, seq_len, dim)
        h = torch.zeros(x.size(0), self.hidden, device=x.device, dtype=x.dtype)
        for t in range(self.seq_len):
            h = self.cell(x[:, t, :], h)
        return self.head(h)


class VanillaLSTM(nn.Module):
    """Single LSTM cell unrolled. Saturating gates by design."""
    def __init__(self, dim, hidden, seq_len):
        super().__init__()
        self.cell = nn.LSTMCell(dim, hidden)
        self.head = nn.Linear(hidden, dim)
        self.hidden = hidden; self.dim = dim; self.seq_len = seq_len
    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        for t in range(self.seq_len):
            h, c = self.cell(x[:, t, :], (h, c))
        return self.head(h)


def fim_diagonal(net, dim, seq_len, n_probes):
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for _ in range(n_probes):
        x = torch.randn(1, seq_len, dim)
        y = net(x)
        target = x[:, 0, :]  # predict the first time step (arbitrary fixed target)
        loss = 0.5 * (y - target).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data.double() ** 2
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-probes", type=int, default=300)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_3_rnn_lstm_results.json"))
    args = ap.parse_args()

    rows = []
    for arch_name, ArchCls in [("rnn", VanillaRNN), ("lstm", VanillaLSTM)]:
        per_L_lr = {}
        for L in args.seq_lens:
            per_L_lr[L] = []
            for seed in args.seeds:
                torch.manual_seed(seed)
                net = ArchCls(args.dim, args.hidden, L)
                n_params = sum(p.numel() for p in net.parameters())
                t0 = time.time()
                fim = fim_diagonal(net, args.dim, L, args.n_probes)
                t1m, t3m, ratio = tier_ratio(fim)
                dt = time.time() - t0
                rows.append({"arch": arch_name, "seq_len": L, "seed": seed,
                              "n_params": int(n_params), "tier_ratio": ratio,
                              "elapsed_s": dt})
                if ratio > 0:
                    per_L_lr[L].append(math.log(ratio))
                print(f"  {arch_name:<5} seq={L:>2} seed={seed} N={n_params:>5,} "
                      f"T1/T3={ratio:.3e} ({dt:.1f}s)", flush=True)
        depths = sorted(per_L_lr.keys())
        sqrtL = [math.sqrt(L) for L in depths]
        means = [float(np.mean(per_L_lr[L])) if per_L_lr[L] else float("nan") for L in depths]
        slope, _, r2 = fit_linear(sqrtL, means)
        print(f"\n  {arch_name:<5} log(T1/T3) ~ sqrt(seq_len)  slope={slope:.3f}  R²={r2:.3f}\n", flush=True)

    summary = {}
    for arch_name in ["rnn", "lstm"]:
        per_L = {}
        for r in rows:
            if r["arch"] == arch_name and r["tier_ratio"] > 0:
                per_L.setdefault(r["seq_len"], []).append(math.log(r["tier_ratio"]))
        depths = sorted(per_L.keys())
        sqrtL = [math.sqrt(L) for L in depths]
        means = [float(np.mean(per_L[L])) for L in depths]
        slope, _, r2 = fit_linear(sqrtL, means)
        summary[arch_name] = {"depths": depths, "mean_log_T1T3": means,
                               "slope_sqrt_seq_len": slope, "R2": r2,
                               "pass": bool(slope > 0 and r2 > 0.8)}

    payload = {"config": vars(args), "per_run": rows, "summary": summary,
               "interpretation": (
                   "Tests whether temporal sequential composition (RNN time-steps) "
                   "induces the same FIM hierarchy as spatial layered composition. "
                   "The honest expected outcome: vanilla RNN may pass at small seq_len "
                   "and saturate at large seq_len due to vanishing/exploding gradients; "
                   "LSTM with gating is expected to attenuate the mechanism via "
                   "saturating sigmoids. Either way, the result is informative."
               )}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
