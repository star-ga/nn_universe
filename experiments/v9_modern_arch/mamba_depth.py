"""V9.4 — Mamba-style state-space-model (SSM) depth sweep.

OUT-OF-SAMPLE PREDICTION TEST. Registered before measurement:

  H1 (positive): Mamba-style SSM with DISTINCT per-layer parameters
    follows the Hanin-Nica √L scaling law, R² > 0.85, slope > 0.
    Reasoning: SSM is depth-stacked (layers have independent weights);
    its selective scan adds gating but the stack is structurally
    layer-by-layer composition, like ResNet/Transformer-vanilla.

  H2 (alternate): If Mamba's selective gating (similar to attention's
    softmax saturation) attenuates Var[log F] accumulation, the slope
    would be smaller than ResNet's (16.74) but still positive.

  H3 (null): If something structurally distinct from layer-stack is
    happening, slope could be flat or negative — same family as
    GPT-Tiny attention (V9.1).

This is the cleanest "no post-hoc narrowing" test: prediction registered
before result, fresh substrate not in V2 panel.

Architecture: simplified Mamba-block stack — 1D conv + state-space scan
+ output projection per block, distinct weights per block.
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
# Simplified Mamba-style block: 1D conv + selective-scan + projection
# Each block has DISTINCT parameters — depth-stacked.
# ============================================================

class MambaBlock(nn.Module):
    """Simplified Mamba block: input projection → conv1d → SiLU →
    selective-scan SSM → output projection. Each block has its own
    distinct parameters (no weight tying across blocks)."""
    def __init__(self, dim, d_state=16, d_conv=4):
        super().__init__()
        self.dim = dim; self.d_state = d_state; self.d_conv = d_conv
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, d_conv, padding=d_conv-1, groups=dim, bias=True)
        # SSM parameters (selective)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(dim, 1)))
        self.D = nn.Parameter(torch.ones(dim))
        # Selective projections
        self.x_proj = nn.Linear(dim, d_state * 2, bias=False)  # B and C selective
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: (B, T, dim)
        B, T, D = x.shape
        xz = self.in_proj(x)               # (B, T, 2*dim)
        x_in, z = xz.chunk(2, dim=-1)      # (B, T, dim), (B, T, dim)
        # 1D conv on time axis
        x_in = x_in.transpose(1, 2)         # (B, dim, T)
        x_in = self.conv(x_in)[:, :, :T]    # (B, dim, T) — keep length T
        x_in = x_in.transpose(1, 2)         # (B, T, dim)
        x_in = F.silu(x_in)
        # Selective SSM
        A = -torch.exp(self.A_log.float())  # (dim, d_state) negative for stability
        BC = self.x_proj(x_in)              # (B, T, 2*d_state)
        Bm, Cm = BC.chunk(2, dim=-1)        # (B, T, d_state) each
        dt = F.softplus(self.dt_proj(x_in)) # (B, T, dim)
        # Discretized SSM step: simplified parallel scan via cumprod approximation
        # h_t = exp(dt*A) h_{t-1} + dt*B*x_t  (per-channel)
        # Output y_t = C h_t + D x_t
        # For tractability with full backprop on RTX 3080, do explicit recurrence:
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t]  # (B, dim)
            A_eff = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, dim, d_state)
            B_eff = dt_t.unsqueeze(-1) * Bm[:, t].unsqueeze(1)      # (B, dim, d_state) (broadcast)
            h = A_eff * h + B_eff * x_in[:, t].unsqueeze(-1)        # (B, dim, d_state)
            y_t = (h * Cm[:, t].unsqueeze(1)).sum(-1) + self.D * x_in[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)          # (B, T, dim)
        # Gate by z and project out
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaStack(nn.Module):
    def __init__(self, depth, dim, d_state=16, seq_len=8):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state=d_state) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)
        self.seq_len = seq_len; self.dim = dim
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(self.norm(h.mean(dim=1)))


def fim_diagonal(net, dim, seq_len, n_probes):
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for _ in range(n_probes):
        x = torch.randn(1, seq_len, dim)
        y = net(x)
        target = x[:, 0, :]
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
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_4_mamba_results.json"))
    args = ap.parse_args()

    rows = []; per_L_lr = {}
    for L in args.depths:
        per_L_lr[L] = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            net = MambaStack(L, args.dim, args.d_state, args.seq_len)
            n_params = sum(p.numel() for p in net.parameters())
            t0 = time.time()
            fim = fim_diagonal(net, args.dim, args.seq_len, args.n_probes)
            t1m, t3m, ratio = tier_ratio(fim)
            dt = time.time() - t0
            rows.append({"depth": L, "seed": seed, "n_params": int(n_params),
                          "tier_ratio": ratio, "elapsed_s": dt})
            if ratio > 0:
                per_L_lr[L].append(math.log(ratio))
            print(f"  Mamba L={L:>2} seed={seed} N={n_params:>6,} T1/T3={ratio:.3e} ({dt:.1f}s)", flush=True)

    depths = sorted(per_L_lr.keys())
    sqrtL = [math.sqrt(L) for L in depths]
    means = [float(np.mean(per_L_lr[L])) if per_L_lr[L] else float("nan") for L in depths]
    slope, _, r2 = fit_linear(sqrtL, means)

    if slope > 0 and r2 > 0.85:
        verdict = "H1 PASS — Mamba follows √L (positive direction)"
    elif slope > 0 and r2 > 0.5:
        verdict = "H2 PARTIAL — positive slope but lower R²; selective gating attenuates"
    elif abs(slope) < 0.5 and r2 < 0.5:
        verdict = "H3 — flat trend; like attention narrowing"
    else:
        verdict = "Negative direction — out-of-sample failure"

    print(f"\n==== Mamba pre-registered hypothesis test ====")
    print(f"  log(T1/T3) ~ √L  slope={slope:.3f}  R²={r2:.3f}")
    print(f"  Verdict: {verdict}", flush=True)

    payload = {
        "preregistered_hypothesis": (
            "H1 positive: Mamba SSM with distinct per-layer params follows √L "
            "(R² > 0.85, slope > 0); H2: smaller slope than ResNet due to "
            "selective gating; H3 null: flat/negative if something distinct."
        ),
        "config": vars(args), "per_run": rows,
        "summary": {"depths": depths, "mean_log_T1T3": means,
                     "slope_sqrt_L": slope, "R2": r2, "verdict": verdict,
                     "h1_pass": bool(slope > 0 and r2 > 0.85)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
