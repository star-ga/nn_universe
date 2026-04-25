"""V9.1 — GPT-Tiny depth sweep with UNTIED input/output embeddings.

V9 GPT-Tiny had tied embeddings (shared weight matrix between token
embedding and output projection); the result was a NEGATIVE √L slope,
which we framed as a narrowing of the universality claim.

This script tests the narrowing hypothesis directly: if we UNTIE the
embeddings, does the slope flip from negative to positive? That's the
falsifiable prediction the paper makes.
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


def fit_linear(xs, ys) -> tuple[float, float, float]:
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


class CausalAttn(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return out


class GPTBlock(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalAttn(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTUntied(nn.Module):
    """GPT-Tiny with SEPARATE input embedding and output projection."""
    def __init__(self, depth, dim, vocab, seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([GPTBlock(dim) for _ in range(depth)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)  # SEPARATE from self.tok
        # Force a fresh init that doesn't share with tok.weight.
        nn.init.normal_(self.head.weight, std=0.02)
        self.seq_len = seq_len
    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device).unsqueeze(0))
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))


def fim_diagonal(net, vocab, seq_len, n_probes):
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for _ in range(n_probes):
        x = torch.randint(0, vocab, (1, seq_len))
        logits = net(x)
        target = torch.cat([x[:, 1:], x[:, :1]], dim=1)
        loss = F.cross_entropy(logits.view(-1, vocab), target.view(-1))
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
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_1_gpt_untied_results.json"))
    args = ap.parse_args()

    rows = []
    per_L_lr = {}
    for L in args.depths:
        per_L_lr[L] = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            net = GPTUntied(L, args.dim, args.vocab, args.seq_len)
            n_params = sum(p.numel() for p in net.parameters())
            t0 = time.time()
            fim = fim_diagonal(net, args.vocab, args.seq_len, args.n_probes)
            t1m, t3m, ratio = tier_ratio(fim)
            dt = time.time() - t0
            rows.append({"depth": L, "seed": seed, "n_params": int(n_params),
                          "tier_ratio": ratio, "elapsed_s": dt})
            if ratio > 0:
                per_L_lr[L].append(math.log(ratio))
            print(f"  untied L={L:>2} seed={seed} N={n_params:>7,} T1/T3={ratio:.3e} ({dt:.1f}s)", flush=True)

    depths = sorted(per_L_lr.keys())
    sqrtL = [math.sqrt(L) for L in depths]
    means = [float(np.mean(per_L_lr[L])) if per_L_lr[L] else float("nan") for L in depths]
    slope, _, r2 = fit_linear(sqrtL, means)

    print(f"\n==== Untied GPT-Tiny ====")
    print(f"  log(T1/T3) vs sqrt(L)  slope={slope:.3f}  R²={r2:.3f}  "
          f"{'PASS √L (positive)' if slope > 0 and r2 > 0.8 else 'FAIL/negative'}")

    payload = {"config": vars(args), "per_run": rows,
               "summary": {"slope_sqrt_L": slope, "R2": r2, "depths": depths,
                           "mean_log_T1T3": means},
               "interpretation": (
                   "If slope > 0 with R² > 0.8, the prediction in the paper "
                   "(GPT-Tiny's negative slope is due to tied embeddings) is "
                   "confirmed. If slope is still negative, the narrowing is "
                   "more general than just embedding tying."
               )}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
