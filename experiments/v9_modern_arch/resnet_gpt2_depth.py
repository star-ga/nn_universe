"""V9 — modern-architecture depth sweep on real ML scales.

NeurIPS reviewers flagged toy-architecture coverage. This module measures
the FIM tier hierarchy for:

  1. ResNet-style 'Bottleneck-Tiny' depth sweep (depths 4, 8, 16, 32, 64
     residual blocks) on the V1.0 self-prediction task. Tests the √L
     law on ResNet residual stacks with batch-norm.
  2. GPT-2-style 'GPT-Tiny' depth sweep (depths 1, 2, 4, 6, 8, 12 blocks
     with multi-head attention + FFN + LayerNorm + residual) on a
     synthetic next-token-prediction task with vocab=64 and seq_len=16.

These cover the modern-architecture gap (ResNet + GPT) at a scale that
fits the RTX 3080 budget.
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


def tier_ratio(values: np.ndarray, top_pct: float = 1.0, bot_pct: float = 50.0):
    s = np.sort(values)[::-1]; n = len(s)
    k1 = max(1, int(n * top_pct / 100)); k3 = max(1, int(n * bot_pct / 100))
    t1 = float(s[:k1].mean()); t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1/t3 if t3 > 0 else float("inf"))


# ============================================================
# ResNet-Tiny: residual blocks with BatchNorm
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.bn2(self.fc2(h))
        return F.relu(h + x)


class ResNetTiny(nn.Module):
    def __init__(self, depth: int, width: int, dim: int):
        super().__init__()
        self.stem = nn.Linear(dim, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.head = nn.Linear(width, dim)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


# ============================================================
# GPT-Tiny: pre-norm transformer + causal mask
# ============================================================

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
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalAttn(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTTiny(nn.Module):
    def __init__(self, depth: int, dim: int, vocab: int, seq_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([GPTBlock(dim) for _ in range(depth)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.seq_len = seq_len

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device).unsqueeze(0))
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))


# ============================================================
# FIM diagonal estimators
# ============================================================

def fim_resnet(net: ResNetTiny, dim: int, n_probes: int) -> np.ndarray:
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.train()  # need BatchNorm to use batch stats; running mean/var should not matter for FIM
    for _ in range(n_probes):
        x = torch.randn(2, dim)  # batch=2 minimum for BatchNorm
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


def fim_gpt(net: GPTTiny, vocab: int, seq_len: int, n_probes: int) -> np.ndarray:
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for _ in range(n_probes):
        x = torch.randint(0, vocab, (1, seq_len))
        # next-token loss with shifted target
        logits = net(x)  # (1, T, vocab)
        target = torch.cat([x[:, 1:], x[:, :1]], dim=1)  # cyclic shift
        loss = F.cross_entropy(logits.view(-1, vocab), target.view(-1))
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data.double() ** 2
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


# ============================================================
# Driver
# ============================================================

def run_resnet(depths, seeds, width, dim, n_probes):
    rows = []
    for L in depths:
        for seed in seeds:
            torch.manual_seed(seed)
            net = ResNetTiny(L, width, dim)
            n_params = sum(p.numel() for p in net.parameters())
            t0 = time.time()
            fim = fim_resnet(net, dim, n_probes)
            t1m, t3m, ratio = tier_ratio(fim)
            dt = time.time() - t0
            rows.append({"family": "resnet", "depth": L, "seed": seed,
                          "n_params": int(n_params), "tier_ratio": ratio,
                          "elapsed_s": dt})
            print(f"  resnet L={L:>2} seed={seed} N={n_params:>7,} T1/T3={ratio:.3e} ({dt:.1f}s)", flush=True)
    return rows


def run_gpt(depths, seeds, dim, vocab, seq_len, n_probes):
    rows = []
    for L in depths:
        for seed in seeds:
            torch.manual_seed(seed)
            net = GPTTiny(L, dim, vocab, seq_len)
            n_params = sum(p.numel() for p in net.parameters())
            t0 = time.time()
            fim = fim_gpt(net, vocab, seq_len, n_probes)
            t1m, t3m, ratio = tier_ratio(fim)
            dt = time.time() - t0
            rows.append({"family": "gpt", "depth": L, "seed": seed,
                          "n_params": int(n_params), "tier_ratio": ratio,
                          "elapsed_s": dt})
            print(f"  gpt    L={L:>2} seed={seed} N={n_params:>7,} T1/T3={ratio:.3e} ({dt:.1f}s)", flush=True)
    return rows


def summarise(rows, family):
    depths = sorted(set(r["depth"] for r in rows if r["family"] == family))
    means_log = []
    for L in depths:
        rs = [r["tier_ratio"] for r in rows if r["family"] == family and r["depth"] == L and r["tier_ratio"] > 0]
        means_log.append(float(np.mean(np.log(rs))) if rs else float("nan"))
    sqrtL = [math.sqrt(L) for L in depths]
    slope, _, r2 = fit_linear(sqrtL, means_log)
    return {"depths": depths, "mean_log_T1T3": means_log,
            "slope_sqrt_L": slope, "R2_sqrt_L": r2,
            "pass": bool(r2 > 0.8)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet-depths", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--gpt-depths", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_resnet_gpt2_results.json"))
    args = ap.parse_args()

    rows = run_resnet(args.resnet_depths, args.seeds, args.width, args.dim, args.n_probes)
    rows += run_gpt(args.gpt_depths, args.seeds, args.dim, args.vocab, args.seq_len, args.n_probes)

    summary = {
        "resnet": summarise(rows, "resnet"),
        "gpt": summarise(rows, "gpt"),
    }
    print("\n==== summary ====")
    for fam, s in summary.items():
        print(f"  {fam:<6} slope={s['slope_sqrt_L']:.3f} R²={s['R2_sqrt_L']:.3f}  "
              f"{'PASS' if s['pass'] else 'FAIL'}")

    payload = {"config": vars(args), "per_run": rows, "summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
