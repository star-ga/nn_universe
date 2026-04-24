"""V4.3 — tier-partition sensitivity.

Multi-LLM audit v3 flag (3/5 reviewers): the 1% / 49% / 50% tier
partition is arbitrary with no spectral-gap justification. Here we
vary the partition and report how the reported ratios move.

Re-computes the FIM diagonal on a small MLP (width 16, ~1400 params,
200 per-sample gradient probes) and measures T1/T3 ratios at six
different split definitions. If the ratio is partition-sensitive
within an order of magnitude, the "1%/50% tier" language is
defensible. If it varies across 4+ orders of magnitude, the partition
is effectively a fit parameter and must be flagged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_net(width: int, dim: int, layers: int) -> nn.Module:
    mods: list[nn.Module] = [nn.Linear(dim, width), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(width, width), nn.ReLU()]
    mods.append(nn.Linear(width, dim))
    return nn.Sequential(*mods)


def compute_fim_diag(net: nn.Module, n_samples: int, dim: int) -> np.ndarray:
    P = sum(p.numel() for p in net.parameters())
    diag = torch.zeros(P, dtype=torch.float64)
    for _ in range(n_samples):
        x = torch.randn(1, dim)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        g = torch.cat([p.grad.flatten().double() for p in net.parameters() if p.grad is not None])
        diag += g.pow(2)
    return (diag / n_samples).numpy()


def tier_ratio(values: np.ndarray, top_pct: float, bot_pct: float) -> float:
    sorted_desc = np.sort(values)[::-1]
    n = len(sorted_desc)
    k1 = max(1, int(n * top_pct / 100))
    k3 = max(1, int(n * bot_pct / 100))
    t1 = float(sorted_desc[:k1].mean())
    t3 = float(sorted_desc[-k3:].mean())
    if t3 <= 0:
        nonzero = sorted_desc[sorted_desc > 0]
        t3 = float(nonzero[-max(len(nonzero)//10, 1):].mean()) if len(nonzero) else 1e-30
    return t1 / t3 if t3 > 0 else float("inf")


def main() -> int:
    partitions = [
        (0.1, 10),    # tight — top 0.1% / bottom 10%
        (1.0, 50),    # canonical V1.0 choice
        (5.0, 50),    # looser top
        (10.0, 50),   # even looser top
        (1.0, 30),    # tighter bottom
        (1.0, 70),    # looser bottom
        (0.5, 50),    # half of canonical top
    ]

    seeds = [0, 1, 2, 3, 4]
    results: dict = {}
    for seed in seeds:
        torch.manual_seed(seed)
        net = make_net(width=16, dim=8, layers=5)
        diag = compute_fim_diag(net, n_samples=200, dim=8)
        row = {"seed": seed, "P": len(diag), "ratios": {}}
        for top_pct, bot_pct in partitions:
            key = f"top_{top_pct}pct_vs_bot_{bot_pct}pct"
            row["ratios"][key] = tier_ratio(diag, top_pct, bot_pct)
        results[f"seed_{seed}"] = row

    # Aggregate
    agg = {}
    for top_pct, bot_pct in partitions:
        key = f"top_{top_pct}pct_vs_bot_{bot_pct}pct"
        vals = np.array([results[f"seed_{s}"]['ratios'][key] for s in seeds])
        agg[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "cv": float(vals.std(ddof=1) / vals.mean()) if vals.mean() > 0 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    # Report
    print(f"{'Partition':<30} {'mean':>12} {'min':>12} {'max':>12} {'CV':>7}")
    for key, v in agg.items():
        print(f"{key:<30} {v['mean']:>12,.1f} {v['min']:>12,.1f} {v['max']:>12,.1f} {v['cv']*100:>6.1f}%")

    canonical = agg["top_1.0pct_vs_bot_50pct"]["mean"]
    print("\nRatio vs canonical (top 1% / bot 50%):")
    for key, v in agg.items():
        rel = v['mean'] / canonical if canonical > 0 else float('inf')
        print(f"  {key:<30} {rel:>8.3f}x")

    payload = {"partitions": partitions, "per_seed": results, "aggregate": agg}
    out_path = Path(__file__).resolve().parent / "v4_3_tier_partition_sensitivity.json"
    os.makedirs(out_path.parent, exist_ok=True)
    json.dump(payload, open(out_path, 'w'), indent=2)
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
