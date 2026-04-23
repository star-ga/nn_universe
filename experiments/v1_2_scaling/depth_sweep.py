"""V1.2 — depth sweep at fixed width.

Complements the V1.0 width sweep by varying depth at fixed width=256.
Answers: does the SV hierarchy strengthen with depth, saturate, or degrade?
The FIM-Onsager picture predicts deeper networks give stronger hierarchies
(more 'physical-constant-like' parameters in Tier 1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scaling_experiment_extended import _CheckpointedFC, _fim_tier_ratio, _max_sv_ratio  # noqa: E402


def build_deep_net(dim: int, width: int, depth: int) -> nn.Module:
    """FC net with the requested depth of hidden layers (Linear+ReLU)."""
    layers: list[nn.Module] = [nn.Linear(dim, width), nn.ReLU()]
    for _ in range(depth - 1):
        layers.append(nn.Linear(width, width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, dim))
    return nn.Sequential(*layers)


def run_one(args, depth: int) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_deep_net(args.dim, args.width, depth).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    t0 = time.time()
    for _ in range(args.steps):
        x = torch.randn(args.batch, args.dim, device=device)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt = time.time() - t0

    # Reuse V1.2 analyzers.
    sv = _max_sv_ratio(net)
    fim = _fim_tier_ratio(net, device, torch.float32)
    return {
        "depth": depth,
        "width": args.width,
        "params": n_params,
        "max_sv_ratio": round(sv, 1),
        "fim_tier1_tier3": round(fim, 1),
        "train_time": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 5, 8, 12, 20])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "depth_sweep_results.json"))
    args = ap.parse_args()

    results = []
    for d in args.depths:
        print(f"[{time.strftime('%H:%M:%S')}] depth={d} width={args.width}")
        try:
            r = run_one(args, d)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at depth={d}; skipping.")
            torch.cuda.empty_cache()
            continue
        print(f"  params={r['params']:,} SV={r['max_sv_ratio']}x FIM={r['fim_tier1_tier3']}x ({r['train_time']}s)")
        results.append(r)
        torch.cuda.empty_cache()

    payload = {"config": vars(args), "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
