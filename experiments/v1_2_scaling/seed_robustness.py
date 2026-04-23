"""V1.2 — seed-robustness check for the scaling power law.

For a fixed width, train N times with different RNG seeds and report the
standard deviation of max_sv_ratio and fim_tier1_tier3. A low stddev
relative to the mean is required before we can interpret slope shifts in
the power-law fit as real physics.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scaling_experiment_extended import _CheckpointedFC, _fim_tier_ratio, _max_sv_ratio  # noqa: E402


def run(args, seed: int) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _CheckpointedFC(args.dim, args.width, grad_ckpt=False).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    t0 = time.time()
    for _ in range(args.steps):
        x = torch.randn(args.batch, args.dim, device=device)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    sv = _max_sv_ratio(net)
    fim = _fim_tier_ratio(net, device, torch.float32)
    return {"seed": seed, "max_sv_ratio": round(sv, 1), "fim_tier1_tier3": round(fim, 1), "train_time": round(dt, 1)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "seed_robustness_results.json"))
    args = p.parse_args()

    results = []
    for s in args.seeds:
        print(f"[seed {s}] width={args.width} steps={args.steps}")
        r = run(args, s)
        print(f"  SV={r['max_sv_ratio']}x FIM={r['fim_tier1_tier3']}x  ({r['train_time']}s)", flush=True)
        results.append(r)

    sv = np.array([r["max_sv_ratio"] for r in results])
    fim = np.array([r["fim_tier1_tier3"] for r in results])

    summary = {
        "config": vars(args),
        "results": results,
        "sv_mean": float(sv.mean()),
        "sv_std": float(sv.std(ddof=1)) if len(sv) > 1 else 0.0,
        "sv_cv": float(sv.std(ddof=1) / sv.mean()) if len(sv) > 1 and sv.mean() > 0 else 0.0,
        "fim_mean": float(fim.mean()),
        "fim_std": float(fim.std(ddof=1)) if len(fim) > 1 else 0.0,
        "fim_cv": float(fim.std(ddof=1) / fim.mean()) if len(fim) > 1 and fim.mean() > 0 else 0.0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMean SV ratio: {summary['sv_mean']:.1f} ± {summary['sv_std']:.1f}  (CV={summary['sv_cv']:.2%})")
    print(f"Mean FIM T1/T3: {summary['fim_mean']:.1f} ± {summary['fim_std']:.1f}  (CV={summary['fim_cv']:.2%})")
    print(f"Saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
