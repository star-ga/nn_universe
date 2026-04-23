"""Cluster-side multi-seed sweep that writes PER-SEED json rather than
overwriting ``scaling_results.json``.

Use on the Runpod pod (or any other cluster node) as::

    python3 experiments/v1_2_scaling/multiseed_remote.py \\
        --width 14000 --seeds 42 43 44 45 46 \\
        --steps 10000 --bf16 --grad-ckpt \\
        --out /workspace/v3_0_width14000_seeds.json

Each seed's full result (SV per layer, max SV ratio, FIM tier stats) is
recorded; the aggregated output has mean/std/CV and is ready for
ingestion by ``experiments/visualize.py`` via the ``extra`` hook.
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

from scaling_experiment_extended import (  # noqa: E402
    _CheckpointedFC,
    _fim_tier_ratio,
    _max_sv_ratio,
)


def run(*, width: int, seed: int, dim: int, steps: int, batch: int, bf16: bool, grad_ckpt: bool, fim_samples: int) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (bf16 and device.type == "cuda") else torch.float32
    net = _CheckpointedFC(dim, width, grad_ckpt=grad_ckpt).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in net.parameters())

    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    t0 = time.time()
    for _ in range(steps):
        x = torch.randn(batch, dim, device=device, dtype=dtype)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_time = time.time() - t0

    sv = _max_sv_ratio(net)
    fim = _fim_tier_ratio(net, device, dtype)
    del net
    torch.cuda.empty_cache()
    return {
        "seed": seed,
        "width": width,
        "params": n_params,
        "max_sv_ratio": round(sv, 1),
        "fim_tier1_tier3": round(fim, 1),
        "train_time": round(train_time, 1),
        "bf16": bool(bf16),
        "grad_ckpt": bool(grad_ckpt),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fim-samples", type=int, default=500)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    results: list[dict] = []
    for s in args.seeds:
        print(f"[{time.strftime('%H:%M:%S')}] seed={s} width={args.width}", flush=True)
        r = run(
            width=args.width,
            seed=s,
            dim=args.dim,
            steps=args.steps,
            batch=args.batch,
            bf16=args.bf16,
            grad_ckpt=args.grad_ckpt,
            fim_samples=args.fim_samples,
        )
        print(f"  SV={r['max_sv_ratio']}x FIM={r['fim_tier1_tier3']}x ({r['train_time']}s)", flush=True)
        results.append(r)
        # Save incrementally in case of interruption.
        _save(args.out, results)

    summary = _summarize(results)
    print(f"\nSV   mean={summary['sv_mean']:.1f}  std={summary['sv_std']:.1f}  CV={summary['sv_cv']:.2%}")
    print(f"FIM  mean={summary['fim_mean']:.1f}  std={summary['fim_std']:.1f}  CV={summary['fim_cv']:.2%}")
    return 0


def _save(out: str, results: list[dict]) -> None:
    payload = {
        "config": {"n_seeds": len(results)},
        "results": results,
        **_summarize(results),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def _summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    sv = np.array([r["max_sv_ratio"] for r in results])
    fim = np.array([r["fim_tier1_tier3"] for r in results])
    out = {
        "sv_mean": float(sv.mean()),
        "sv_std": float(sv.std(ddof=1)) if len(sv) > 1 else 0.0,
        "sv_cv": float(sv.std(ddof=1) / sv.mean()) if len(sv) > 1 and sv.mean() > 0 else 0.0,
        "fim_mean": float(fim.mean()),
        "fim_std": float(fim.std(ddof=1)) if len(fim) > 1 else 0.0,
        "fim_cv": float(fim.std(ddof=1) / fim.mean()) if len(fim) > 1 and fim.mean() > 0 else 0.0,
    }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
