#!/usr/bin/env python3
"""V1.2 — fill the scaling ladder with intermediate widths for a denser power-law fit.

Augments the V1.0 sweep by training at widths that sit between the original
ladder points (16, 64, 256, 1024, 4096, 8192), giving ~10 points rather than 6.

Each width is an idempotent, separately-committed run: if the process is
interrupted, re-running resumes at the next missing width. Results land in
scaling_results.json (same schema as scaling_experiment.py).
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

from scaling_experiment_extended import (  # noqa: E402
    _RESULTS_PATH,
    _CheckpointedFC,
    _fim_tier_ratio,
    _load_results,
    _max_sv_ratio,
    _refit_power_law,
    _save_results,
)

FILL_LADDER = [32, 128, 512, 2048]
DIM = 32
STEPS = 20_000
BATCH = 128
SEED = 42


def _run_width(width: int, *, bf16: bool, grad_ckpt: bool) -> dict:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (bf16 and device.type == "cuda") else torch.float32
    net = _CheckpointedFC(DIM, width, grad_ckpt=grad_ckpt).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  width={width} params={n_params:,} device={device} dtype={dtype}")

    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    t0 = time.time()
    for _ in range(STEPS):
        x = torch.randn(BATCH, DIM, device=device, dtype=dtype)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_time = time.time() - t0

    sv_ratio = _max_sv_ratio(net)
    fim_ratio = _fim_tier_ratio(net, device, dtype)
    return {
        "width": width,
        "params": n_params,
        "max_sv_ratio": round(sv_ratio, 1),
        "fim_tier1_tier3": round(fim_ratio, 1),
        "train_time": round(train_time, 1),
        "bf16": bool(bf16),
        "grad_ckpt": bool(grad_ckpt),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--widths", type=str, default=None, help="comma-separated widths; default fill-ladder")
    args = parser.parse_args()

    widths = [int(w) for w in args.widths.split(",")] if args.widths else FILL_LADDER
    existing_path = REPO / _RESULTS_PATH
    os.chdir(REPO)

    payload = _load_results()
    existing_widths = {r["width"] for r in payload["results"]}
    targets = [w for w in widths if w not in existing_widths]
    if not targets:
        print("All target widths already present in scaling_results.json; nothing to do.")
        return 0

    for w in targets:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] width={w}  bf16={args.bf16}  grad_ckpt={args.grad_ckpt}")
            result = _run_width(w, bf16=args.bf16, grad_ckpt=args.grad_ckpt)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at width={w}; skipping.")
            torch.cuda.empty_cache()
            continue
        print(f"  SV={result['max_sv_ratio']}x  FIM={result['fim_tier1_tier3']}x  ({result['train_time']}s)")

        payload = _load_results()
        payload["results"] = [r for r in payload["results"] if r["width"] != w]
        payload["results"].append(result)
        payload["results"].sort(key=lambda r: r["params"])
        payload["sv_power_law"], payload["fim_power_law"] = _refit_power_law(payload["results"])
        _save_results(payload)

        print(
            f"  SV power law: N^{payload['sv_power_law']['exponent']}  "
            f"R²={payload['sv_power_law']['r_squared']}"
        )
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
