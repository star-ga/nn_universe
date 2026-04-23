"""V2.1 — width sweep across QEC neural decoders.

Tests whether the SV ~ N^alpha scaling law observed in the self-prediction
cosmology experiment (V1.0) persists for decoders trained on a completely
different task (toric-code syndrome correction). If yes, the spectral
hierarchy is an architectural-universal property; if no, it is task-dependent.

Uses the same 5-layer ReLU MLP with variable width, matches V1.0 protocol.
Writes `v2_1_sweep_results.json`.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import fim_diagonal, max_sv_ratio, tier_partition  # noqa: E402
from decoder import MLPDecoder  # noqa: E402
from toric_code import compute_syndromes, make_batch, parity_check_matrix  # noqa: E402

DEFAULT_WIDTHS = [32, 64, 128, 256, 512, 1024]


def run_one(*, width: int, L: int, p: float, steps: int, batch: int, seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    H = parity_check_matrix(L)
    n_syn, n_qub = H.shape
    net = MLPDecoder(n_syn, n_qub, width=width, hidden_layers=5).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    bce = nn.BCEWithLogitsLoss()

    t0 = time.time()
    final_loss = None
    for step in range(steps):
        s_np, e_np = make_batch(L, p, batch, rng, H)
        s = torch.from_numpy(s_np).to(device)
        e = torch.from_numpy(e_np).to(device)
        logits = net(s)
        loss = bce(logits, e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach())
    train_time = time.time() - t0

    sv_ratio = max_sv_ratio(net)

    def sample_loss():
        s_np, e_np = make_batch(L, p, 32, rng, H)
        s = torch.from_numpy(s_np).to(device)
        e = torch.from_numpy(e_np).to(device)
        return bce(net(s), e)

    fim = fim_diagonal(net, sample_loss, n_samples=300)
    tiers = tier_partition(fim)
    return {
        "width": width,
        "n_params": n_params,
        "L": L,
        "p": p,
        "max_sv_ratio": round(sv_ratio, 1),
        "fim_tier1_tier3": round(tiers["ratio_tier1_tier3"], 1),
        "final_loss": round(final_loss or 0.0, 6),
        "train_time": round(train_time, 1),
        "seed": seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(HERE / "v2_1_sweep_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu})")

    results = []
    for w in args.widths:
        print(f"[{time.strftime('%H:%M:%S')}] width={w}")
        try:
            r = run_one(width=w, L=args.L, p=args.p, steps=args.steps, batch=args.batch, seed=args.seed, device=device)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at width={w}; skipping.")
            torch.cuda.empty_cache()
            continue
        print(
            f"  params={r['n_params']:,}  SV={r['max_sv_ratio']}x  FIM={r['fim_tier1_tier3']}x  "
            f"loss={r['final_loss']:.4f}  ({r['train_time']}s)"
        )
        results.append(r)
        torch.cuda.empty_cache()

    # Power-law fit (log-log)
    if len(results) >= 2:
        log_p = np.log10([r["n_params"] for r in results])
        log_sv = np.log10(np.clip([r["max_sv_ratio"] for r in results], 1, None))
        log_fim = np.log10(np.clip([r["fim_tier1_tier3"] for r in results], 1, None))
        sv_fit = np.polyfit(log_p, log_sv, 1)
        fim_fit = np.polyfit(log_p, log_fim, 1)

        def _r2(y, fit):
            pred = np.polyval(fit, log_p)
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        sv_r2 = _r2(log_sv, sv_fit)
        fim_r2 = _r2(log_fim, fim_fit)
        sv_pl = {"exponent": round(float(sv_fit[0]), 3), "r_squared": round(sv_r2, 3)}
        fim_pl = {"exponent": round(float(fim_fit[0]), 3), "r_squared": round(fim_r2, 3)}
    else:
        sv_pl = fim_pl = {"exponent": 0.0, "r_squared": 0.0}

    payload = {
        "device": str(device),
        "gpu": gpu,
        "config": vars(args),
        "results": results,
        "sv_power_law": sv_pl,
        "fim_power_law": fim_pl,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSV power law:  exponent={sv_pl['exponent']}  R²={sv_pl['r_squared']}")
    print(f"FIM power law: exponent={fim_pl['exponent']}  R²={fim_pl['r_squared']}")
    print(f"Saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
