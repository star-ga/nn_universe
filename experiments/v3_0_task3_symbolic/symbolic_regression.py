"""Task-3 universality: symbolic regression of random polynomials.

Task spec
---------
Given 16 evaluation pairs (x_i, y_i) where y_i = sum_{k=0}^{D-1} c_k x_i^k
with c_k drawn i.i.d. from N(0, 1), predict the coefficient vector c in
R^D from the flat input (x_0, y_0, x_1, y_1, ..., x_15, y_15) in R^32.

This is structurally distinct from both V1.0 cosmology (self-prediction
of Gaussian noise) and V2.1 toric-code decoding (binary error correction
on a lattice). A positive signal here -- same SV power-law, same FIM
tier hierarchy -- would make the universality claim hold across three
genuinely different tasks, satisfying Naestro Tier-1 item 1.

Architecture
------------
Matches V1.0 and V2.1: 5-layer, 256-neuron ReLU MLP. Loss is MSE on
coefficient vector. Adam optimizer (matches V2.1, outperforms SGD on
this structured task).
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
sys.path.insert(0, str(HERE.parents[1] / "experiments/v2_1_qec"))
from analyze import fim_diagonal, sv_per_layer, tier_partition  # noqa: E402


def sample_polynomial_batch(batch: int, degree: int, n_evals: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (flat_input, coefficients).

    flat_input: shape (batch, 2 * n_evals) -- interleaved (x, y) pairs.
    coefficients: shape (batch, degree).
    """
    xs = rng.uniform(-1.0, 1.0, size=(batch, n_evals))  # eval points in [-1, 1]
    coeffs = rng.standard_normal(size=(batch, degree))
    powers = np.power(xs[:, :, None], np.arange(degree)[None, None, :])  # (B, n_evals, degree)
    ys = np.einsum("bnd,bd->bn", powers, coeffs)  # (B, n_evals)
    flat = np.empty((batch, 2 * n_evals), dtype=np.float32)
    flat[:, 0::2] = xs
    flat[:, 1::2] = ys
    return flat.astype(np.float32), coeffs.astype(np.float32)


class MLPReg(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, hidden_layers: int = 5):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run(*, degree: int, n_evals: int, width: int, steps: int, batch: int,
        seed: int, fim_samples: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_dim = 2 * n_evals
    out_dim = degree
    net = MLPReg(in_dim, out_dim, width=width, hidden_layers=5).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    mse = nn.MSELoss()

    t0 = time.time()
    final_loss = None
    for step in range(steps):
        x_np, c_np = sample_polynomial_batch(batch, degree, n_evals, rng)
        x = torch.from_numpy(x_np).to(device)
        c = torch.from_numpy(c_np).to(device)
        pred = net(x)
        loss = mse(pred, c)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach())
    train_time = time.time() - t0

    # Baseline: MSE of all-zeros prediction = variance of random gaussian coefs = 1.0.
    print(f"  final MSE = {final_loss:.4f} (trivial all-zero baseline = 1.0)", flush=True)

    sv_stats = sv_per_layer(net)
    max_sv = max((s["sv_ratio"] for s in sv_stats), default=0.0)

    def sample_loss():
        x_np, c_np = sample_polynomial_batch(32, degree, n_evals, rng)
        x = torch.from_numpy(x_np).to(device)
        c = torch.from_numpy(c_np).to(device)
        return mse(net(x), c)

    fim = fim_diagonal(net, sample_loss, n_samples=fim_samples)
    tiers = tier_partition(fim)
    return {
        "width": width,
        "n_params": n_params,
        "degree": degree,
        "n_evals": n_evals,
        "max_sv_ratio": round(max_sv, 1),
        "fim_tier1_tier3": round(tiers["ratio_tier1_tier3"], 1),
        "final_loss": round(final_loss or 0.0, 6),
        "train_time": round(train_time, 1),
        "seed": seed,
        "sv_stats": sv_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--n-evals", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fim-samples", type=int, default=300)
    ap.add_argument("--out", type=str, default=str(HERE / "v3_0_task3_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Task: symbolic regression, degree={args.degree}, n_evals={args.n_evals}")

    results = []
    for w in args.widths:
        print(f"[{time.strftime('%H:%M:%S')}] width={w}", flush=True)
        try:
            r = run(
                degree=args.degree,
                n_evals=args.n_evals,
                width=w,
                steps=args.steps,
                batch=args.batch,
                seed=args.seed,
                fim_samples=args.fim_samples,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at width={w}")
            torch.cuda.empty_cache()
            continue
        # Drop the per-layer sv_stats from the per-row table to keep output small;
        # we keep them at the top level for the first width (as an example).
        sv_stats = r.pop("sv_stats")
        if w == args.widths[0]:
            r["sv_stats"] = sv_stats
        print(
            f"  params={r['n_params']:,}  SV={r['max_sv_ratio']}x  FIM={r['fim_tier1_tier3']}x  loss={r['final_loss']:.4f}",
            flush=True,
        )
        results.append(r)
        torch.cuda.empty_cache()

    # Power-law fit
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

        sv_pl = {"exponent": round(float(sv_fit[0]), 3), "r_squared": round(_r2(log_sv, sv_fit), 3)}
        fim_pl = {"exponent": round(float(fim_fit[0]), 3), "r_squared": round(_r2(log_fim, fim_fit), 3)}
    else:
        sv_pl = fim_pl = {"exponent": 0.0, "r_squared": 0.0}

    payload = {
        "config": vars(args),
        "device": str(device),
        "results": results,
        "sv_power_law": sv_pl,
        "fim_power_law": fim_pl,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSV power law:  exponent={sv_pl['exponent']}  R^2={sv_pl['r_squared']}", flush=True)
    print(f"FIM power law: exponent={fim_pl['exponent']}  R^2={fim_pl['r_squared']}", flush=True)
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
