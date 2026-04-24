"""V4.1.2 — trained-vs-untrained across MLP / CNN / ViT.

Extends V4.1 (which tested only ReLU MLPs) to conv and attention
architectures. Same protocol: for each architecture, build the network,
measure FIM T1/T3 diagonal before training (untrained), then train on
the shared 32×32×3 Gaussian-noise autoencoder task and re-measure
(trained). Report the trained/untrained ratio per architecture.

If training dissipates the hierarchy in all three architectures, the
V4.1 re-interpretation ("hierarchy is architecture-induced, not
learning-induced") extends to conv+attention. If it's MLP-specific,
V4.1 needs to be hedged.
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
sys.path.insert(0, str(HERE.parents[1] / "experiments/v2_1_qec"))

from arch_baselines import MLPCtrl, SmallCNN, SmallViT, count_params  # noqa: E402
from analyze import fim_diagonal, tier_partition  # noqa: E402


def build(arch_name: str, seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    if arch_name == "mlp":
        return MLPCtrl(in_dim=3 * 32 * 32, width=256, layers=5).to(device)
    if arch_name == "cnn":
        return SmallCNN(base_ch=32).to(device)
    if arch_name == "vit":
        return SmallViT(img_size=32, patch=4, dim=192, depth=4, heads=3).to(device)
    raise ValueError(arch_name)


def train_one(model: nn.Module, batch: int, steps: int, device: torch.device) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    mse = nn.MSELoss()
    final = None
    for _ in range(steps):
        x = torch.randn(batch, 3, 32, 32, device=device)
        y = model(x)
        loss = mse(y, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final = float(loss.detach())
    return final or 0.0


def measure_fim(model: nn.Module, batch: int, probes: int, device: torch.device) -> dict:
    mse = nn.MSELoss()
    def sample_loss():
        x = torch.randn(batch, 3, 32, 32, device=device)
        return mse(model(x), x)
    fim = fim_diagonal(model, sample_loss, n_samples=probes)
    tiers = tier_partition(fim)
    return {"n_params": count_params(model), "tier_ratio": tiers["ratio_tier1_tier3"]}


def run_one(arch: str, seed: int, steps: int, batch: int, probes: int, device: torch.device) -> tuple[dict, dict]:
    untrained_model = build(arch, seed, device)
    untrained = measure_fim(untrained_model, batch, probes, device)
    del untrained_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    trained_model = build(arch, seed, device)
    train_one(trained_model, batch=batch, steps=steps, device=device)
    trained = measure_fim(trained_model, batch, probes, device)
    del trained_model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return untrained, trained


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["mlp", "cnn", "vit"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--probes", type=int, default=200)
    ap.add_argument("--out", type=str,
                    default=str(HERE / "v4_1_2_arch_trained_vs_untrained.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu})", flush=True)

    rows: list[dict] = []
    for arch in args.archs:
        print(f"=== arch: {arch} ===", flush=True)
        u_list = []; t_list = []
        for s in args.seeds:
            t0 = time.time()
            u, t = run_one(arch, s, args.steps, args.batch, args.probes, device)
            dt = time.time() - t0
            u_list.append(u["tier_ratio"]); t_list.append(t["tier_ratio"])
            print(
                f"  {arch} seed={s}  N={u['n_params']:>9,}  "
                f"untrained={u['tier_ratio']:>12,.1f}  trained={t['tier_ratio']:>12,.1f}  "
                f"({dt:.1f}s)",
                flush=True,
            )
        ua = np.array(u_list); ta = np.array(t_list)
        rows.append({
            "arch": arch,
            "n_params": u["n_params"],
            "n_seeds": len(args.seeds),
            "untrained_mean": float(ua.mean()),
            "untrained_std": float(ua.std(ddof=1)) if len(ua) > 1 else 0.0,
            "trained_mean": float(ta.mean()),
            "trained_std": float(ta.std(ddof=1)) if len(ta) > 1 else 0.0,
            "training_reduction_factor": float(ua.mean() / ta.mean()) if ta.mean() > 0 else float("inf"),
        })

    payload = {"config": vars(args), "results": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
