"""V9.7 — ImageNet ViT-L/16 FIM measurement (production-scale vision transformer).

Uses torchvision's ImageNet-pretrained ViT-L/16 (304 M parameters; the
canonical Vision Transformer Large with 16x16 patches). Multiple weight
checkpoints are available; we use the IMAGENET1K_SWAG_LINEAR_V1 variant
(79.66 % ImageNet-1K top-1) as the default real-scale ViT baseline.

Goal: confirm FIM tier hierarchy (T1/T3 magnitude + Gini + top-1% mass)
on a production-scale attention-based vision architecture. Combined
with V9.5 ResNet-50 (CNN production-scale) and V9.6 GPT-2-medium
(language production-scale), this gives three production-scale data
points across the canonical modern-architecture families.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gini(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v >= 0]
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v.sort()
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def effective_rank(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v > 0]
    n = v.size
    if n == 0:
        return 0.0
    return float((v.sum() ** 2) / (n * (v ** 2).sum()))


def top_1pct_mass(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    s = np.sort(v)[::-1]
    k1 = max(1, int(s.size * 0.01))
    return float(s[:k1].sum() / s.sum())


def tier_ratio(values: np.ndarray):
    s = np.sort(values)[::-1]
    n = len(s)
    k1 = max(1, int(n * 0.01))
    k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean())
    t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz) // 10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1 / t3 if t3 > 0 else float("inf"))


def imagenet_synthetic_probe(device, batch=1, size=224):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.randn(batch, 3, size, size, device=device)
    return (x - mean) / std


def fim_diagonal(net, n_probes, device, input_size=224):
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for i in range(n_probes):
        x = imagenet_synthetic_probe(device, batch=1, size=input_size)
        out = net(x)
        target = torch.randint(0, out.shape[1], (1,), device=device)
        loss = F.cross_entropy(out, target)
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data.double() ** 2
        if (i + 1) % 25 == 0:
            print(f"  probe {i+1}/{n_probes}", flush=True)
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_7_imagenet_vit_l16_results.json"))
    args = ap.parse_args()

    import torchvision
    from torchvision.models import vit_l_16, ViT_L_16_Weights

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading pretrained ViT-L/16 (ImageNet-1K SWAG linear, 79.66% top-1)...", flush=True)
    weights = ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    net = vit_l_16(weights=weights).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    pretrain_acc = "79.66%"
    print(f"ViT-L/16 params: {n_params:,}", flush=True)
    print(f"Pretrained ImageNet-1K top-1: {pretrain_acc}", flush=True)

    # ViT-L/16 SWAG_LINEAR_V1 expects 224x224 inputs (linear classifier on SWAG features)
    print(f"\nMeasuring FIM diagonal with {args.n_probes} ImageNet-statistics probes...", flush=True)
    t0 = time.time()
    fim = fim_diagonal(net, args.n_probes, device, input_size=224)
    t_fim = time.time() - t0
    t1m, t3m, ratio = tier_ratio(fim)
    g = gini(fim); rn = effective_rank(fim) / fim.size; tp1 = top_1pct_mass(fim)

    print(f"\n=== ImageNet ViT-L/16 FIM (V9.7) ===")
    print(f"  N params       = {n_params:,}")
    print(f"  Pretrained acc = {pretrain_acc}")
    print(f"  T1/T3          = {ratio:.3e}")
    print(f"  Gini           = {g:.4f}")
    print(f"  r_eff/n        = {rn:.5f}")
    print(f"  top-1% mass    = {tp1:.4f}")
    print(f"  FIM time       = {t_fim:.1f}s")

    payload = {
        "config": vars(args),
        "n_params": int(n_params),
        "pretrained_imagenet_top1": pretrain_acc,
        "tier1_mean": t1m, "tier3_mean": t3m, "tier_ratio": ratio,
        "partition_invariant": {
            "gini": g,
            "effective_rank_normalised": rn,
            "top_1pct_mass": tp1,
        },
        "fim_measurement_s": t_fim,
        "interpretation": (
            "FIM tier hierarchy on ImageNet-pretrained ViT-L/16 (304M params, "
            "79.66% top-1). Production-scale vision transformer (attention "
            "architecture). Combined with V9.5 ResNet-50 (CNN production-scale) "
            "and V9.6 GPT-2-medium (language production-scale), gives three "
            "production-scale data points across CNN + ViT + autoregressive-LM."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
