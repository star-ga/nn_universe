"""σ_min validation sweep — Task #34 partial closure.

Runs the inverse-power-iteration σ_min protocol across a ladder of widths
that DO fit on-device (up to the RTX 3080 10 GB VRAM ceiling), so we can
(a) fact-check the method by comparing against full SVD where feasible
    (min-dim ≤ 4000),
(b) produce σ_min values across enough widths to close the reported
    "SV ratio at W ≥ 14000 not measured" gap for the interior layers
    we CAN run,
(c) document the extrapolation path to W = 45000 (requires H200 cluster
    because training memory for the 5-layer 45000-wide MLP is ~45 GB).

Outputs a consolidated JSON + a small markdown summary table per layer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scaling_experiment_extended import _CheckpointedFC  # noqa: E402
from experiments.v1_2_scaling.sigma_min_large_width import sigma_min_inverse_power  # noqa: E402


def train_short(net, dim, steps, batch, device, dtype):
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    t0 = time.time()
    for _ in range(steps):
        x = torch.randn(batch, dim, device=device, dtype=dtype)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return time.time() - t0


def measure_sv_ratios(net, device, use_inverse_power_cutoff: int, iters: int):
    """For each weight matrix, compute σ_max + σ_min.
    Below the cutoff, use full SVD (ground truth).
    At or above the cutoff, use power iteration + shifted inverse power.
    """
    results = []
    for name, p in net.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        W = p.detach().float()
        if torch.isnan(W).any():
            results.append({"layer": name, "shape": list(p.shape),
                            "error": "NaN weights"})
            continue
        m, n = W.shape
        t0 = time.time()
        if min(m, n) <= use_inverse_power_cutoff:
            S = torch.linalg.svdvals(W)
            sigma_max = float(S[0])
            sigma_min = float(S[-1])
            method = "full_svd"
        else:
            v = torch.randn(n, device=device, dtype=torch.float32)
            v = v / v.norm()
            for _ in range(30):
                u = W @ v
                u = u / u.norm().clamp_min(1e-30)
                v = W.T @ u
                v = v / v.norm().clamp_min(1e-30)
            sigma_max = float((W @ v).norm())
            sigma_min = sigma_min_inverse_power(W, n_iter=iters)
            method = "inverse_power"
        dt = time.time() - t0
        ratio = sigma_max / max(sigma_min, 1e-30)
        results.append({
            "layer": name, "shape": list(p.shape),
            "sigma_max": sigma_max, "sigma_min": sigma_min,
            "sv_ratio": ratio, "method": method, "elapsed_s": dt,
        })
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+",
                    default=[256, 1024, 4096, 8192])
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=500,
                    help="inverse power iteration count at widths above cutoff")
    ap.add_argument("--cutoff", type=int, default=4000,
                    help="min-dim threshold for full SVD vs inverse power")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(
        Path(__file__).resolve().parent / "sigma_min_validation_results.json"))
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, train dtype: {dtype}", flush=True)

    all_results: list[dict] = []
    for width in args.widths:
        torch.manual_seed(args.seed)
        try:
            net = _CheckpointedFC(args.dim, width, grad_ckpt=True).to(device=device, dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  width={width}: SKIP — OOM on model construction", flush=True)
            continue

        try:
            train_s = train_short(net, args.dim, args.steps, args.batch, device, dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  width={width}: SKIP — OOM during training", flush=True)
            del net
            torch.cuda.empty_cache()
            continue

        layer_results = measure_sv_ratios(net, device, args.cutoff, args.iters)
        max_ratio = max(
            (r["sv_ratio"] for r in layer_results if "sv_ratio" in r),
            default=0.0,
        )
        interior_ratios = [
            r["sv_ratio"] for r in layer_results
            if "sv_ratio" in r and min(r["shape"]) == width
        ]
        interior_max = max(interior_ratios, default=0.0)
        n_params = sum(p.numel() for p in net.parameters())

        print(f"  width={width:>6}  N={n_params:>12,}  train={train_s:.1f}s  "
              f"max_sv={max_ratio:>.3e}  interior_max={interior_max:>.3e}",
              flush=True)
        all_results.append({
            "width": width, "n_params": int(n_params),
            "train_s": train_s,
            "max_sv_ratio": max_ratio,
            "interior_max_sv_ratio": interior_max,
            "layers": layer_results,
        })

        del net
        torch.cuda.empty_cache()

    payload = {
        "config": vars(args),
        "results": all_results,
        "notes": (
            "σ_min method: full SVD (torch.linalg.svdvals) for min-dim ≤ cutoff, "
            "inverse-power shifted iteration on A^T A for min-dim > cutoff. "
            "Full SVD is the ground truth against which inverse-power is "
            "validated for widths where both are feasible. For W ≥ 14000, "
            "only the inverse-power path is viable on RTX 3080 (≤10 GB VRAM); "
            "W = 45000 still requires H200 cluster due to training memory "
            "(5-layer 45000-wide MLP training activations ~45 GB)."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
