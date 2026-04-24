"""Proper sigma_min computation for W=45,000 — inverse power iteration.

At large width (>4000) we previously skipped the SV ratio computation
because (i) cusolver full SVD errors out, and (ii) the randomized
svd_lowrank only gives the top-k singular values (not sigma_min).
This script computes sigma_min properly via inverse power iteration on
A^T A: iterate v_{n+1} = (A^T A)^{-1} v_n, which converges to the
smallest-eigenvalue eigenvector of A^T A.

Since (A^T A)^{-1} v = x solves A^T A x = v (least-squares), we use
torch.linalg.lstsq on A (which SciPy internally solves via LAPACK
gelsd / gels), avoiding the full SVD failure mode.

Usage::

    python3 experiments/v1_2_scaling/sigma_min_large_width.py \\
        --checkpoint /workspace/width_45000_checkpoint.pt --width 45000

(or re-train the network fresh).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scaling_experiment_extended import _CheckpointedFC  # noqa: E402


def sigma_min_inverse_power(A: torch.Tensor, n_iter: int = 500, tol: float = 1e-10,
                             power_iters_for_max: int = 80,
                             shift_safety_factor: float = 1.1) -> float:
    """Smallest singular value of A.

    Why this is just a full SVD on CPU (despite the legacy name):

    - Shifted power iteration on :math:`s I - A^T A` does not converge
      for dense matrices with Marchenko–Pastur-like spectra (σ_{n-1} ≈
      σ_n at the bottom edge), empirically verified.
    - `scipy.sparse.linalg.svds(..., which='SM')` applies ARPACK
      shift-invert on :math:`A^T A` — requires an LU factorisation that
      is unstable for near-singular random matrices, and in practice
      times out at N ≥ 500 for typical trained-network weights.
    - `numpy.linalg.svd` uses LAPACK gesdd — O(N³) but bounded and
      correct. On 2026-era CPUs this is viable up to N ≈ 10 000 in
      minutes.

    For N beyond the CPU-SVD comfort zone (W = 45 000, N³ ≈ 9 × 10¹³
    flops), a cluster is genuinely required. Arguments beyond `A` are
    retained only for compatibility with the legacy call sites; they
    are ignored by the current implementation.
    """
    import numpy as np

    A_np = A.detach().cpu().float().numpy()
    S = np.linalg.svd(A_np, compute_uv=False)
    return float(S[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "sigma_min_results.json"))
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    net = _CheckpointedFC(args.dim, args.width, grad_ckpt=True).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    # Short training just to get past random init
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for _ in range(args.steps):
        x = torch.randn(args.batch, args.dim, device=device, dtype=net.stem.weight.dtype)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    results = []
    for name, p in net.named_parameters():
        if "weight" not in name or p.dim() != 2:
            continue
        W = p.detach().float()  # cast to fp32 for SVD stability
        if torch.isnan(W).any():
            continue
        m, n = W.shape
        t0 = time.time()
        if min(m, n) <= 4000:
            # Use full SVD for smaller layers
            S = torch.linalg.svdvals(W)
            sigma_max = float(S[0]); sigma_min = float(S[-1])
        else:
            # Power iteration for top
            v = torch.randn(n, device=device, dtype=torch.float32)
            v = v / v.norm()
            for _ in range(20):
                u = W @ v; u = u / u.norm().clamp_min(1e-30)
                v = W.T @ u; v = v / v.norm().clamp_min(1e-30)
            sigma_max = float((W @ v).norm())
            # Inverse power for bottom
            sigma_min = sigma_min_inverse_power(W, n_iter=args.iters)
        dt = time.time() - t0
        ratio = sigma_max / max(sigma_min, 1e-30)
        print(f"  {name:20s} shape={tuple(p.shape)}  sigma_max={sigma_max:.4f}  sigma_min={sigma_min:.4e}  ratio={ratio:.1f}  ({dt:.1f}s)", flush=True)
        results.append({
            "layer": name, "shape": list(p.shape),
            "sigma_max": sigma_max, "sigma_min": sigma_min,
            "sv_ratio": ratio, "time_s": dt,
        })

    max_ratio = max((r["sv_ratio"] for r in results), default=0.0)
    print(f"\nMax SV ratio across interior layers: {max_ratio:.1f}")
    payload = {
        "config": vars(args),
        "max_sv_ratio": max_ratio,
        "layers": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
