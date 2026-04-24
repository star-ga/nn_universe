"""V4.2 — validate the FIM-diagonal proxy against the full FIM spectrum.

Earlier iterations of the repo described the "FIM eigenvalue
hierarchy" but only measured the diagonal of the FIM. In
high-dimensional positive-definite matrices the diagonal is an upper
bound on eigenvalues and can diverge from the true spectrum.

This experiment measures both at small N (where full-spectrum Lanczos
is tractable) and reports the agreement. Protocol:

1. Build a small MLP (N_params ~ 1000).
2. Construct the empirical FIM as a (P, P) matrix via `M = J^T J`
   where `J` is the (B, P) Jacobian of the loss w.r.t. parameters over
   a batch of B samples. This is O(P^2) memory but tractable for P < 5k.
3. Compare tier ratios computed on:
     (a) diagonal of M
     (b) eigenvalues of M (via np.linalg.eigvalsh or scipy Lanczos)

If (a) and (b) agree within 20–30% across tier boundaries, the diagonal
proxy is validated for our claims. If they disagree by an order of
magnitude, all FIM-diagonal language in the repo must be corrected.
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
import torch.autograd.functional as F


def make_net(width: int, dim: int = 8, layers: int = 5) -> nn.Module:
    mods: list[nn.Module] = [nn.Linear(dim, width), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(width, width), nn.ReLU()]
    mods.append(nn.Linear(width, dim))
    return nn.Sequential(*mods)


def flat_params(net: nn.Module) -> torch.Tensor:
    return torch.cat([p.flatten() for p in net.parameters()])


def flat_grads(net: nn.Module) -> torch.Tensor:
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in net.parameters()
    ])


def compute_fim_full(net: nn.Module, n_samples: int, dim: int, batch_per_sample: int = 1) -> np.ndarray:
    """Empirical FIM: M = (1/N) sum_n grad_n grad_n^T, where grad_n is
    the batch-1 gradient of the loss on sample n.

    Builds the full (P, P) matrix by accumulating grad_n outer-products.
    """
    P = flat_params(net).numel()
    M = torch.zeros(P, P, dtype=torch.float64)
    for _ in range(n_samples):
        x = torch.randn(batch_per_sample, dim)
        y = net(x)
        loss = 0.5 * (y - x).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        g = flat_grads(net).detach().cpu().double()
        M += torch.outer(g, g)
    M /= n_samples
    return M.numpy()


def compute_fim_diag(net: nn.Module, n_samples: int, dim: int, batch_per_sample: int = 1) -> np.ndarray:
    """Classic FIM diagonal via E[g_i^2]."""
    P = flat_params(net).numel()
    diag = torch.zeros(P, dtype=torch.float64)
    for _ in range(n_samples):
        x = torch.randn(batch_per_sample, dim)
        y = net(x)
        loss = 0.5 * (y - x).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        g = flat_grads(net).detach().cpu().double()
        diag += g.pow(2)
    diag /= n_samples
    return diag.numpy()


def tier_stats(values: np.ndarray) -> dict:
    """Tier-1/2/3 partition (top 1% / 1-50% / bottom 50%)."""
    sorted_desc = np.sort(values)[::-1]
    n = len(sorted_desc)
    k1 = max(1, n // 100)
    k2 = n // 2
    t1 = float(sorted_desc[:k1].mean())
    t2 = float(sorted_desc[k1:k2].mean()) if k2 > k1 else 0.0
    t3 = float(sorted_desc[k2:].mean()) if k2 < n else 0.0
    # guard underflow
    if t3 <= 0 and len(sorted_desc) > 0:
        nonzero = sorted_desc[sorted_desc > 0]
        t3 = float(nonzero[-max(len(nonzero)//10, 1):].mean()) if len(nonzero) else 1e-30
    return {
        "n": n, "k1": k1, "k2": k2,
        "tier1_mean": t1, "tier2_mean": t2, "tier3_mean": t3,
        "ratio_t1_t3": t1 / t3 if t3 > 0 else float("inf"),
        "ratio_t1_t2": t1 / t2 if t2 > 0 else float("inf"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v4_2_diag_vs_spectrum.json"))
    args = ap.parse_args()

    rows: list[dict] = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        net = make_net(args.width, args.dim, 5)
        P = flat_params(net).numel()
        if P > 3000:
            print(f"seed={seed}: P={P} too big for full FIM ({P**2*8/1e9:.2f} GB). Skipping.")
            continue
        t0 = time.time()
        M = compute_fim_full(net, args.n_samples, args.dim)
        t_full = time.time() - t0
        # Recompute diag separately for true like-for-like
        torch.manual_seed(seed)  # reset so RNG matches
        net2 = make_net(args.width, args.dim, 5)
        t0 = time.time()
        diag = compute_fim_diag(net2, args.n_samples, args.dim)
        t_diag = time.time() - t0

        eigvals = np.linalg.eigvalsh(M)[::-1]  # descending
        eigvals = np.clip(eigvals, 0, None)

        t_diag_stats = tier_stats(diag)
        t_eig_stats = tier_stats(eigvals)
        t_md_stats = tier_stats(np.diag(M))  # diagonal of the same M we eigendecomposed

        print(
            f"seed={seed}  P={P}  "
            f"diag T1/T3={t_diag_stats['ratio_t1_t3']:.1f}  "
            f"eig T1/T3={t_eig_stats['ratio_t1_t3']:.1f}  "
            f"M-diag T1/T3={t_md_stats['ratio_t1_t3']:.1f}  "
            f"(full {t_full:.1f}s, diag {t_diag:.1f}s)"
        )
        rows.append({
            "seed": seed, "P": P,
            "per_sample_diag": t_diag_stats,
            "M_diag": t_md_stats,
            "M_eigvals": t_eig_stats,
            "ratio_diag_over_eig": t_diag_stats['ratio_t1_t3'] / t_eig_stats['ratio_t1_t3'] if t_eig_stats['ratio_t1_t3'] > 0 else float('inf'),
            "time_full_s": t_full, "time_diag_s": t_diag,
        })

    # Aggregate
    if rows:
        diag_ratios = np.array([r['per_sample_diag']['ratio_t1_t3'] for r in rows])
        eig_ratios = np.array([r['M_eigvals']['ratio_t1_t3'] for r in rows])
        md_ratios = np.array([r['M_diag']['ratio_t1_t3'] for r in rows])
        print()
        print(f"Aggregate ({len(rows)} seeds):")
        print(f"  per-sample-diag  T1/T3:  mean={diag_ratios.mean():.1f}  CV={diag_ratios.std(ddof=1)/diag_ratios.mean()*100:.1f}%")
        print(f"  M-diag           T1/T3:  mean={md_ratios.mean():.1f}  CV={md_ratios.std(ddof=1)/md_ratios.mean()*100:.1f}%")
        print(f"  M-eigenvalues    T1/T3:  mean={eig_ratios.mean():.1f}  CV={eig_ratios.std(ddof=1)/eig_ratios.mean()*100:.1f}%")

    payload = {"config": vars(args), "results": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
