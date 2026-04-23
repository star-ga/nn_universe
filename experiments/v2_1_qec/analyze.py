"""Spectral and FIM-tier analysis for V2.1 QEC decoders.

Replicates the V1.0 cosmology-experiment measurements on a trained decoder:

- Per-weight-layer singular-value ratios  (`sv_stats`)
- FIM diagonal spectrum + 3-tier partition (top 1% / 1-50% / bottom 50%)
- Summary statistics suitable for direct comparison with
  `/home/n/nn_universe/toy_experiment_results.json`.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def sv_per_layer(net: nn.Module) -> list[dict]:
    stats: list[dict] = []
    for name, param in net.named_parameters():
        if "weight" in name and param.dim() == 2:
            S = torch.linalg.svdvals(param.data.float())
            top3 = S[:3].tolist()
            ratio = float(S[0] / S[-1]) if S[-1] > 1e-10 else float("inf")
            stats.append(
                {
                    "layer": name,
                    "shape": list(param.shape),
                    "top3_sv": [round(s, 4) for s in top3],
                    "sv_ratio": round(ratio, 2),
                    "sv_std": round(float(S.std()), 4),
                }
            )
    return stats


def max_sv_ratio(net: nn.Module) -> float:
    return max((s["sv_ratio"] for s in sv_per_layer(net)), default=0.0)


def fim_diagonal(
    net: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    n_samples: int = 1000,
) -> dict[str, torch.Tensor]:
    """Empirical FIM diagonal: E[(∂_θ loss)^2] with per-sample batches.

    `loss_fn` should return a scalar loss computed on a *single fresh sample*
    per call (caller is responsible for sampling).
    """
    fim: dict[str, torch.Tensor] = {
        n: torch.zeros_like(p, dtype=torch.float32) for n, p in net.named_parameters()
    }
    for _ in range(n_samples):
        loss = loss_fn()
        net.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in net.named_parameters():
            if p.grad is not None:
                fim[name] += p.grad.data.float() ** 2
    for name in fim:
        fim[name] /= n_samples
    return fim


def tier_partition(fim: dict[str, torch.Tensor]) -> dict:
    """Partition FIM diagonal into 3 tiers (top 1% / 1-50% / bottom 50%)."""
    all_fim = torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()
    sorted_fim = np.sort(all_fim)[::-1]
    total = len(sorted_fim)
    t1_thresh = np.percentile(sorted_fim, 99)
    t2_thresh = np.percentile(sorted_fim, 50)
    n_t1 = int(np.sum(sorted_fim >= t1_thresh))
    n_t2 = int(np.sum((sorted_fim >= t2_thresh) & (sorted_fim < t1_thresh)))
    n_t3 = total - n_t1 - n_t2
    t1_mean = float(np.mean(sorted_fim[:n_t1])) if n_t1 > 0 else 0.0
    t2_mean = float(np.mean(sorted_fim[n_t1 : n_t1 + n_t2])) if n_t2 > 0 else 0.0
    t3_mean = float(np.mean(sorted_fim[n_t1 + n_t2 :])) if n_t3 > 0 else 0.0
    # Guard: at large scales the smallest FIM values can underflow; fall back to
    # the smallest observed non-zero value to keep the ratio comparable across sizes.
    if t3_mean <= 0.0:
        nonzero = sorted_fim[sorted_fim > 0.0]
        t3_mean = float(np.mean(nonzero[-max(len(nonzero) // 10, 1) :])) if len(nonzero) else 1e-30
    ratio = t1_mean / t3_mean if t3_mean > 0 else float("inf")
    return {
        "total_params": total,
        "tier1": {"count": n_t1, "mean": t1_mean},
        "tier2": {"count": n_t2, "mean": t2_mean},
        "tier3": {"count": n_t3, "mean": t3_mean},
        "ratio_tier1_tier3": ratio,
        "top10": [float(v) for v in sorted_fim[:10]],
        "spectrum_percentiles": {
            "p99": float(np.percentile(sorted_fim, 99)),
            "p95": float(np.percentile(sorted_fim, 95)),
            "p50": float(np.percentile(sorted_fim, 50)),
            "p10": float(np.percentile(sorted_fim, 10)),
            "p1": float(np.percentile(sorted_fim, 1)),
        },
    }
