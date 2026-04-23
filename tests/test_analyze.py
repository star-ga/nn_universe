"""Unit tests for the V2.1 spectral / FIM analyzers."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from analyze import fim_diagonal, max_sv_ratio, sv_per_layer, tier_partition


@pytest.mark.unit
def test_sv_per_layer_shape() -> None:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 4))
    stats = sv_per_layer(net)
    # Two Linear weights (biases are excluded).
    assert len(stats) == 2
    for s in stats:
        assert {"layer", "shape", "top3_sv", "sv_ratio", "sv_std"} <= set(s.keys())
        assert s["sv_ratio"] >= 1.0 - 1e-6


@pytest.mark.unit
def test_max_sv_ratio_positive() -> None:
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(5, 5))
    assert max_sv_ratio(net) >= 1.0 - 1e-6


@pytest.mark.unit
def test_tier_partition_sums() -> None:
    # Hand-constructed: 100 values with 1% top tier, 49% mid, 50% bottom.
    vals = torch.arange(100, dtype=torch.float32)
    fim = {"p": vals}
    t = tier_partition(fim)
    assert t["total_params"] == 100
    assert t["tier1"]["count"] + t["tier2"]["count"] + t["tier3"]["count"] == 100
    # Tier-1 mean should exceed Tier-3 mean.
    assert t["tier1"]["mean"] > t["tier3"]["mean"]
    assert t["ratio_tier1_tier3"] > 1


@pytest.mark.unit
def test_fim_diagonal_scales_with_samples() -> None:
    torch.manual_seed(2)
    net = nn.Linear(3, 1)

    def loss_fn():
        x = torch.randn(1, 3)
        return 0.5 * (net(x)).pow(2).mean()

    fim_small = fim_diagonal(net, loss_fn, n_samples=20)
    fim_large = fim_diagonal(net, loss_fn, n_samples=200)
    # Empirical Fisher average should be ~ same order of magnitude; we only
    # check that values are non-negative and shapes match.
    for name, v in fim_small.items():
        assert v.shape == fim_large[name].shape
        assert (v >= 0).all()
        assert (fim_large[name] >= 0).all()
