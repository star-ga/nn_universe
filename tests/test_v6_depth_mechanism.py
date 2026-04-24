"""Tests for V6.0 depth-sweep / Hanin-Nica mechanism experiment.

Covers: make_net, fim_diagonal, tier_ratio, log_stats, fit_linear.
All tests are fast (<1s) and deterministic under fixed seeds.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/v6_0_depth_mechanism"))
from depth_sweep import (  # noqa: E402
    fim_diagonal,
    fit_linear,
    log_stats,
    make_net,
    tier_ratio,
)


# ---------------------------------------------------------------------------
# make_net
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_make_net_param_count() -> None:
    """Expected parameter counts for depths 2/3/4 with width=8, dim=4."""
    dim, width = 4, 8
    # depth=2: Linear(4,8) + Linear(8,4)  → 4*8+8 + 8*4+4 = 40+36 = 76
    net2 = make_net(2, width, dim)
    assert sum(p.numel() for p in net2.parameters()) == dim * width + width + width * dim + dim

    # depth=3: Linear(4,8) + Linear(8,8) + Linear(8,4)  → 40+72+36 = 148
    net3 = make_net(3, width, dim)
    expected3 = (dim * width + width) + (width * width + width) + (width * dim + dim)
    assert sum(p.numel() for p in net3.parameters()) == expected3

    # depth=4: adds one more hidden layer
    net4 = make_net(4, width, dim)
    expected4 = expected3 + width * width + width
    assert sum(p.numel() for p in net4.parameters()) == expected4


@pytest.mark.unit
def test_make_net_raises_on_shallow() -> None:
    with pytest.raises(ValueError, match="depth must be >= 2"):
        make_net(1, 8, 4)


# ---------------------------------------------------------------------------
# fim_diagonal
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fim_diagonal_shape() -> None:
    """Output length equals total parameter count of the network."""
    torch.manual_seed(0)
    dim, width, depth = 4, 8, 3
    net = make_net(depth, width, dim)
    n_params = sum(p.numel() for p in net.parameters())
    fim = fim_diagonal(net, dim, n_probes=20)
    assert fim.shape == (n_params,)


@pytest.mark.unit
def test_fim_diagonal_positive() -> None:
    """All FIM diagonal entries are >= 0 (they are squared gradients)."""
    torch.manual_seed(1)
    net = make_net(3, 8, 4)
    fim = fim_diagonal(net, dim=4, n_probes=30)
    assert np.all(fim >= 0), "FIM diagonal must be non-negative (grad²)"


# ---------------------------------------------------------------------------
# tier_ratio
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tier_ratio_t3_zero_fallback() -> None:
    """values=[1,1,1,1,0,0,0,0] must give a finite ratio via underflow fallback."""
    values = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    t1, t3, ratio = tier_ratio(values)
    assert np.isfinite(ratio), "ratio must be finite even when bot half is all zeros"
    assert ratio > 0.0


@pytest.mark.unit
def test_tier_ratio_flat_distribution() -> None:
    """Uniform values give ratio near 1.0."""
    values = np.ones(100)
    t1, t3, ratio = tier_ratio(values)
    assert abs(ratio - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# fit_linear
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fit_linear_perfect_fit() -> None:
    """y = 2x + 3 recovers slope=2, intercept=3, R²=1.0."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [3.0, 5.0, 7.0, 9.0, 11.0]
    slope, intercept, r2 = fit_linear(xs, ys)
    assert abs(slope - 2.0) < 1e-10
    assert abs(intercept - 3.0) < 1e-10
    assert abs(r2 - 1.0) < 1e-10


@pytest.mark.unit
def test_fit_linear_single_point_returns_nan() -> None:
    slope, intercept, r2 = fit_linear([1.0], [2.0])
    assert math.isnan(slope)
    assert math.isnan(r2)


@pytest.mark.unit
def test_fit_linear_constant_x_returns_nan() -> None:
    """Degenerate case: all x equal → division by zero → NaN."""
    slope, intercept, r2 = fit_linear([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])
    assert math.isnan(slope)


# ---------------------------------------------------------------------------
# log_stats
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_log_stats_on_gaussian() -> None:
    """log-normal samples give skew ≈ 0 and excess kurtosis ≈ 0 for large N.

    We draw from N(0,1) and exponentiate → log-normal. The log of those
    values is exactly N(0,1), which has skew=0, excess_kurtosis=0.
    """
    rng = np.random.default_rng(42)
    N = 50_000
    samples = np.exp(rng.standard_normal(N))  # log(samples) ~ N(0,1)
    stats = log_stats(samples)
    assert abs(stats["skew"]) < 0.1, f"skew={stats['skew']:.4f} unexpectedly large"
    assert abs(stats["excess_kurtosis"]) < 0.2, f"excess_kurtosis={stats['excess_kurtosis']:.4f}"


@pytest.mark.unit
def test_log_stats_too_few_nonzero_returns_nan() -> None:
    """Fewer than 10 nonzero entries gives NaN moments."""
    values = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stats = log_stats(values[:9])  # only 3 nonzero out of 9 — well under 10
    assert math.isnan(stats["skew"])


# ---------------------------------------------------------------------------
# depth_ratio_monotone — central scientific sanity check
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_depth_ratio_monotone() -> None:
    """For a fixed seed, T1/T3 at depth=8 > T1/T3 at depth=2.

    This is the qualitative core of the Hanin-Nica mechanism: deeper
    networks have more pronounced FIM hierarchies.
    """
    torch.manual_seed(7)
    net2 = make_net(2, width=32, dim=8)
    fim2 = fim_diagonal(net2, dim=8, n_probes=100)
    _, _, ratio2 = tier_ratio(fim2)

    torch.manual_seed(7)
    net8 = make_net(8, width=32, dim=8)
    fim8 = fim_diagonal(net8, dim=8, n_probes=100)
    _, _, ratio8 = tier_ratio(fim8)

    assert ratio8 > ratio2, (
        f"T1/T3 should increase with depth: depth=8 gave {ratio8:.2f}, "
        f"depth=2 gave {ratio2:.2f}"
    )
