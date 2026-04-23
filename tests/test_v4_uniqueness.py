"""Tests for V4.0 uniqueness experiment.

Sanity-only — does not validate the scientific verdict, which requires a
production run. These tests only check that each baseline's importance
vector has the right shape and is nonzero, that the tier analysis is
deterministic under fixed seed, and that the classifier code path works
end-to-end on a small synthetic example.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines import REGISTRY, make
from run_uniqueness import RATIO_CAP, tier_mass, tier_ratio
from analyze import analyze, load_features, simple_logistic


@pytest.mark.parametrize("name", [spec.name for spec in REGISTRY])
def test_baseline_importance_shape(name: str) -> None:
    """Each baseline returns a nonzero 1D float array."""
    sys = make(name, seed=0)
    imp = sys.parameter_importance(n_probes=4)
    assert imp.ndim == 1
    assert imp.size > 0
    assert np.isfinite(imp).all()
    # Must have some positive mass (degenerate zero-arrays are a bug).
    assert float(imp.sum()) > 0.0


@pytest.mark.parametrize("name", [spec.name for spec in REGISTRY])
def test_baseline_seed_determinism(name: str) -> None:
    """Two independent constructions with the same seed give identical importance.

    Note: NN uses torch.manual_seed(seed) inside its __init__, so strict
    determinism requires we don't interleave other torch calls between the
    two constructions. This test runs them sequentially which is the
    expected usage.
    """
    a = make(name, seed=42).parameter_importance(n_probes=4)
    b = make(name, seed=42).parameter_importance(n_probes=4)
    # Allow small tolerance for systems with floating-point reduction order
    # variability (NN in particular)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_tier_ratio_basic() -> None:
    """Tier ratio is large on a power-law distribution, small on uniform."""
    x_heavy = 1.0 / (np.arange(1, 1001) ** 2)
    x_flat = np.ones(1000)
    t1_h, t3_h, r_h = tier_ratio(x_heavy)
    t1_f, t3_f, r_f = tier_ratio(x_flat)
    assert r_h > 100.0, f"heavy-tail ratio {r_h} should be >100"
    assert r_f < 2.0, f"flat ratio {r_f} should be <2"
    assert t1_h > t3_h
    assert abs(t1_f - t3_f) < 1e-6


def test_tier_ratio_degenerate_floor() -> None:
    """Tier ratio is capped at RATIO_CAP for ill-conditioned arrays."""
    x = np.zeros(1000, dtype=np.float64)
    x[:5] = 1.0
    _, _, r = tier_ratio(x)
    assert r <= RATIO_CAP + 1e-6
    assert np.isfinite(r)


def test_tier_mass_top1pct() -> None:
    """Top-1% mass is high on power-law, low on uniform."""
    x_heavy = 1.0 / np.arange(1, 1001) ** 2
    x_flat = np.ones(1000)
    assert tier_mass(x_heavy) > 0.5
    assert tier_mass(x_flat) < 0.05


def test_simple_logistic_converges() -> None:
    """Logistic regression on linearly-separable 2D data reaches 100% train acc."""
    rng = np.random.default_rng(0)
    n_per = 40
    X_neg = rng.standard_normal((n_per, 2)) - 2.0
    X_pos = rng.standard_normal((n_per, 2)) + 2.0
    X = np.vstack([X_neg, X_pos])
    y = np.hstack([np.zeros(n_per), np.ones(n_per)])
    coef, bias = simple_logistic(X, y)
    pred = (X @ coef + bias > 0).astype(int)
    acc = float((pred == y).mean())
    assert acc > 0.95


def test_analyze_returns_verdict_keys() -> None:
    """analyze() returns a dict with expected keys on a synthetic mini-result."""
    # Build a minimal synthetic results dict with NN clearly separable.
    results = {
        "baselines": {
            "neural_network": {
                "n_params": 100,
                "per_seed": [
                    {"seed": s, "tier_ratio": 10000.0 + s, "top1pct_mass": 0.8}
                    for s in range(3)
                ],
                "tier_ratio_mean": 10001.0,
                "tier_ratio_std": 1.0,
                "tier_ratio_cv": 0.0001,
                "top1pct_mass_mean": 0.8,
                "top1pct_mass_cv": 0.0,
            },
            "random_matrix": {
                "n_params": 100,
                "per_seed": [
                    {"seed": s, "tier_ratio": 50.0 + s * 5, "top1pct_mass": 0.1}
                    for s in range(3)
                ],
                "tier_ratio_mean": 55.0,
                "tier_ratio_std": 5.0,
                "tier_ratio_cv": 0.09,
                "top1pct_mass_mean": 0.1,
                "top1pct_mass_cv": 0.0,
            },
        }
    }
    v = analyze(results)
    expected = {
        "nn_mean_ratio", "nn_cv", "max_other_ratio", "magnitude_z",
        "magnitude_pass", "stability_pass", "classifier_loo_accuracy",
        "classifier_pass", "verdict",
    }
    assert expected.issubset(v.keys())
    # With the synthetic data above, verdict should be NN_UNIQUE
    assert v["verdict"] == "NN_UNIQUE"
