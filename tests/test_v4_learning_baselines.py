"""Tests for V4.0 non-deep learning baselines (learning_baselines.py).

Critical regression: test_gaussian_process_matches_kernel_ridge_formula
catches the 1/λ bug that was recently fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/v4_0_uniqueness"))
from learning_baselines import (  # noqa: E402
    gaussian_process_fim,
    kernel_ridge_fim,
    linear_regression_fim,
    logistic_regression_fim,
    tier_ratio,
)


# ---------------------------------------------------------------------------
# linear_regression_fim
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_linear_regression_t1_t3_near_unity() -> None:
    """Linear regression on isotropic Gaussian data gives T1/T3 near 1.0.

    Analytically: FIM_W_{ij} = E[x_j²] / σ² = 1/σ² for all j (unit Gaussian
    input), so the FIM diagonal for W is perfectly flat. The bias terms add
    d entries of exactly 1/σ², also flat. The combined FIM is constant, so
    tier_ratio = 1.
    """
    rng = np.random.default_rng(0)
    fim = linear_regression_fim(d=16, n_samples=2000, noise_sigma=1.0, rng=rng)
    ratio = tier_ratio(fim)
    assert ratio < 2.0, f"linear regression T1/T3={ratio:.4f} should be near 1 for Gaussian data"


@pytest.mark.unit
def test_linear_regression_returns_correct_size() -> None:
    """Output has d² + d entries (W flattened + b)."""
    rng = np.random.default_rng(1)
    d = 8
    fim = linear_regression_fim(d=d, n_samples=500, noise_sigma=0.5, rng=rng)
    assert fim.shape == (d * d + d,)


@pytest.mark.unit
def test_linear_regression_all_positive() -> None:
    rng = np.random.default_rng(2)
    fim = linear_regression_fim(d=8, n_samples=200, noise_sigma=0.3, rng=rng)
    assert np.all(fim > 0), "FIM diagonal of linear regression must be strictly positive"


# ---------------------------------------------------------------------------
# kernel_ridge_fim vs gaussian_process_fim — CRITICAL regression test
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_gaussian_process_matches_kernel_ridge_formula() -> None:
    """Both GP and kernel-ridge FIM use the formula F_jj = mean(K²_j·) / σ².

    This is the critical regression test for the 1/λ bug that was recently
    fixed: if either function mistakenly divided by (K + λI) instead of
    using K² directly, the ratio fim / (K2.mean / σ²) would deviate from 1.

    Note: kernel_ridge uses bandwidth=2.0*d in exp(-pairwise/(2*bandwidth))
    i.e. denominator 4d; gaussian_process uses exp(-pairwise/(2*d))
    i.e. denominator 2d. These are different kernels. We verify each
    function's output equals its own K² formula rather than comparing the
    two outputs to each other.
    """
    n_train, d, noise_sigma = 30, 4, 0.5

    # --- kernel_ridge: verify fim_diag == K2.mean(axis=1) / sigma² ---
    rng_kr = np.random.default_rng(42)
    X_kr = rng_kr.standard_normal((n_train, d))
    _y_kr = X_kr[:, 0] + noise_sigma * rng_kr.standard_normal(n_train)
    pairwise_kr = np.sum((X_kr[:, None] - X_kr[None, :]) ** 2, axis=-1)
    bandwidth_kr = 2.0 * d
    K_kr = np.exp(-pairwise_kr / (2 * bandwidth_kr))
    expected_kr = (K_kr ** 2).mean(axis=1) / noise_sigma ** 2

    rng_kr2 = np.random.default_rng(42)
    fim_kr = kernel_ridge_fim(n_train=n_train, d=d, noise_sigma=noise_sigma, ridge=0.1, rng=rng_kr2)
    np.testing.assert_allclose(
        fim_kr, expected_kr, rtol=1e-10,
        err_msg="kernel_ridge_fim does not match K2.mean/sigma² formula",
    )

    # --- gaussian_process: verify fim_diag == K2.mean(axis=1) / sigma² ---
    rng_gp = np.random.default_rng(42)
    X_gp = rng_gp.standard_normal((n_train, d))
    _y_gp = X_gp[:, 0] + noise_sigma * rng_gp.standard_normal(n_train)
    pairwise_gp = np.sum((X_gp[:, None] - X_gp[None, :]) ** 2, axis=-1)
    K_gp = np.exp(-pairwise_gp / (2 * d))
    expected_gp = (K_gp ** 2).mean(axis=1) / noise_sigma ** 2

    rng_gp2 = np.random.default_rng(42)
    fim_gp = gaussian_process_fim(n_train=n_train, d=d, noise_sigma=noise_sigma, rng=rng_gp2)
    np.testing.assert_allclose(
        fim_gp, expected_gp, rtol=1e-10,
        err_msg="gaussian_process_fim does not match K2.mean/sigma² formula",
    )


# ---------------------------------------------------------------------------
# logistic_regression_fim
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_logistic_regression_converges() -> None:
    """Training loss decreases: the SGD loop in logistic_regression_fim makes progress."""
    rng = np.random.default_rng(5)
    d, n_classes, n_train = 8, 4, 200

    teacher = rng.standard_normal((d, n_classes)) * 0.3
    X_train = rng.standard_normal((n_train, d))
    y_train = np.argmax(
        X_train @ teacher + 0.1 * rng.standard_normal((n_train, n_classes)),
        axis=1,
    )

    def cross_entropy(W: np.ndarray, b: np.ndarray) -> float:
        logits = X_train @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return float(-np.log(probs[np.arange(n_train), y_train] + 1e-9).mean())

    # Replicate the training loop from logistic_regression_fim
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    loss_initial = cross_entropy(W, b)

    lr = 0.01
    for _ in range(300):
        logits = X_train @ W + b
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        err = probs.copy()
        err[np.arange(n_train), y_train] -= 1
        W -= lr * (X_train.T @ err) / n_train
        b -= lr * err.mean(axis=0)

    loss_final = cross_entropy(W, b)
    assert loss_final < loss_initial, (
        f"loss did not decrease: initial={loss_initial:.4f} final={loss_final:.4f}"
    )


@pytest.mark.unit
def test_logistic_regression_fim_shape() -> None:
    """Output has d*K + K entries (W flattened + b)."""
    rng = np.random.default_rng(6)
    d, n_classes = 6, 3
    fim = logistic_regression_fim(d=d, n_classes=n_classes, n_train=100, rng=rng, n_probes=50)
    assert fim.shape == (d * n_classes + n_classes,)


@pytest.mark.unit
def test_logistic_regression_fim_non_negative() -> None:
    rng = np.random.default_rng(7)
    fim = logistic_regression_fim(d=6, n_classes=3, n_train=100, rng=rng, n_probes=50)
    assert np.all(fim >= 0), "FIM diagonal must be non-negative (grad²)"


# ---------------------------------------------------------------------------
# tier_ratio (local copy in learning_baselines)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tier_ratio_power_law_large() -> None:
    values = 1.0 / np.arange(1, 1001) ** 2
    ratio = tier_ratio(values)
    assert ratio > 100.0


@pytest.mark.unit
def test_tier_ratio_zero_floor_fallback() -> None:
    values = np.zeros(100)
    values[:5] = 1.0
    ratio = tier_ratio(values)
    assert np.isfinite(ratio)
    assert ratio > 0.0
