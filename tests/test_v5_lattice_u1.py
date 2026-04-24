"""Tests for V5.0 U(1) pure-gauge lattice Monte Carlo experiment.

Critical regression: test_grad_action_finite_difference catches the sign
bug that was recently fixed in grad_action().
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/v5_0_lattice_qcd"))
from lattice_u1 import (  # noqa: E402
    action,
    grad_action,
    init_links,
    metropolis_sweep_local,
    plaquette_sum,
    tier_ratio,
)


# ---------------------------------------------------------------------------
# plaquette_sum
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_action_plaquette_antisymmetry() -> None:
    """P(x,mu,nu) = -P(x,nu,mu) for all sites and all plane pairs."""
    rng = np.random.default_rng(0)
    L, d = 4, 3
    theta = init_links(L, d, rng)
    for mu in range(d):
        for nu in range(d):
            if mu == nu:
                continue
            P_mn = plaquette_sum(theta, mu, nu)
            P_nm = plaquette_sum(theta, nu, mu)
            np.testing.assert_allclose(
                P_mn, -P_nm, atol=1e-12,
                err_msg=f"antisymmetry failed for mu={mu}, nu={nu}",
            )


# ---------------------------------------------------------------------------
# action
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_action_translation_invariant() -> None:
    """Shifting every link phase by a global constant leaves action unchanged (mod 2π).

    The shift cancels in each oriented plaquette sum theta_P = theta(x,mu)
    + theta(x+mu,nu) - theta(x+nu,mu) - theta(x,nu), so cos(theta_P) and
    hence S are invariant.
    """
    rng = np.random.default_rng(1)
    L, d, beta = 4, 3, 1.0
    theta = init_links(L, d, rng)
    S_orig = action(theta, beta, d)
    # Shift by an arbitrary constant that does NOT wrap around 2π alone
    delta = 0.3
    theta_shifted = theta + delta
    S_shifted = action(theta_shifted, beta, d)
    assert abs(S_orig - S_shifted) < 1e-8, (
        f"action should be translation-invariant: {S_orig} vs {S_shifted}"
    )


@pytest.mark.unit
def test_action_negative_at_cold_start() -> None:
    """Cold start (all phases = 0) gives action = -beta * n_plaquettes."""
    L, d, beta = 4, 3, 2.0
    theta = np.zeros((L,) * d + (d,))
    n_planes = d * (d - 1) // 2
    n_sites = L ** d
    expected = -beta * n_planes * n_sites
    assert abs(action(theta, beta, d) - expected) < 1e-8


# ---------------------------------------------------------------------------
# grad_action — CRITICAL regression test
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_grad_action_finite_difference() -> None:
    """Analytic gradient agrees with numerical finite-difference gradient to 1e-6.

    This is the critical regression test that catches the sign bug that was
    recently fixed: if grad_action returns +beta*... instead of the correct
    sign, the numerical check fails.
    """
    rng = np.random.default_rng(2)
    L, d, beta = 3, 3, 1.5
    theta = init_links(L, d, rng)
    eps = 1e-5

    analytic = grad_action(theta, beta, d)
    numerical = np.zeros_like(theta)

    # Sample a subset of links to keep the test fast (<0.5s for L=3, d=3)
    rng2 = np.random.default_rng(99)
    flat_indices = rng2.choice(theta.size, size=40, replace=False)
    multi = np.unravel_index(flat_indices, theta.shape)

    for coords in zip(*multi):
        idx = coords
        theta_plus = theta.copy()
        theta_plus[idx] += eps
        theta_minus = theta.copy()
        theta_minus[idx] -= eps
        numerical[idx] = (action(theta_plus, beta, d) - action(theta_minus, beta, d)) / (2 * eps)

    # Only compare the sampled links
    for coords in zip(*multi):
        idx = coords
        assert abs(analytic[idx] - numerical[idx]) < 1e-6, (
            f"Gradient mismatch at {idx}: analytic={analytic[idx]:.8f} "
            f"numerical={numerical[idx]:.8f}"
        )


# ---------------------------------------------------------------------------
# tier_ratio
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tier_ratio_t3_zero_fallback() -> None:
    """values=[1,1,1,1,0,0,0,0] gives a finite ratio via the underflow fallback."""
    values = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    t1, t3, ratio = tier_ratio(values)
    assert math.isfinite(ratio), "ratio must be finite even when bottom half is all zeros"
    assert ratio > 0.0


@pytest.mark.unit
def test_tier_ratio_all_equal() -> None:
    """Uniform values give ratio = 1.0."""
    values = np.ones(200)
    t1, t3, ratio = tier_ratio(values)
    assert abs(ratio - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# metropolis_sweep_local — detailed balance sanity
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_metropolis_detailed_balance() -> None:
    """After thermalization sweeps, the distribution of plaquette cos values
    is stable: the mean of <cos P> does not drift between two halves of
    additional sweeps.

    This is a coarse stationarity check, not a strict reversibility proof.
    Uses a tiny L=3, d=2 lattice for speed.
    """
    rng = np.random.default_rng(3)
    L, d, beta, step = 3, 2, 1.0, 0.5
    theta = init_links(L, d, rng)

    # Thermalise
    for _ in range(30):
        metropolis_sweep_local(theta, beta, d, step, rng)

    # Collect cos-plaquette means from two equal-length windows
    def mean_cos_plaq(th: np.ndarray) -> float:
        total = 0.0
        count = 0
        for mu in range(d):
            for nu in range(mu + 1, d):
                P = plaquette_sum(th, mu, nu)
                total += float(np.cos(P).mean())
                count += 1
        return total / count if count else 0.0

    n_meas = 20
    first_half, second_half = [], []
    for i in range(2 * n_meas):
        metropolis_sweep_local(theta, beta, d, step, rng)
        val = mean_cos_plaq(theta)
        (first_half if i < n_meas else second_half).append(val)

    mean_first = float(np.mean(first_half))
    mean_second = float(np.mean(second_half))

    # If the chain has reached stationarity the two windows agree within noise
    assert abs(mean_first - mean_second) < 0.3, (
        f"Chain appears non-stationary: first={mean_first:.4f}, second={mean_second:.4f}"
    )
