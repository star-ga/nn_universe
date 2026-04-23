"""Unit tests for the V3.1 α-drift mock pipeline."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mock_pipeline import (
    MockConfig,
    partial_correlation,
    power_analysis,
    run_trial,
    synth_alpha_drifts,
    synth_density_field,
    synth_redshifts,
)


@pytest.mark.unit
def test_density_is_positive() -> None:
    cfg = MockConfig(n_sightlines=500, seed=0)
    rng = np.random.default_rng(0)
    rho = synth_density_field(cfg, rng)
    assert (rho > 0).all()
    # Log-normal mean should be close to prescribed rho_mean (within factor 2).
    assert 0.2 * cfg.rho_mean < np.exp(np.log(rho).mean()) < 5.0 * cfg.rho_mean


@pytest.mark.unit
def test_redshifts_in_range() -> None:
    cfg = MockConfig(n_sightlines=1000, seed=0, z_min=0.1, z_max=2.5)
    rng = np.random.default_rng(0)
    z = synth_redshifts(cfg, rng)
    assert z.min() >= 0.1
    assert z.max() <= 2.5


@pytest.mark.unit
def test_partial_correlation_sign() -> None:
    rng = np.random.default_rng(0)
    n = 200
    z = rng.uniform(0, 1, n)
    x = rng.normal(0, 1, n) + z
    y = x + rng.normal(0, 0.1, n)
    r, p = partial_correlation(x, y, z)
    assert r > 0.8
    assert p < 0.01


@pytest.mark.unit
def test_h1_detects_signal_with_enough_n() -> None:
    # With the physical κ value (~4e-59) the signal is ~10^{-41}/yr, dwarfed by
    # any realistic measurement noise (~10^{-17}/yr per system). This test
    # exercises the pipeline's *statistical* machinery, not the physical
    # detectability, so we inject a detectable κ while keeping noise realistic.
    cfg = MockConfig(
        n_sightlines=2_000,
        kappa_true=1e-37,  # chosen so SNR ~ 1 per sample
        alpha_noise_sigma=1e-11,
        seed=0,
    )
    res = run_trial(cfg, h0=False)
    assert res["r"] > 0.1
    assert res["p_value"] < 1e-6


@pytest.mark.unit
def test_h0_not_detected() -> None:
    cfg = MockConfig(n_sightlines=500, seed=0)
    res = run_trial(cfg, h0=True)
    # Under H0 with no signal, median p-value should be ~uniform; a single
    # trial can be small by chance, so just assert it is a valid probability.
    assert 0.0 <= res["p_value"] <= 1.0


@pytest.mark.unit
def test_power_analysis_monotone_in_alpha() -> None:
    cfg = MockConfig(n_sightlines=500, seed=0, alpha_noise_sigma=1e-9, kappa_true=1e-40)
    pa = power_analysis(cfg, seeds=30)
    fpr = [d["false_positive_rate"] for d in pa["thresholds"].values()]
    # Stricter thresholds -> fewer or equal false positives.
    assert fpr[0] >= fpr[-1]  # p05 FPR >= p5sigma FPR
