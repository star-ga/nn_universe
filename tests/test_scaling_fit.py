"""Unit tests for the V1.2 power-law refit logic."""
from __future__ import annotations

import numpy as np
import pytest

from scaling_experiment_extended import _refit_power_law


@pytest.mark.unit
def test_refit_returns_zero_for_empty() -> None:
    sv, fim = _refit_power_law([])
    assert sv == {"exponent": 0.0, "r_squared": 0.0}
    assert fim == {"exponent": 0.0, "r_squared": 0.0}


@pytest.mark.unit
def test_refit_recovers_exact_power_law() -> None:
    # Synthesize data: SV = 10 * N^0.5, FIM = 100 * N^0.3
    params = np.logspace(3, 10, 8)
    rows = [
        {
            "width": int(p ** 0.5),
            "params": int(p),
            "max_sv_ratio": 10 * p ** 0.5,
            "fim_tier1_tier3": 100 * p ** 0.3,
        }
        for p in params
    ]
    sv, fim = _refit_power_law(rows)
    assert abs(sv["exponent"] - 0.5) < 1e-2
    assert sv["r_squared"] > 0.999
    assert abs(fim["exponent"] - 0.3) < 1e-2
    assert fim["r_squared"] > 0.999


@pytest.mark.unit
def test_refit_r_squared_bounded() -> None:
    rng = np.random.default_rng(0)
    params = np.logspace(3, 10, 8)
    rows = [
        {
            "width": int(p ** 0.5),
            "params": int(p),
            "max_sv_ratio": 10 * p ** 0.5 * (1 + 0.2 * rng.standard_normal()),
            "fim_tier1_tier3": 100 * p ** 0.3 * (1 + 0.2 * rng.standard_normal()),
        }
        for p in params
    ]
    sv, fim = _refit_power_law(rows)
    assert -1.0 <= sv["r_squared"] <= 1.0
    assert -1.0 <= fim["r_squared"] <= 1.0
