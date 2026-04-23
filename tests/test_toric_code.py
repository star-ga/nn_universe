"""Unit tests for the toric-code syndrome generator."""
from __future__ import annotations

import numpy as np
import pytest

from toric_code import (
    compute_syndromes,
    logical_error_rate,
    parity_check_matrix,
    sample_errors,
)


@pytest.mark.unit
@pytest.mark.parametrize("L", [3, 5, 7])
def test_parity_check_shape(L: int) -> None:
    H = parity_check_matrix(L)
    assert H.shape == (L * L, 2 * L * L)
    # Each plaquette touches exactly 4 edges.
    assert np.all(H.sum(axis=1) == 4)


@pytest.mark.unit
@pytest.mark.parametrize("L", [3, 5])
def test_each_edge_appears_in_two_plaquettes(L: int) -> None:
    H = parity_check_matrix(L)
    # On a torus every edge is shared by exactly two plaquettes.
    assert np.all(H.sum(axis=0) == 2)


@pytest.mark.unit
def test_zero_error_zero_syndrome() -> None:
    L = 5
    H = parity_check_matrix(L)
    n_qubits = 2 * L * L
    e = np.zeros((10, n_qubits), dtype=np.int8)
    s = compute_syndromes(e, H)
    assert s.shape == (10, L * L)
    assert np.all(s == 0)


@pytest.mark.unit
def test_single_error_triggers_two_stabilizers() -> None:
    L = 4
    H = parity_check_matrix(L)
    n_qubits = 2 * L * L
    for edge in range(n_qubits):
        e = np.zeros((1, n_qubits), dtype=np.int8)
        e[0, edge] = 1
        s = compute_syndromes(e, H)
        assert s.sum() == 2, f"edge {edge} triggered {s.sum()} stabilizers (expected 2)"


@pytest.mark.unit
def test_sample_errors_rate() -> None:
    rng = np.random.default_rng(0)
    e = sample_errors(batch=10_000, L=4, p=0.1, rng=rng)
    rate = e.mean()
    assert 0.08 < rate < 0.12, f"empirical error rate {rate} far from p=0.1"


@pytest.mark.unit
def test_logical_error_rate_zero_on_perfect_predictions() -> None:
    L = 3
    H = parity_check_matrix(L)
    rng = np.random.default_rng(0)
    e = sample_errors(batch=100, L=L, p=0.05, rng=rng).astype(np.float32)
    # Perfect predictions -> zero residual syndrome.
    rate = logical_error_rate(pred=e, true=e, H=H, L=L)
    assert rate == 0.0
