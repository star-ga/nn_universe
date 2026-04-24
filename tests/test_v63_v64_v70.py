"""Regression tests for V6.3 (boolean circuit), V6.4 (transformer), V7.0 (SU(2) lattice).

All tests are unit-level, deterministic under fixed seeds, and run in < 1 second each.
They guard the scientific correctness of the three uncovered modules without testing
implementation details.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — mirror the pattern used in test_v6_depth_mechanism.py
# ---------------------------------------------------------------------------
_EXP = Path(__file__).resolve().parents[1] / "experiments"
_V6 = str(_EXP / "v6_0_depth_mechanism")
_V7 = str(_EXP / "v7_0_lattice_su2")
for _p in (_V6, _V7):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bc_depth_sweep import LayeredBooleanCircuit  # noqa: E402
from lattice_su2 import (  # noqa: E402
    action,
    build_links,
    links_to_matrices,
    plaquette_trace,
    su2_from_alpha,
)
from transformer_depth_sweep import Transformer, fim_diagonal_transformer  # noqa: E402

# ---------------------------------------------------------------------------
# V6.3 — LayeredBooleanCircuit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bc_forward_batch_deterministic_fixed_seed() -> None:
    """_forward_batch is pure given fixed weights and wiring (seed-locked)."""
    bc = LayeredBooleanCircuit(depth=2, gates_per_layer=4, n_inputs=4, seed=0)
    rng = np.random.default_rng(99)
    xs = rng.uniform(0.0, 1.0, size=(8, 4))
    out1 = bc._forward_batch(xs, bc.W)
    out2 = bc._forward_batch(xs, bc.W)
    np.testing.assert_array_equal(out1, out2)


@pytest.mark.unit
def test_bc_forward_batch_output_shape() -> None:
    """_forward_batch returns (B,) for any batch size B."""
    bc = LayeredBooleanCircuit(depth=3, gates_per_layer=4, n_inputs=4, seed=1)
    rng = np.random.default_rng(0)
    for B in (1, 5, 16):
        xs = rng.uniform(0.0, 1.0, size=(B, 4))
        out = bc._forward_batch(xs, bc.W)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


@pytest.mark.unit
def test_bc_forward_batch_output_range() -> None:
    """Softmax-mixture of AND/OR/XOR on [0,1] inputs stays in [0,1]."""
    bc = LayeredBooleanCircuit(depth=4, gates_per_layer=8, n_inputs=4, seed=2)
    rng = np.random.default_rng(7)
    xs = rng.uniform(0.0, 1.0, size=(50, 4))
    out = bc._forward_batch(xs, bc.W)
    assert np.all(out >= -1e-9), "output below 0"
    assert np.all(out <= 1.0 + 1e-9), "output above 1"


@pytest.mark.unit
def test_bc_fim_diagonal_shape_equals_n_params() -> None:
    """fim_diagonal length must equal n_params = depth * gpl * 3."""
    depth, gpl = 2, 6
    bc = LayeredBooleanCircuit(depth=depth, gates_per_layer=gpl, n_inputs=4, seed=3)
    expected = bc.n_params()
    assert expected == depth * gpl * 3
    fim = bc.fim_diagonal(n_probes=20)
    assert fim.shape == (expected,), f"Expected ({expected},), got {fim.shape}"


@pytest.mark.unit
def test_bc_fim_diagonal_non_negative() -> None:
    """FIM diagonal entries are squared gradients — must be >= 0."""
    bc = LayeredBooleanCircuit(depth=2, gates_per_layer=4, n_inputs=4, seed=4)
    fim = bc.fim_diagonal(n_probes=20)
    assert np.all(fim >= 0), "FIM entries must be non-negative"


@pytest.mark.unit
def test_bc_strict_layering_wires_in_bounds() -> None:
    """Every wire at layer k must index into layer k-1 (or inputs), never forward."""
    for depth in (2, 4, 8):
        bc = LayeredBooleanCircuit(depth=depth, gates_per_layer=8, n_inputs=6, seed=depth)
        for k in range(depth):
            in_size = bc.n_inputs if k == 0 else bc.gates_per_layer
            wires_k = bc.wires[k]  # (gpl, 2)
            assert np.all(wires_k >= 0), f"negative wire index at layer {k}"
            assert np.all(wires_k < in_size), (
                f"layer {k}: wire index {wires_k.max()} >= in_size {in_size}"
            )


@pytest.mark.unit
def test_bc_depth_lt2_raises() -> None:
    """Depth < 2 must raise ValueError."""
    with pytest.raises(ValueError, match="depth"):
        LayeredBooleanCircuit(depth=1, gates_per_layer=4, n_inputs=4, seed=0)


# ---------------------------------------------------------------------------
# V7.0 — SU(2) lattice
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_su2_from_alpha_is_unitary_at_zero() -> None:
    """Zero rotation vector → identity matrix."""
    U = su2_from_alpha(np.zeros(3))
    np.testing.assert_allclose(U, np.eye(2, dtype=np.complex128), atol=1e-14)


@pytest.mark.unit
def test_su2_from_alpha_unitarity() -> None:
    """U U^† = I for random α vectors (the defining property of SU(2))."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        alpha = rng.standard_normal(3)
        U = su2_from_alpha(alpha)
        UUdag = U @ U.conj().T
        np.testing.assert_allclose(UUdag, np.eye(2, dtype=np.complex128), atol=1e-12,
                                   err_msg=f"UU† ≠ I for alpha={alpha}")


@pytest.mark.unit
def test_su2_from_alpha_determinant_one() -> None:
    """det(U) = +1 for any SU(2) element."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        alpha = rng.standard_normal(3)
        U = su2_from_alpha(alpha)
        det = np.linalg.det(U)
        assert abs(det - 1.0) < 1e-12, f"|det - 1| = {abs(det - 1.0):.2e}"


@pytest.mark.unit
def test_plaquette_trace_is_real() -> None:
    """Re tr of a product of SU(2) matrices is real by construction."""
    rng = np.random.default_rng(0)
    L, d = 3, 2
    alpha = build_links(L, d, rng, amplitude=0.5)
    U = links_to_matrices(alpha)
    tr = plaquette_trace(U, mu=0, nu=1, d=d)
    # plaquette_trace already takes np.real, so all entries are float
    assert tr.dtype in (np.float64, np.float32), f"unexpected dtype {tr.dtype}"
    assert tr.shape == (L,) * d


@pytest.mark.unit
def test_plaquette_trace_gauge_invariance() -> None:
    """Re tr U_plaq is gauge-invariant: conjugating every link by site-local g ∈ SU(2) leaves the trace unchanged.

    For a single plaquette U = U1 U2 U3† U4†, conjugating each Ui by the
    appropriate site gauge matrices is a similarity transformation that cancels
    in the trace: tr(g U g†) = tr(U) when the g's telescope around the loop.
    """
    rng = np.random.default_rng(13)
    L, d = 3, 2
    alpha = build_links(L, d, rng, amplitude=0.4)
    # Gauge transform: for each site x, draw a random g(x) ∈ SU(2)
    g_alpha = 0.5 * rng.standard_normal((L,) * d + (3,))
    # Build link matrices before transform
    U_before = links_to_matrices(alpha)
    # Apply gauge: U(x, mu) -> g(x) U(x, mu) g(x + mu_hat)^†
    U_after = U_before.copy()
    for idx in np.ndindex(*((L,) * d)):
        g_x = su2_from_alpha(g_alpha[idx])
        for mu in range(d):
            # x + mu_hat (periodic)
            idx_shifted = list(idx)
            idx_shifted[mu] = (idx[mu] + 1) % L
            g_xmu = su2_from_alpha(g_alpha[tuple(idx_shifted)])
            U_after[idx + (mu,)] = g_x @ U_before[idx + (mu,)] @ g_xmu.conj().T

    tr_before = plaquette_trace(U_before, mu=0, nu=1, d=d)
    tr_after = plaquette_trace(U_after, mu=0, nu=1, d=d)
    np.testing.assert_allclose(tr_after, tr_before, atol=1e-10,
                               err_msg="Plaquette trace not gauge-invariant")


@pytest.mark.unit
def test_action_is_real_valued() -> None:
    """Wilson SU(2) action must be a real Python float (not complex)."""
    rng = np.random.default_rng(5)
    L, d = 3, 2
    alpha = build_links(L, d, rng)
    S = action(alpha, beta=1.5, d=d)
    assert isinstance(S, float), f"action returned {type(S)}"
    assert math.isfinite(S), f"action is not finite: {S}"


@pytest.mark.unit
def test_action_cold_start_lower_than_hot() -> None:
    """Near-identity config (cold) should have lower Wilson action (more negative S)
    than a hot (random) config because Re tr ≈ 2 (identity) > random."""
    rng = np.random.default_rng(99)
    L, d, beta = 3, 2, 2.0
    alpha_cold = build_links(L, d, rng, amplitude=0.01)   # near identity
    alpha_hot = build_links(L, d, rng, amplitude=3.14)    # fully random
    S_cold = action(alpha_cold, beta, d)
    S_hot = action(alpha_hot, beta, d)
    assert S_cold < S_hot, (
        f"Cold config should have lower action: S_cold={S_cold:.4f} S_hot={S_hot:.4f}"
    )


# ---------------------------------------------------------------------------
# V6.4 — Transformer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_transformer_forward_no_nan() -> None:
    """Forward pass must not produce NaN for any standard depth."""
    for depth in (1, 2, 4, 8):
        torch.manual_seed(depth)
        net = Transformer(depth=depth, d_model=16, seq_len=4, n_heads=4)
        x = torch.randn(2, 4, 16)
        out = net(x)
        assert not torch.isnan(out).any(), f"NaN in Transformer output at depth {depth}"


@pytest.mark.unit
def test_transformer_output_shape() -> None:
    """Output shape is (batch, seq_len, d_model) — same as input."""
    torch.manual_seed(0)
    net = Transformer(depth=3, d_model=16, seq_len=6, n_heads=4)
    x = torch.randn(4, 6, 16)
    out = net(x)
    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"


@pytest.mark.unit
def test_fim_diagonal_transformer_shape() -> None:
    """fim_diagonal_transformer output length equals total parameter count."""
    torch.manual_seed(1)
    net = Transformer(depth=2, d_model=16, seq_len=4, n_heads=4)
    n_params = sum(p.numel() for p in net.parameters())
    fim = fim_diagonal_transformer(net, seq_len=4, d_model=16, n_probes=5)
    assert fim.shape == (n_params,), f"Expected ({n_params},), got {fim.shape}"


@pytest.mark.unit
def test_fim_diagonal_transformer_non_negative() -> None:
    """FIM diagonal entries are squared gradients — must be >= 0."""
    torch.manual_seed(2)
    net = Transformer(depth=2, d_model=16, seq_len=4, n_heads=4)
    fim = fim_diagonal_transformer(net, seq_len=4, d_model=16, n_probes=10)
    assert np.all(fim >= 0), "Transformer FIM diagonal must be non-negative"


@pytest.mark.unit
def test_transformer_residual_stream_finite_deep() -> None:
    """Residual stream must stay finite (no explosion) for 16-block stack."""
    torch.manual_seed(3)
    net = Transformer(depth=16, d_model=16, seq_len=4, n_heads=4)
    x = torch.randn(1, 4, 16)
    out = net(x)
    assert torch.isfinite(out).all(), "Residual stream blew up at depth=16"
    # Heuristic: output norm should not massively exceed input norm
    in_norm = x.norm().item()
    out_norm = out.norm().item()
    assert out_norm < 1000 * in_norm, (
        f"Output norm {out_norm:.1f} >> input norm {in_norm:.1f} — likely explosion"
    )


@pytest.mark.unit
def test_transformer_deterministic_with_fixed_seed() -> None:
    """Same seed produces identical fim_diagonal_transformer output."""
    torch.manual_seed(5)
    net = Transformer(depth=2, d_model=16, seq_len=4, n_heads=4)
    torch.manual_seed(10)
    fim1 = fim_diagonal_transformer(net, seq_len=4, d_model=16, n_probes=5)
    torch.manual_seed(10)
    fim2 = fim_diagonal_transformer(net, seq_len=4, d_model=16, n_probes=5)
    np.testing.assert_array_equal(fim1, fim2, err_msg="fim_diagonal_transformer not deterministic")
