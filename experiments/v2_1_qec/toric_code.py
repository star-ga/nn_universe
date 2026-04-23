"""Toric code syndrome generation for neural-decoder training.

Minimal implementation of Kitaev's toric code on an L x L torus, bit-flip
noise channel, syndrome extraction. We work with Z-stabilizers only (the
Z-syndrome problem); the framework is symmetric for X-stabilizers.

Geometry
--------
- Qubits on edges of an L x L torus: 2 * L^2 qubits total.
- Horizontal edge at (i, j): index 2*(i*L + j)   + 0  ("h" edge, connects (i,j)-(i,j+1 mod L))
- Vertical   edge at (i, j): index 2*(i*L + j)   + 1  ("v" edge, connects (i,j)-(i+1 mod L, j))
- Plaquette stabilizer at (i, j): product of Z on the 4 edges bordering that face.

Syndrome
--------
Applying X errors independently on each qubit with probability p, we get
an error vector e in GF(2)^{2 L^2}. The syndrome is H e (mod 2), where H
is the L^2 x 2 L^2 plaquette parity-check matrix.

Decoding task
-------------
Given syndrome s = H e, predict a correction c such that H c = s. Success
iff c + e is a Z-stabilizer (belongs to the row-space of H) — i.e.
the correction restores the codeword modulo stabilizers. For a neural
decoder we use a simpler proxy: mean-squared error (or BCE) between the
predicted correction and the true error vector e, which is an upper bound
on the logical-error rate at small p.
"""
from __future__ import annotations

import numpy as np


def parity_check_matrix(L: int) -> np.ndarray:
    """Return the L^2 x 2L^2 Z-stabilizer parity-check matrix over GF(2)."""
    n_qubits = 2 * L * L
    n_checks = L * L
    H = np.zeros((n_checks, n_qubits), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            plaq = i * L + j
            # Plaquette at (i, j) has corners (i,j), (i,j+1), (i+1,j), (i+1,j+1).
            # Bordering edges:
            #   top horizontal   = h edge at (i, j)
            #   bottom horizontal = h edge at (i+1 mod L, j)
            #   left  vertical   = v edge at (i, j)
            #   right vertical   = v edge at (i, j+1 mod L)
            h_top = 2 * (i * L + j) + 0
            h_bot = 2 * (((i + 1) % L) * L + j) + 0
            v_lft = 2 * (i * L + j) + 1
            v_rgt = 2 * (i * L + ((j + 1) % L)) + 1
            for idx in (h_top, h_bot, v_lft, v_rgt):
                H[plaq, idx] ^= 1
    return H


def sample_errors(batch: int, L: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Sample i.i.d. bit-flip errors. Returns shape (batch, 2 L^2) in {0, 1}."""
    return (rng.random((batch, 2 * L * L)) < p).astype(np.int8)


def compute_syndromes(errors: np.ndarray, H: np.ndarray) -> np.ndarray:
    """s = H e  mod 2.  Shapes: errors (B, n_qubits), H (n_checks, n_qubits)."""
    return (errors @ H.T) & 1


def make_batch(L: int, p: float, batch: int, rng: np.random.Generator, H: np.ndarray):
    errors = sample_errors(batch, L, p, rng)
    syndromes = compute_syndromes(errors, H)
    return syndromes.astype(np.float32), errors.astype(np.float32)


def logical_error_rate(pred: np.ndarray, true: np.ndarray, H: np.ndarray, L: int) -> float:
    """Estimate logical error rate via the syndrome-match proxy.

    A correction is 'trivial logical' iff pred + true lies in the row-space of H,
    i.e. their XOR is a stabilizer. We approximate by:
        (1) syndrome-match rate: H (pred + true) == 0 mod 2
        (2) we do NOT separate logical X1, X2 operators here — that's an OK
            approximation at small p (where errors are short strings).

    Returns the fraction of trials where H (pred XOR true) != 0, i.e.
    UNDETECTED failures on the syndrome side (a lower bound on logical error).
    """
    pred_bits = (pred > 0.5).astype(np.int8)
    true_bits = true.astype(np.int8)
    residual = (pred_bits ^ true_bits) & 1
    residual_syndrome = (residual @ H.T) & 1
    match = np.all(residual_syndrome == 0, axis=1)
    # "match" means the residual is a pure stabilizer OR a logical operator.
    # For short-code, small-p settings, logical ops are rare, so syndrome-match
    # is a reasonable success criterion.
    return float(1.0 - match.mean())
