"""V2.0 — analytic (untrained) FIM convergence demo.

The V2.0 theorem is a statement about the discrete FIM *functional*
:math:`g_{x,x'}(\\theta) = \\mathbb{E}_y\\!\\left[\\partial_{\\theta_x} f(y)\\,\\partial_{\\theta_{x'}} f(y)\\right]`,
not about training dynamics. Training-based demos (``lattice_refinement.py``)
confound the clean Cauchy convergence with stochastic training noise.

For the 1-hidden-layer lattice-embedded network

.. math::

   f(y;\\theta) \\;=\\; \\sum_{x \\in \\Lambda_a} w_x\\,\\phi(y - x),

the derivative is :math:`\\partial_{w_x} f(y) = \\phi(y - x)`, independent
of :math:`w`. Hence

.. math::

   g_{x,x'} \\;=\\; \\int \\phi(y - x)\\,\\phi(y - x')\\,\\mathrm{d}\\mu(y)
            \\;=\\; (\\phi * \\phi)(x - x'),

which for a Gaussian receptive field :math:`\\phi` of width :math:`\\sigma`
is a Gaussian of width :math:`\\sigma\\sqrt 2`. As :math:`a \\to 0`, the
discrete sum over the lattice approximates this integral at rate
:math:`O(a^2)` (trapezoidal / midpoint error), giving a clean Cauchy
sequence.

This script measures :math:`g_{0, x}(a_n)` — the FIM row at the origin —
across a sequence of lattice refinements and compares it to the
analytically-known limit kernel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def gaussian_conv_exact(delta: np.ndarray, sigma: float, d: int) -> np.ndarray:
    """Convolution (phi * phi)(delta) for a d-dimensional Gaussian of width sigma.

    Normalization: phi(x) = exp(-||x||^2 / (2 sigma^2)), so
    (phi * phi)(delta) = (sqrt(pi) * sigma)^d * exp(-||delta||^2 / (4 sigma^2)).
    """
    norm = (math.sqrt(math.pi) * sigma) ** d
    return norm * np.exp(-np.sum(delta ** 2, axis=-1) / (4 * sigma ** 2))


def discrete_fim_row(
    domain: float, spacing: float, sigma: float, d: int, eval_density: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the discrete FIM row at the origin of a lattice.

    Returns (site_coords, fim_row).
    """
    n_per_axis = int(round(domain / spacing)) + 1
    axis = np.linspace(-domain / 2, domain / 2, n_per_axis)
    mesh = np.stack(np.meshgrid(*([axis] * d), indexing="ij"), axis=-1).reshape(-1, d)
    site_coords = mesh  # shape (n_sites, d)

    # Evaluation integration over the domain using midpoint rule.
    eval_axis = np.linspace(-domain / 2, domain / 2, eval_density)
    eval_mesh = np.stack(np.meshgrid(*([eval_axis] * d), indexing="ij"), axis=-1).reshape(-1, d)
    dy = (domain / (eval_density - 1)) ** d

    # origin site.
    origin = np.zeros((1, d))

    def phi(y_minus_x: np.ndarray) -> np.ndarray:
        r2 = np.sum(y_minus_x ** 2, axis=-1)
        return np.exp(-r2 / (2 * sigma ** 2))

    # phi at origin for every eval point y
    phi_at_origin = phi(eval_mesh - origin)  # (n_eval,)
    # phi at each site x for every eval point y
    # shape (n_eval, n_sites)
    deltas = eval_mesh[:, None, :] - site_coords[None, :, :]
    phi_at_sites = np.exp(-np.sum(deltas ** 2, axis=-1) / (2 * sigma ** 2))

    # Discrete FIM row: g_{0,x} = integral_y phi(y-0) phi(y-x) dy ~ sum_y phi_at_origin[y] * phi_at_sites[y,x] * dy
    fim_row = (phi_at_origin[:, None] * phi_at_sites).sum(axis=0) * dy
    return site_coords, fim_row


def analytic_fim_row(site_coords: np.ndarray, sigma: float, d: int) -> np.ndarray:
    """Analytic closed form for the continuum FIM row at origin."""
    # delta = x - 0 = x
    return gaussian_conv_exact(site_coords, sigma, d)


def test_function(x: np.ndarray, d: int, freq: float = 1.0) -> np.ndarray:
    """Smooth test function u(x) = exp(-||x||^2/2) * cos(freq * sum(x)).
    Used to contract the bilinear form."""
    return np.exp(-np.sum(x ** 2, axis=-1) / 2) * np.cos(freq * np.sum(x, axis=-1))


def bilinear_form_discrete(
    domain: float, spacing: float, sigma: float, d: int, eval_density: int, freq: float
) -> float:
    """Compute u^T G_a u via the discretized lattice, where G_a is the FIM bilinear form.

    Returns float.
    """
    n_per_axis = int(round(domain / spacing)) + 1
    axis = np.linspace(-domain / 2, domain / 2, n_per_axis)
    mesh = np.stack(np.meshgrid(*([axis] * d), indexing="ij"), axis=-1).reshape(-1, d)
    u_vals = test_function(mesh, d, freq)

    # G_a(x, x') = (phi * phi)(x - x') -- Gaussian of width sigma*sqrt(2), normalized.
    norm = (math.sqrt(math.pi) * sigma) ** d
    # Pairwise compute
    deltas = mesh[:, None, :] - mesh[None, :, :]
    kernel = norm * np.exp(-np.sum(deltas ** 2, axis=-1) / (4 * sigma ** 2))
    # Continuum normalization: sum over Λ_a with quadrature weight a^d gives integral.
    # u^T G u = sum_{x, x'} a^{2d} u(x) u(x') G_a(x, x')
    return float(spacing ** (2 * d) * (u_vals @ kernel @ u_vals))


def bilinear_form_continuum(domain: float, sigma: float, d: int, eval_density: int, freq: float) -> float:
    """High-resolution reference: the continuum bilinear form ∫∫ u(x) u(x') G(x-x') dx dx'."""
    axis = np.linspace(-domain / 2, domain / 2, eval_density)
    mesh = np.stack(np.meshgrid(*([axis] * d), indexing="ij"), axis=-1).reshape(-1, d)
    u_vals = test_function(mesh, d, freq)
    dx = (domain / (eval_density - 1)) ** d
    norm = (math.sqrt(math.pi) * sigma) ** d
    deltas = mesh[:, None, :] - mesh[None, :, :]
    kernel = norm * np.exp(-np.sum(deltas ** 2, axis=-1) / (4 * sigma ** 2))
    return float(dx ** 2 * (u_vals @ kernel @ u_vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--domain", type=float, default=4.0)
    ap.add_argument("--sigma", type=float, default=0.5, help="physical receptive-field width")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--eval-density", type=int, default=81, help="integration grid for reference continuum integral")
    ap.add_argument("--freq", type=float, default=1.0)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "lattice_analytic_results.json"),
    )
    args = ap.parse_args()

    # Continuum reference.
    ref = bilinear_form_continuum(args.domain, args.sigma, args.d, args.eval_density, args.freq)
    print(f"Continuum reference u^T G u = {ref:.6e}  (eval_density={args.eval_density})")

    # Refinement sequence a_n = a_0 / 2^n.
    a0 = 1.0
    levels_data = []
    for n in range(args.levels):
        spacing = a0 / (2 ** n)
        val = bilinear_form_discrete(args.domain, spacing, args.sigma, args.d, args.eval_density, args.freq)
        err = val - ref
        rel_err = abs(err) / max(abs(ref), 1e-12)
        levels_data.append(
            {
                "level": n,
                "spacing": spacing,
                "bilinear_form": val,
                "abs_err": abs(err),
                "rel_err": rel_err,
            }
        )
        print(f"  level {n}: a={spacing:.4f}  u^T G u = {val:.6e}  |err|={abs(err):.3e}  rel={rel_err:.3e}")

    # Fit convergence rate.
    ns = np.arange(args.levels)
    log_err = np.log2(np.clip([d["abs_err"] for d in levels_data], 1e-30, None))
    slope, intercept = np.polyfit(ns, log_err, 1)
    observed_rate = -slope
    print(f"\nObserved convergence rate: |err| ~ a^{observed_rate:.3f}  (midpoint-rule theory: O(a^2))")

    payload = {
        "config": vars(args),
        "continuum_reference": ref,
        "levels": levels_data,
        "observed_convergence_rate": observed_rate,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
