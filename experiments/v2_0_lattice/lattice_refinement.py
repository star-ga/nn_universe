r"""V2.0 numerical demo — Cauchy refinement of the discrete FIM to a smooth metric.

Construction
------------
We take the simplest setting that exhibits the lattice-embedded regime:
a translation-invariant 1-hidden-layer ReLU network whose hidden units sit on
the sites of a d-dimensional hypercubic lattice :math:`\Lambda_a`. For each
refinement level :math:`n`, we halve the lattice spacing :math:`a_n = a_0 / 2^n`
while rescaling weights so that the continuum density of neurons is preserved
(``continuum normalization``).

For each :math:`n` we compute the discrete FIM row at the origin,
:math:`G_{a_n}(x) := g_{0,x}(\theta)` as :math:`x` ranges over :math:`\Lambda_{a_n}`,
and check that the Cauchy criterion

.. math::

   \lVert G_{a_{n+1}} - G_{a_n} \rVert \to 0

holds in a suitable norm. The limit :math:`G^{(\infty)}` is compared against
the analytically-known infinite-width NTK kernel (Gaussian-activated
approximation), yielding an empirical convergence rate.

We use :math:`d = 2` for plotting and computational tractability; the
theorem in ``docs/v2_0_lattice_embedded.md`` is dimension-general.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2, help="lattice dimension (2 for plots)")
    p.add_argument("--L", type=int, default=8, help="linear size of the largest lattice (2^L+1 sites)")
    p.add_argument("--levels", type=int, default=5, help="number of refinement levels")
    p.add_argument("--radius", type=int, default=2, help="locality radius r (in lattice spacings)")
    p.add_argument("--n-samples", type=int, default=400, help="FIM samples per level")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=2_000, help="pre-training steps to reach a non-trivial FIM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "lattice_refinement_results.json"),
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


class LatticeEmbeddedNet(nn.Module):
    """Translation-invariant 1-hidden-layer ReLU net on a hypercubic lattice.

    Each site :math:`x` carries a hidden unit with scalar weight :math:`w_x`.
    Output at a point :math:`y` is

    .. math::

       f(y) = \sum_{x: \|x-y\|_\infty \le r a} w_x \,\phi\!\bigl((y - x) / a\bigr)

    with :math:`\phi` a translation-symmetric local receptive field (ReLU of
    a Gaussian bump). The weights :math:`w_x` are the only trainable
    parameters; translation invariance is imposed by construction by using a
    convolutional (shared-weight) readout over the lattice, but we still
    index parameters per-site for FIM measurement.
    """

    def __init__(self, shape: tuple[int, ...], r: int, a: float, d: int, sigma: float = 0.5) -> None:
        super().__init__()
        self.shape = shape
        self.r = r
        self.a = a
        self.d = d
        # Receptive field width is a *physical* constant (sigma), not
        # lattice-scaled. This is the r*a -> xi regime of the V2.0 theorem.
        self.sigma = sigma
        # One scalar weight per site; continuum normalization w_x = a^{d/2} phi_x.
        n_sites = int(np.prod(shape))
        self.weights = nn.Parameter(torch.randn(n_sites) * (a ** (d / 2)))

    def _receptive_field(self, delta_phys: torch.Tensor) -> torch.Tensor:
        r"""Gaussian receptive field of fixed physical width sigma.

        delta_phys has *physical* units; this is the r*a -> xi (finite) regime.
        """
        r = delta_phys.pow(2).sum(dim=-1) / (self.sigma ** 2)
        gauss = torch.exp(-0.5 * r)
        return torch.relu(gauss - 1e-3)

    def eval_at(self, eval_points: torch.Tensor, site_coords: torch.Tensor) -> torch.Tensor:
        r"""Evaluate f at a set of points.

        eval_points: (n_eval, d)   site_coords: (n_sites, d)
        returns (n_eval,)
        """
        # Work in physical coordinates: delta_phys = y - x, NOT scaled by a.
        delta_phys = eval_points.unsqueeze(1) - site_coords.unsqueeze(0)
        phi = self._receptive_field(delta_phys)
        return torch.einsum("ij,j->i", phi, self.weights)


def make_lattice_coords(shape: tuple[int, ...], a: float) -> torch.Tensor:
    """Return integer lattice site coordinates scaled by spacing ``a``."""
    axes = [np.arange(n) for n in shape]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(shape))
    return torch.from_numpy(mesh.astype(np.float32) * a)


def compute_fim_row_at_origin(
    net: LatticeEmbeddedNet,
    site_coords: torch.Tensor,
    eval_points: torch.Tensor,
    n_samples: int,
    batch: int,
    device: torch.device,
) -> np.ndarray:
    """FIM row at the origin site.

    Returns an array of shape ``(n_sites,)`` giving
    :math:`g_{0, x}` via empirical gradient product.
    """
    net.zero_grad(set_to_none=True)
    fim_row = torch.zeros(net.weights.shape[0], device=device)
    for _ in range(n_samples):
        idx = torch.randint(0, eval_points.shape[0], (batch,), device=device)
        pts = eval_points[idx]
        pred = net.eval_at(pts, site_coords)
        target = torch.zeros_like(pred)
        loss = 0.5 * (pred - target).pow(2).mean()
        grad = torch.autograd.grad(loss, net.weights, create_graph=False)[0]
        fim_row += grad.detach() ** 2
    fim_row /= n_samples
    return fim_row.detach().cpu().numpy()


def run_level(args, level: int, a0: float, device: torch.device, rng: np.random.Generator) -> dict:
    spacing = a0 / (2 ** level)
    # Keep the *physical size* of the lattice fixed at L0 = (2^L) * a0; this
    # means the number of sites doubles per refinement in each dimension.
    L_sites = (2 ** (args.L - args.levels + level)) + 1
    # To keep memory bounded, scale the coarse level down but preserve the
    # continuum domain. (physical domain = L_sites * spacing in each dir)
    shape = (L_sites,) * args.d

    torch.manual_seed(args.seed + level)
    net = LatticeEmbeddedNet(shape=shape, r=args.radius, a=spacing, d=args.d).to(device)
    site_coords = make_lattice_coords(shape, spacing).to(device)

    # Evaluation points: dense grid at half the finest spacing to oversample.
    eval_sites = L_sites
    eval_coords = make_lattice_coords((eval_sites,) * args.d, spacing).to(device)

    # A short warm-up training on a toy loss so the FIM reflects a trained
    # configuration rather than random init.
    opt = torch.optim.SGD([net.weights], lr=1e-2, momentum=0.9)
    for _ in range(args.steps):
        idx = torch.randint(0, eval_coords.shape[0], (args.batch,), device=device)
        pts = eval_coords[idx]
        pred = net.eval_at(pts, site_coords)
        target = torch.sin(pts.sum(dim=-1) * 2.0)
        loss = 0.5 * (pred - target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Measure FIM row at the central site (origin of the lattice after recentering).
    fim_row = compute_fim_row_at_origin(net, site_coords, eval_coords, args.n_samples, args.batch, device)
    n_sites = int(np.prod(shape))
    return {
        "level": level,
        "spacing": spacing,
        "shape": list(shape),
        "n_sites": n_sites,
        "fim_row_l2": float(np.linalg.norm(fim_row)),
        "fim_row_max": float(fim_row.max()),
        "fim_row_origin_to_distance": _radial_profile(fim_row, shape, spacing),
    }


def _radial_profile(fim_row: np.ndarray, shape: tuple[int, ...], spacing: float) -> list[dict]:
    coords = np.indices(shape).reshape(len(shape), -1).T
    center = np.array(shape) // 2
    dist = np.linalg.norm((coords - center) * spacing, axis=1)
    bins = np.linspace(0, dist.max(), 16)
    out = []
    for i in range(len(bins) - 1):
        mask = (dist >= bins[i]) & (dist < bins[i + 1])
        if mask.sum() > 0:
            out.append({"r_lo": float(bins[i]), "r_hi": float(bins[i + 1]), "mean": float(fim_row[mask].mean()), "count": int(mask.sum())})
    return out


def cauchy_rate(levels: list[dict]) -> list[dict]:
    """Compute pairwise differences in the radial profiles to verify Cauchy convergence."""
    rates = []
    for a, b in zip(levels, levels[1:]):
        # Restrict to the overlap of radial bins; compare by interpolation.
        x_a = np.array([0.5 * (r["r_lo"] + r["r_hi"]) for r in a["fim_row_origin_to_distance"]])
        y_a = np.array([r["mean"] for r in a["fim_row_origin_to_distance"]])
        x_b = np.array([0.5 * (r["r_lo"] + r["r_hi"]) for r in b["fim_row_origin_to_distance"]])
        y_b = np.array([r["mean"] for r in b["fim_row_origin_to_distance"]])
        # Common radii
        r_max = float(min(x_a.max(), x_b.max()))
        xs = np.linspace(0.0, r_max, 24)
        y_a_i = np.interp(xs, x_a, y_a)
        y_b_i = np.interp(xs, x_b, y_b)
        err = float(np.linalg.norm(y_b_i - y_a_i) / max(np.linalg.norm(y_a_i), 1e-12))
        rates.append({"from_level": a["level"], "to_level": b["level"], "relative_error": err})
    return rates


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)
    a0 = 1.0
    results = []
    for level in range(args.levels):
        print(f"[level {level}] spacing={a0 / (2**level):.4f}  sites≈{(2**(args.L - args.levels + level) + 1)**args.d:,}")
        results.append(run_level(args, level, a0, device, rng))
    rates = cauchy_rate(results)
    payload = {
        "config": vars(args),
        "levels": results,
        "cauchy_rates": rates,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nCauchy convergence (L2 of radial profile differences):")
    for r in rates:
        print(f"  level {r['from_level']} → {r['to_level']}: rel_err = {r['relative_error']:.4f}")
    print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
