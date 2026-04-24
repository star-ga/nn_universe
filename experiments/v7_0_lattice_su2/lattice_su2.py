"""V7.0 — SU(2) pure-gauge lattice FIM test.

Extends V5.0's U(1) result to the non-abelian SU(2) case. Each link
variable is now a 2x2 unitary matrix in SU(2), parameterised by 3 real
angles (rotation axis * angle, via exponential map). The Wilson action
is S = -beta/2 * sum_{x,mu<nu} Re tr U_plaq(x,mu,nu), where

    U_plaq(x,mu,nu) = U(x,mu) U(x+mu_hat,nu) U(x+nu_hat,mu)^dagger U(x,nu)^dagger.

This is the standard Wilson lattice-SU(2) action (Creutz 1980).

FIM diagonal: gradient w.r.t. the 3 real angles of each link. The
action is a smooth function of the angles (via the matrix exponential
exp(i sum_a alpha_a sigma_a) on SU(2)), so autograd / finite-differences
give the Jacobian. We use finite differences for simplicity.

Parameters per link: 3 real (Cayley-Klein angles). Number of links per
site: d (spacetime dimension). Total params: L^d * d * 3.

For L=6, d=4: 6^4 * 4 * 3 = 15 552 real parameters. Comparable to the
V5.0 U(1) run (16 384 link phases) for fair comparison.

Prediction: since lattice SU(2) is still spatially parallel (no deep
sequential chain), T1/T3 should remain O(1), confirming the dichotomy
extends to non-abelian gauge theory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SIGMAS = [SIGMA_X, SIGMA_Y, SIGMA_Z]


def su2_from_alpha(alpha: np.ndarray) -> np.ndarray:
    """Build a single SU(2) matrix from 3 real angles via exp(i sum_a alpha_a sigma_a / 2).
    alpha shape: (3,). Returns (2, 2) complex."""
    # A = alpha . sigma, eigenvalues ±|alpha|. So exp(i A / 2) = cos(|alpha|/2) I + i sin(|alpha|/2) * (alpha . sigma / |alpha|).
    mag = float(np.linalg.norm(alpha))
    if mag < 1e-12:
        return np.eye(2, dtype=np.complex128)
    axis = alpha / mag
    c = math.cos(mag / 2)
    s = math.sin(mag / 2)
    a_dot_sigma = axis[0] * SIGMA_X + axis[1] * SIGMA_Y + axis[2] * SIGMA_Z
    return c * np.eye(2, dtype=np.complex128) + 1j * s * a_dot_sigma


def build_links(L: int, d: int, rng: np.random.Generator, amplitude: float = 0.3) -> np.ndarray:
    """Initialise link angles near identity. alpha shape: (L,)*d + (d, 3)."""
    shape = (L,) * d + (d, 3)
    return amplitude * rng.standard_normal(shape).astype(np.float64)


def links_to_matrices(alpha: np.ndarray) -> np.ndarray:
    """Broadcast alpha -> U. Input shape (..., d, 3); output (..., d, 2, 2) complex."""
    shape = alpha.shape[:-1]  # includes d
    flat = alpha.reshape(-1, 3)
    U = np.empty((flat.shape[0], 2, 2), dtype=np.complex128)
    for i in range(flat.shape[0]):
        U[i] = su2_from_alpha(flat[i])
    return U.reshape(shape + (2, 2))


def plaquette_trace(U: np.ndarray, mu: int, nu: int, d: int) -> np.ndarray:
    """Compute Re tr of U_plaq(x, mu, nu) at every site x.
    U shape: (L,)*d + (d, 2, 2). Returns (L,)*d array of real traces."""
    # U(x, mu)
    U_mu = U[..., mu, :, :]
    # U(x+mu_hat, nu) — roll along axis mu by -1 (axis mu corresponds to x^mu dimension)
    U_nu_shift_mu = np.roll(U[..., nu, :, :], -1, axis=mu)
    # U(x+nu_hat, mu)^dagger
    U_mu_shift_nu = np.roll(U[..., mu, :, :], -1, axis=nu)
    U_mu_shift_nu_dag = np.conj(U_mu_shift_nu).swapaxes(-1, -2)
    # U(x, nu)^dagger
    U_nu_dag = np.conj(U[..., nu, :, :]).swapaxes(-1, -2)
    # Compose: U_mu @ U_nu_shift_mu @ U_mu_shift_nu_dag @ U_nu_dag
    step1 = np.einsum("...ij,...jk->...ik", U_mu, U_nu_shift_mu)
    step2 = np.einsum("...ij,...jk->...ik", step1, U_mu_shift_nu_dag)
    plaq = np.einsum("...ij,...jk->...ik", step2, U_nu_dag)
    # Re tr
    return np.real(plaq[..., 0, 0] + plaq[..., 1, 1])


def action(alpha: np.ndarray, beta: float, d: int) -> float:
    """Wilson SU(2) action S = -beta/2 * sum_{x, mu<nu} Re tr U_plaq."""
    U = links_to_matrices(alpha)
    S = 0.0
    for mu in range(d):
        for nu in range(mu + 1, d):
            tr = plaquette_trace(U, mu, nu, d)
            S += -0.5 * beta * np.sum(tr)
    return float(S)


def grad_action_fd(alpha: np.ndarray, beta: float, d: int, eps: float = 1e-3) -> np.ndarray:
    """Finite-difference gradient of the action w.r.t. each angle.
    Returns same shape as alpha. O(n_params) matrix evaluations — use only
    for small lattices."""
    grad = np.zeros_like(alpha)
    it = np.nditer(alpha, flags=["multi_index"], op_flags=["readonly"])
    S_base = action(alpha, beta, d)
    while not it.finished:
        idx = it.multi_index
        alpha_pert = alpha.copy()
        alpha_pert[idx] += eps
        S_new = action(alpha_pert, beta, d)
        grad[idx] = (S_new - S_base) / eps
        it.iternext()
    return grad


def metropolis_sweep(alpha: np.ndarray, beta: float, d: int, step: float, rng: np.random.Generator) -> None:
    """One Metropolis update per angle, in place. Uses local plaquette
    re-evaluation (only 6 plaquettes change per link for d=4)."""
    shape = alpha.shape[:-2]  # (L,)*d
    L = shape[0]
    it = np.ndindex(*shape)
    sites = list(it)
    rng.shuffle(sites)
    for idx in sites:
        for mu in range(d):
            for a in range(3):
                old = alpha[idx + (mu, a)]
                proposed = old + step * rng.standard_normal()
                alpha[idx + (mu, a)] = proposed
                S_new = action(alpha, beta, d)
                alpha[idx + (mu, a)] = old
                S_old = action(alpha, beta, d)
                dS = S_new - S_old
                if dS < 0 or rng.random() < math.exp(-dS):
                    alpha[idx + (mu, a)] = proposed


def metropolis_sweep_local(alpha: np.ndarray, beta: float, d: int, step: float, rng: np.random.Generator) -> None:
    """Local Metropolis — re-evaluate only plaquettes containing the touched link."""
    shape = alpha.shape[:-2]
    L = shape[0]
    sites = list(np.ndindex(*shape))
    rng.shuffle(sites)
    U = links_to_matrices(alpha)
    for idx in sites:
        for mu in range(d):
            # Compute local action contribution from this link
            local = 0.0
            for nu in range(d):
                if nu == mu:
                    continue
                lo, hi = (mu, nu) if mu < nu else (nu, mu)
                # Plaquette at x with (lo, hi)
                tr_forward = plaquette_trace(U, lo, hi, d)[idx]
                # Plaquette at x - nu_hat with (lo, hi)
                idx_bwd = list(idx); idx_bwd[nu] = (idx[nu] - 1) % L
                tr_backward = plaquette_trace(U, lo, hi, d)[tuple(idx_bwd)]
                local += tr_forward + tr_backward
            S_old = -0.5 * beta * local

            for a in range(3):
                old = alpha[idx + (mu, a)]
                proposed = old + step * rng.standard_normal()
                alpha[idx + (mu, a)] = proposed
                U_new_mat = su2_from_alpha(alpha[idx + (mu,)])
                U_old_mat = U[idx + (mu,)].copy()
                U[idx + (mu,)] = U_new_mat
                local_new = 0.0
                for nu in range(d):
                    if nu == mu:
                        continue
                    lo, hi = (mu, nu) if mu < nu else (nu, mu)
                    tr_forward = plaquette_trace(U, lo, hi, d)[idx]
                    idx_bwd = list(idx); idx_bwd[nu] = (idx[nu] - 1) % L
                    tr_backward = plaquette_trace(U, lo, hi, d)[tuple(idx_bwd)]
                    local_new += tr_forward + tr_backward
                S_new = -0.5 * beta * local_new
                dS = S_new - S_old
                if dS < 0 or rng.random() < math.exp(-dS):
                    local = local_new
                    S_old = S_new
                    # accepted — keep U[idx+(mu,)] as the new matrix
                else:
                    alpha[idx + (mu, a)] = old
                    U[idx + (mu,)] = U_old_mat


def tier_ratio(values: np.ndarray, top_pct: float = 1.0, bot_pct: float = 50.0) -> tuple[float, float, float]:
    sorted_desc = np.sort(values)[::-1]
    n = len(sorted_desc)
    k1 = max(1, int(n * top_pct / 100))
    k3 = max(1, int(n * bot_pct / 100))
    t1 = float(sorted_desc[:k1].mean())
    t3 = float(sorted_desc[-k3:].mean())
    if t3 <= 0:
        nz = sorted_desc[sorted_desc > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1 / t3 if t3 > 0 else float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4, help="lattice size (each dimension)")
    ap.add_argument("--d", type=int, default=4, help="spacetime dimension")
    ap.add_argument("--beta", type=float, default=2.0, help="inverse gauge coupling")
    ap.add_argument("--thermalise", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--decorr", type=int, default=10)
    ap.add_argument("--step-size", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v7_0_lattice_su2_results.json"))
    args = ap.parse_args()

    results: list[dict] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        alpha = build_links(args.L, args.d, rng)
        n_params = alpha.size
        print(f"seed={seed}  L={args.L}  d={args.d}  n_params={n_params:,}  beta={args.beta}", flush=True)

        # Thermalise
        t0 = time.time()
        for sweep_i in range(args.thermalise):
            metropolis_sweep_local(alpha, args.beta, args.d, args.step_size, rng)
            if sweep_i == 0:
                print(f"  sweep 1 done in {time.time()-t0:.1f}s", flush=True)
        t_therm = time.time() - t0
        print(f"  thermalised in {t_therm:.1f}s", flush=True)

        # Measure
        t0 = time.time()
        fim = np.zeros(n_params, dtype=np.float64)
        for i in range(args.n_samples):
            for _ in range(args.decorr):
                metropolis_sweep_local(alpha, args.beta, args.d, args.step_size, rng)
            g = grad_action_fd(alpha, args.beta, args.d).ravel()
            fim += g * g
        fim /= args.n_samples
        t_meas = time.time() - t0

        t1, t3, r = tier_ratio(fim)
        print(f"  measured in {t_meas:.1f}s  T1={t1:.3e}  T3={t3:.3e}  T1/T3={r:.2f}", flush=True)

        results.append({
            "seed": seed, "L": args.L, "d": args.d, "beta": args.beta,
            "n_params": int(n_params),
            "fim_tier1_mean": t1, "fim_tier3_mean": t3, "fim_tier1_tier3": r,
            "thermalise_s": t_therm, "measure_s": t_meas,
        })

    tier_ratios = np.array([r["fim_tier1_tier3"] for r in results])
    aggregate = {
        "tier_ratio_mean": float(tier_ratios.mean()),
        "tier_ratio_std": float(tier_ratios.std(ddof=1)) if len(tier_ratios) > 1 else 0.0,
        "tier_ratio_cv": float(tier_ratios.std(ddof=1) / tier_ratios.mean()) if len(tier_ratios) > 1 and tier_ratios.mean() > 0 else 0.0,
    }
    print(f"\nSU(2) lattice FIM T1/T3: mean={aggregate['tier_ratio_mean']:.2f}  CV={aggregate['tier_ratio_cv']*100:.1f}%", flush=True)

    payload = {"config": vars(args), "results": results, "aggregate": aggregate,
               "comparison_context": {
                   "U1_lattice_L8_beta1": 1.6,
                   "trained_NN_W256": 337,
                   "untrained_NN_W256": 3757,
                   "boolean_circuit_N384": 88440,
               },
               "interpretation": (
                   "SU(2) is non-abelian spatially parallel. Dichotomy predicts T1/T3 = O(1). "
                   "If confirmed, the 'spatially parallel substrates' side of the dichotomy "
                   "extends to non-abelian gauge theories."
               )}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
