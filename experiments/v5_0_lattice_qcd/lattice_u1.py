"""V5.0 / Phase E item 16 — FIM tier structure on a U(1) pure-gauge lattice.

Minimal physics-substrate test: does known lattice gauge theory
exhibit the same FIM-diagonal three-tier hierarchy as trained neural
networks?

Setup (U(1) pure-gauge in 4D, Wilson action)
--------------------------------------------
- Lattice: L^4 sites (default L=8 → 4,096 sites, 16,384 links).
- Parameters: the link-phase angles theta_{x,mu} ∈ [-π, π] — one
  scalar parameter per (site, direction). This is the gauge analog
  of the neural network's weight variables.
- Action: S = -β Σ_P Re[U_P],  U_P = exp(i theta_plaquette), where
  theta_plaquette is the oriented sum of link phases around each
  1×1 square in a chosen pair of directions.
- Sampling: Metropolis Monte Carlo at inverse coupling β (default 1.0).

FIM
---
The "likelihood" is exp(-S)/Z. The per-parameter gradient is
∂S/∂theta_{x,mu}. The empirical FIM diagonal is
    F_{(x,mu)} = <(∂S/∂theta_{x,mu})^2> over Monte Carlo configurations
where the expectation is across equilibrated samples. This mirrors
the NN FIM diagonal where grads are taken w.r.t. parameters under
a data distribution; here parameters ARE the links and the "data
distribution" is the Gibbs distribution of the gauge fluctuations.

Then we compute T1/T3 using the same 1% / 50% tier partition used
throughout this repo. If T1/T3 ~ O(1-10), lattice QCD does NOT share
the NN hierarchy — the V4.0 "layered sequential computation" framing
is vindicated (lattice is spatially parallel, not layered). If
T1/T3 ~ 10^3+, the NN universality class extends to known physics
and the FIM-Onsager cosmological interpretation gains empirical
support.

Runs in minutes on CPU for L=8; no GPU required.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def init_links(L: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Random hot-start: theta ~ uniform(-π, π). Shape (L, L, L, L, d)."""
    return rng.uniform(-math.pi, math.pi, size=(L,) * d + (d,)).astype(np.float64)


def plaquette_sum(theta: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Return theta_P for all plaquettes in the (mu, nu) plane.

    theta_P(x) = theta(x, mu) + theta(x+mu, nu) - theta(x+nu, mu) - theta(x, nu)

    Shape: same as theta's spatial shape (sum over planes gives the full action).
    """
    # Shift tensors along axes mu and nu.
    th_mu = theta[..., mu]
    th_nu = theta[..., nu]
    th_mu_plus_nu = np.roll(th_mu, -1, axis=nu)
    th_nu_plus_mu = np.roll(th_nu, -1, axis=mu)
    return th_mu + th_nu_plus_mu - th_mu_plus_nu - th_nu


def action(theta: np.ndarray, beta: float, d: int) -> float:
    """Wilson gauge action. Sum over distinct ordered planes mu<nu."""
    S = 0.0
    for mu in range(d):
        for nu in range(mu + 1, d):
            plaq = plaquette_sum(theta, mu, nu)
            S += -beta * np.sum(np.cos(plaq))
    return float(S)


def grad_action(theta: np.ndarray, beta: float, d: int) -> np.ndarray:
    """Gradient of the Wilson action S = -β Σ_{x,μ<ν} cos(P(x,μ,ν)) w.r.t.
    each link phase θ(x, μ).

    Derivation. With P(x,μ,ν) = θ(x,μ) + θ(x+μ̂,ν) − θ(x+ν̂,μ) − θ(x,ν),
    P(x,ν,μ) = −P(x,μ,ν). For every ν ≠ μ, θ(x,μ) lies in exactly two
    plaquettes in the unique unordered (μ,ν) plane: P(x,μ,ν) (coeff +1
    when μ<ν, coeff −1 when μ>ν by anti-symmetry) and P(x−ν̂,μ,ν) (coeff
    −1 when μ<ν, coeff +1 when μ>ν). Summing ∂/∂θ(x,μ)[−β cos(P)] in
    both cases gives the same expression:

        ∂S/∂θ(x,μ) = Σ_{ν≠μ} β [ sin(P(x,μ,ν)) − sin(P(x−ν̂,μ,ν)) ]

    The loop iterates every ordered (μ,ν) with ν≠μ once, so there is no
    double-counting; the mu<nu vs mu>nu branches carry identical formulas.
    """
    grad = np.zeros_like(theta)
    for mu in range(d):
        for nu in range(d):
            if mu == nu:
                continue
            plaq = plaquette_sum(theta, mu, nu)            # P(x, mu, nu)
            sin_fwd = np.sin(plaq)                         # sin P(x, mu, nu)
            sin_bwd = np.sin(np.roll(plaq, 1, axis=nu))    # sin P(x-nu_hat, mu, nu)
            grad[..., mu] += beta * (sin_fwd - sin_bwd)
    return grad


def metropolis_sweep(theta: np.ndarray, beta: float, d: int, step_size: float, rng: np.random.Generator) -> None:
    """One Metropolis update per link, in place. Uses a random-order site sweep."""
    shape = theta.shape[:-1]
    coords = np.array(np.meshgrid(*[range(s) for s in shape], indexing="ij"))
    coords = coords.reshape(d, -1).T  # (N_sites, d)
    rng.shuffle(coords)
    for idx in coords:
        for mu in range(d):
            old = theta[tuple(idx) + (mu,)]
            proposed = old + rng.uniform(-step_size, step_size)
            # Compute change in action locally: only plaquettes containing this
            # link change. But since we're focused on measurement rather than
            # thermalisation speed, use global recomputation for correctness.
            theta[tuple(idx) + (mu,)] = proposed
            S_new = action(theta, beta, d)
            theta[tuple(idx) + (mu,)] = old
            S_old = action(theta, beta, d)
            dS = S_new - S_old
            if dS < 0 or rng.random() < math.exp(-dS):
                theta[tuple(idx) + (mu,)] = proposed


def metropolis_sweep_local(theta: np.ndarray, beta: float, d: int, step_size: float, rng: np.random.Generator) -> None:
    """Fast local Metropolis — only recompute the action change from plaquettes
    that contain the proposed link.

    For a link (x, mu), the local action change is
      ΔS = Σ_{ν≠μ} -β [cos(P_new(x,mu,nu)) - cos(P_old(x,mu,nu))
                      + cos(P_new(x-nu,mu,nu)) - cos(P_old(x-nu,mu,nu))]
    """
    shape = theta.shape[:-1]
    L = shape[0]
    n_sites = int(np.prod(shape))
    coords = rng.integers(0, L, size=(n_sites * d, d))
    mus = rng.integers(0, d, size=n_sites * d)
    for i in range(n_sites * d):
        site = tuple(coords[i])
        mu = int(mus[i])
        old = theta[site + (mu,)]
        proposed = old + rng.uniform(-step_size, step_size)

        # ΔS from plaquettes containing this link.
        dS = 0.0
        for nu in range(d):
            if nu == mu:
                continue
            # Plaquette at site (x, mu, nu): uses links (x, mu), (x+mu, nu), (x+nu, mu), (x, nu)
            site_plus_mu = list(site); site_plus_mu[mu] = (site_plus_mu[mu] + 1) % L
            site_plus_nu = list(site); site_plus_nu[nu] = (site_plus_nu[nu] + 1) % L
            t_nu_at_xpmu = theta[tuple(site_plus_mu) + (nu,)]
            t_mu_at_xpnu = theta[tuple(site_plus_nu) + (mu,)]
            t_nu_at_x = theta[site + (nu,)]
            P_old = old + t_nu_at_xpmu - t_mu_at_xpnu - t_nu_at_x
            P_new = proposed + t_nu_at_xpmu - t_mu_at_xpnu - t_nu_at_x
            dS += -beta * (math.cos(P_new) - math.cos(P_old))

            # Plaquette at (x - nu, mu, nu): uses links (x-nu, mu), (x-nu+mu, nu), (x, mu), (x-nu, nu)
            site_minus_nu = list(site); site_minus_nu[nu] = (site_minus_nu[nu] - 1) % L
            site_minus_nu_plus_mu = list(site_minus_nu); site_minus_nu_plus_mu[mu] = (site_minus_nu_plus_mu[mu] + 1) % L
            t_mu_at_xmnu = theta[tuple(site_minus_nu) + (mu,)]
            t_nu_at_xmnu_pmu = theta[tuple(site_minus_nu_plus_mu) + (nu,)]
            t_nu_at_xmnu = theta[tuple(site_minus_nu) + (nu,)]
            P_old2 = t_mu_at_xmnu + t_nu_at_xmnu_pmu - old - t_nu_at_xmnu
            P_new2 = t_mu_at_xmnu + t_nu_at_xmnu_pmu - proposed - t_nu_at_xmnu
            dS += -beta * (math.cos(P_new2) - math.cos(P_old2))

        if dS < 0 or rng.random() < math.exp(-dS):
            theta[site + (mu,)] = proposed


def tier_ratio(values: np.ndarray, top_pct: float = 1.0, bot_pct: float = 50.0) -> tuple[float, float, float]:
    sorted_desc = np.sort(values)[::-1]
    n = len(sorted_desc)
    k1 = max(1, int(n * top_pct / 100))
    k3 = max(1, int(n * bot_pct / 100))
    t1 = float(sorted_desc[:k1].mean())
    t3 = float(sorted_desc[-k3:].mean())
    if t3 <= 0:
        nonzero = sorted_desc[sorted_desc > 0]
        t3 = float(nonzero[-max(len(nonzero)//10, 1):].mean()) if len(nonzero) else 1e-30
    return t1, t3, (t1 / t3 if t3 > 0 else float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8, help="lattice size per dimension")
    ap.add_argument("--d", type=int, default=4, help="spacetime dimension")
    ap.add_argument("--beta", type=float, default=1.0, help="inverse gauge coupling")
    ap.add_argument("--thermalise", type=int, default=200, help="MC sweeps before measurement")
    ap.add_argument("--n-samples", type=int, default=100, help="configurations for FIM average")
    ap.add_argument("--decorr", type=int, default=20, help="sweeps between measurements")
    ap.add_argument("--step-size", type=float, default=0.6, help="Metropolis proposal step")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v5_0_lattice_u1_results.json"))
    args = ap.parse_args()

    results: list[dict] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        theta = init_links(args.L, args.d, rng)
        n_params = theta.size
        print(f"seed={seed}  L={args.L}  d={args.d}  n_links={n_params:,}  beta={args.beta}", flush=True)

        # Thermalisation
        t_therm_start = time.time()
        for sweep in range(args.thermalise):
            metropolis_sweep_local(theta, args.beta, args.d, args.step_size, rng)
            if sweep == 0:
                print(f"  sweep 1 done in {time.time()-t_therm_start:.1f}s — estimated thermalisation: {args.thermalise*(time.time()-t_therm_start):.1f}s", flush=True)
        t_therm = time.time() - t_therm_start
        print(f"  thermalised in {t_therm:.1f}s", flush=True)

        # Measurement
        t_meas_start = time.time()
        fim_diag = np.zeros(n_params, dtype=np.float64)
        decorr = args.decorr
        for i in range(args.n_samples):
            for _ in range(decorr):
                metropolis_sweep_local(theta, args.beta, args.d, args.step_size, rng)
            g = grad_action(theta, args.beta, args.d).ravel()
            fim_diag += g * g
        fim_diag /= args.n_samples
        t_meas = time.time() - t_meas_start

        t1, t3, ratio = tier_ratio(fim_diag)
        print(f"  measured in {t_meas:.1f}s  T1={t1:.4e}  T3={t3:.4e}  T1/T3={ratio:.1f}", flush=True)

        results.append({
            "seed": seed, "L": args.L, "d": args.d, "beta": args.beta,
            "n_params": int(n_params),
            "fim_tier1_mean": t1, "fim_tier3_mean": t3, "fim_tier1_tier3": ratio,
            "thermalise_s": t_therm, "measure_s": t_meas,
        })

    ratios = np.array([r["fim_tier1_tier3"] for r in results])
    payload = {
        "config": vars(args),
        "results": results,
        "aggregate": {
            "tier_ratio_mean": float(ratios.mean()),
            "tier_ratio_std": float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0,
            "tier_ratio_cv": float(ratios.std(ddof=1) / ratios.mean()) if len(ratios) > 1 and ratios.mean() > 0 else 0.0,
        },
        "comparison_context": {
            "cosmology_NN_trained_W_256": 404,
            "QEC_NN_trained_W_256": 1762,
            "untrained_NN_W_256": 1500,
            "boolean_circuit_N_384": 101_000_000,
            "random_matrix_N_3003": 81,
            "ising_chain_N_256": 2.6,
            "harmonic_chain_N_128": 3.7,
            "cellular_automaton_N_128": 3.8,
        },
        "interpretation_guide": (
            "If T1/T3 is in the O(1-10) range, lattice U(1) gauge theory does NOT exhibit the "
            "three-tier hierarchy seen in trained NNs and boolean circuits. The 'layered sequential "
            "computation' universality framing (adopted after the V4.0 BC-contradiction audit) is "
            "vindicated, and NN universality does NOT extend to known physics via this substrate. "
            "If T1/T3 is >1000, the universality class extends to at least U(1) gauge theory, and "
            "the FIM-Onsager cosmological framing gains empirical support."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {args.out}", flush=True)
    print(f"U(1) lattice FIM T1/T3: mean={payload['aggregate']['tier_ratio_mean']:.1f}  CV={payload['aggregate']['tier_ratio_cv']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
