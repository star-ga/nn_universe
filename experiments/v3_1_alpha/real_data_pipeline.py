"""V3.1 real-data α-drift pipeline — **STUB**.

This file establishes the exact protocol for the V3.1 prediction
($\\dot\\alpha/\\alpha = \\kappa \\rho_I(x)$) against real archival data.

Execution requires:
  1. UVES + HIRES absorption-spectrum archival data with per-sightline
     $\\Delta\\alpha/\\alpha$ (Webb / King / Murphy / Carswell work).
  2. SDSS DR18 environmental density $\\rho_I$ per sightline at
     absorber redshift (bit-density proxy from local galaxy count).
  3. Redshift z and S/N metadata for the partial-correlation controls.

None of this data is included in the repo; the script below is
a placeholder that documents the expected interface and the
decision rule. It executes against synthetic data and produces a
structurally-correct null result, so it is still testable via
`pytest tests/`.

When real data arrives, populate `load_archival_data()` to return
columns (delta_alpha_over_alpha, rho_I, z, snr, sightline_id) and
re-run. The rest of the pipeline is locked at preregistration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats

# Locked-at-preregistration constants.
N_BOOT = 2000
SEED = 42
KAPPA_PHYSICAL = 4e-59  # yr^-1 bit^-1 Mpc^3, V3.1 §1.3
NOISE_FLOOR_HIRES = 1e-6  # typical 1σ per-sightline (Webb 2011)
rng = np.random.default_rng(SEED)


def load_archival_data() -> dict:
    """Return a dict with the five required columns. REPLACE ME.

    The current stub generates a synthetic dataset at the physical κ
    which is guaranteed to produce a null at current noise.
    """
    n = 300  # matches V3.1 §2 "5σ sample size ≈ 834" requirement; n=300 is
             # representative; the real pipeline should accept n ≥ 200 if
             # bootstrap CIs are included.
    z = rng.uniform(0.5, 3.5, size=n)
    snr = rng.uniform(10, 60, size=n)
    # rho_I in bits per Mpc^3 at sightline redshift — log-normal proxy.
    rho_I = np.exp(rng.normal(0, 1, size=n))
    # True signal at physical kappa.
    signal = KAPPA_PHYSICAL * rho_I
    # Instrumental noise per sightline.
    noise = NOISE_FLOOR_HIRES * rng.standard_normal(size=n)
    delta_alpha_over_alpha = signal + noise
    return {
        "delta_alpha_over_alpha": delta_alpha_over_alpha,
        "rho_I": rho_I,
        "z": z,
        "snr": snr,
        "sightline_id": np.arange(n),
    }


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson partial correlation of (x, y) controlling for z.

    Regresses both x and y on z, correlates the residuals.
    """
    # Fit z-vs-x, z-vs-y linear regressions.
    def resid(vec: np.ndarray) -> np.ndarray:
        A = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(A, vec, rcond=None)
        return vec - A @ beta
    rx = resid(x)
    ry = resid(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_partial_r(delta_alpha: np.ndarray, rho_I: np.ndarray,
                         z: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    n = len(delta_alpha)
    point = partial_correlation(delta_alpha, np.log(rho_I), z)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(partial_correlation(
            delta_alpha[idx], np.log(rho_I[idx]), z[idx]
        ))
    boots = np.array(boots)
    ci_low = float(np.percentile(boots, 2.5))
    ci_high = float(np.percentile(boots, 97.5))
    return point, ci_low, ci_high


def null_test(delta_alpha: np.ndarray, rho_I: np.ndarray, z: np.ndarray,
              sightline_id: np.ndarray) -> float:
    """Instrument-systematics null: re-pair delta_alpha with shuffled
    rho_I (destroying the physical correlation), re-run the partial-r
    estimator, return the mean |r| across shuffles. A positive detection
    from the main analysis must exceed 2× this null's 95 %-ile to be
    credited to the framework."""
    null_rs = []
    for _ in range(200):
        perm = rng.permutation(len(rho_I))
        null_rs.append(abs(partial_correlation(delta_alpha, np.log(rho_I[perm]), z)))
    return float(np.percentile(null_rs, 95))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v3_1_real_data_results.json"))
    ap.add_argument("--synthetic", action="store_true",
                    help="Run against the synthetic stub (current default).")
    args = ap.parse_args()

    print("=== V3.1 α-drift real-data pipeline (stub) ===")
    print(f"  κ_physical   = {KAPPA_PHYSICAL:.2e}  yr^-1 bit^-1 Mpc^3")
    print(f"  HIRES noise  = {NOISE_FLOOR_HIRES:.1e}")
    print(f"  Seed         = {SEED}")
    print()

    data = load_archival_data()
    n = len(data["delta_alpha_over_alpha"])
    print(f"Loaded {n} sightlines.")

    r_point, r_low, r_high = bootstrap_partial_r(
        data["delta_alpha_over_alpha"], data["rho_I"], data["z"], n_boot=N_BOOT
    )
    null_r95 = null_test(
        data["delta_alpha_over_alpha"], data["rho_I"], data["z"],
        data["sightline_id"]
    )

    print(f"\nPartial correlation r(Δα/α, log ρ_I | z):")
    print(f"  point estimate = {r_point:+.4f}")
    print(f"  95 % CI        = [{r_low:+.4f}, {r_high:+.4f}]")
    print(f"  null-test r95  = {null_r95:.4f}")

    ci_crosses_zero = r_low < 0 < r_high
    detection_vs_null = abs(r_point) > 2 * null_r95
    decision = (
        "POSITIVE" if (not ci_crosses_zero and r_point > 0 and detection_vs_null)
        else "NULL" if ci_crosses_zero
        else "AMBIGUOUS"
    )
    print(f"\nDecision: {decision}")

    payload = {
        "kappa_physical": KAPPA_PHYSICAL,
        "hires_noise": NOISE_FLOOR_HIRES,
        "n_sightlines": n,
        "partial_r_point": r_point,
        "partial_r_95ci": [r_low, r_high],
        "null_r95": null_r95,
        "decision": decision,
        "locked_protocol_version": "V2_preregistration_2026-04-24",
        "status_note": (
            "This run used the synthetic stub. When real archival UVES+HIRES "
            "× SDSS DR18 data is available, replace load_archival_data() "
            "and re-run. The decision rule, bootstrap protocol, null-test "
            "construction, and partial-correlation formula are locked and "
            "must not be altered between the preregistration commit and "
            "the real-data execution."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
