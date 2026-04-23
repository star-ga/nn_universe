"""V3.1 mock observational pipeline — α-drift × information density.

Simulates the full end-to-end test protocol from
``docs/v3_1_alpha_drift_prediction.md`` §7 using synthetic data. This lets
us:

1. Quantify the *power* of the proposed statistical test before touching
   any real archive. If the test cannot detect the predicted correlation
   even in a noiseless mock, V3.1 should not be run on real data.
2. Compute false-positive / false-negative rates, ROC curves, and
   effective sample sizes.
3. Produce pre-registrable thresholds: "with N systems drawn from a
   DR18-density field we reject the null at 5σ with power 1 − β = ...".

The pipeline is deliberately low-dependency: numpy + scipy + json only.
No cosmological datasets are required; all ρ_I samples are drawn from
either a log-normal toy density model or a user-supplied ``density.npy``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
try:
    from scipy import stats as sp_stats
except ImportError:  # pragma: no cover
    sp_stats = None


# ----------------------------------------------------------------------
#  Physical prediction (from docs/v3_1_alpha_drift_prediction.md §1, §4)
# ----------------------------------------------------------------------

KAPPA = 4.0e-59  # yr^-1 bit^-1 Mpc^3  — best-estimate from Tier-1 FIM
RHO_MEAN_BITS_MPC3 = 2.5e17  # mean cosmic information density proxy
PREDICTED_MEAN_DRIFT_PER_YR = KAPPA * RHO_MEAN_BITS_MPC3  # ~1e-41 (dimensionless/yr). See §1 caveats.


@dataclass
class MockConfig:
    n_sightlines: int = 1000
    z_min: float = 0.5
    z_max: float = 3.0
    density_sigma: float = 1.0  # log-normal scatter in ρ_I
    alpha_noise_sigma: float = 1e-6  # measurement scatter (per-system)
    kappa_true: float = KAPPA
    rho_mean: float = RHO_MEAN_BITS_MPC3
    seed: int = 42
    # For real-archive-style sample sizes, see §5 power analysis in the docs.


def synth_density_field(cfg: MockConfig, rng: np.random.Generator) -> np.ndarray:
    """Log-normal density proxy. Returns ρ_I in bits/Mpc^3."""
    mu = math.log(cfg.rho_mean) - 0.5 * cfg.density_sigma ** 2
    return rng.lognormal(mean=mu, sigma=cfg.density_sigma, size=cfg.n_sightlines)


def synth_redshifts(cfg: MockConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(cfg.z_min, cfg.z_max, size=cfg.n_sightlines)


def synth_alpha_drifts(cfg: MockConfig, rho: np.ndarray, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject the theoretical correlation: Δα/α = κ * ρ_I * t_lookback + noise.

    t_lookback is a crude z-dependent handle (Gpc-yr-ish); the absolute
    scale gets absorbed into the fitted slope, so only *relative* spread
    matters for the hypothesis test.
    """
    # t_lookback ∝ z for z≪1, saturates for large z; use a simple bounded proxy.
    t_lookback_gyr = 10.0 * z / (1.0 + z)
    t_lookback_yr = t_lookback_gyr * 1e9
    signal = cfg.kappa_true * rho * t_lookback_yr
    noise = rng.normal(0.0, cfg.alpha_noise_sigma, size=cfg.n_sightlines)
    return signal + noise


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Pearson partial correlation of (x, y) controlling for z.

    Returns (r, p) where p is the two-tailed p-value.
    """
    xr = x - np.polyval(np.polyfit(z, x, 1), z)
    yr = y - np.polyval(np.polyfit(z, y, 1), z)
    # Use scipy when available for proper p-value; otherwise fall back to a
    # t-statistic computed by hand.
    if sp_stats is not None:
        r, p = sp_stats.pearsonr(xr, yr)
        return float(r), float(p)
    n = len(x)
    r = float(np.corrcoef(xr, yr)[0, 1])
    # Student-t with n-3 degrees of freedom for partial correlation.
    df = max(n - 3, 1)
    t = r * math.sqrt(df) / math.sqrt(max(1.0 - r ** 2, 1e-12))
    # Two-tailed p (no scipy): approximate via normal for large df.
    z_score = t
    p = 2.0 * 0.5 * math.erfc(abs(z_score) / math.sqrt(2.0))
    return r, p


def run_trial(cfg: MockConfig, h0: bool) -> dict:
    """Single trial.

    If ``h0`` is True, the injected κ is zero (null hypothesis).
    """
    rng = np.random.default_rng(cfg.seed)
    rho = synth_density_field(cfg, rng)
    z = synth_redshifts(cfg, rng)
    kappa = 0.0 if h0 else cfg.kappa_true
    dalpha = synth_alpha_drifts(
        MockConfig(**{**asdict(cfg), "kappa_true": kappa}), rho, z, rng
    )
    x = np.log(rho)
    r, p = partial_correlation(x, dalpha, z)
    return {"r": r, "p_value": p, "n": cfg.n_sightlines}


def power_analysis(cfg: MockConfig, n_trials: int = 500, seeds: int = 500) -> dict:
    """Monte Carlo power analysis.

    Repeats the trial ``seeds`` times under H1 and under H0; counts at what
    rate the partial correlation test rejects H0 at α=0.05 and α=3σ/5σ.
    """
    h0_p = []
    h1_p = []
    base = cfg.seed
    for i in range(seeds):
        cfg_i = MockConfig(**{**asdict(cfg), "seed": base + i})
        h0_p.append(run_trial(cfg_i, h0=True)["p_value"])
        h1_p.append(run_trial(cfg_i, h0=False)["p_value"])
    h0_p = np.array(h0_p)
    h1_p = np.array(h1_p)

    # 5σ two-tailed: p < 5.7e-7
    thresholds = {"p05": 0.05, "p01": 0.01, "p001": 0.001, "p3sigma": 0.0027, "p5sigma": 5.73e-7}
    results = {}
    for name, alpha in thresholds.items():
        type_i = float(np.mean(h0_p < alpha))          # false-positive rate
        power = float(np.mean(h1_p < alpha))           # true-positive rate
        results[name] = {"alpha": alpha, "false_positive_rate": type_i, "power": power}
    return {"thresholds": results, "h0_p_median": float(np.median(h0_p)), "h1_p_median": float(np.median(h1_p))}


def roc_curve(cfg: MockConfig, seeds: int = 200) -> list[dict]:
    """Approximate ROC curve by sweeping the p-value threshold."""
    h0_p, h1_p = [], []
    base = cfg.seed
    for i in range(seeds):
        cfg_i = MockConfig(**{**asdict(cfg), "seed": base + i})
        h0_p.append(run_trial(cfg_i, h0=True)["p_value"])
        h1_p.append(run_trial(cfg_i, h0=False)["p_value"])
    h0_p = np.sort(h0_p)
    h1_p = np.sort(h1_p)
    thresholds = np.geomspace(1e-10, 1.0, 40)
    curve = []
    for t in thresholds:
        fpr = float(np.mean(h0_p < t))
        tpr = float(np.mean(h1_p < t))
        curve.append({"threshold": float(t), "fpr": fpr, "tpr": tpr})
    return curve


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=300)
    p.add_argument(
        "--kappa-scale",
        type=float,
        default=1.0,
        help="multiplier on the physical κ=4e-59. 1.0 = undetectable; use ~1e22 to test pipeline.",
    )
    p.add_argument("--noise-sigma", type=float, default=1e-6)
    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "mock_results.json"))
    return p.parse_args()


def main() -> int:
    args = _parse()
    cfg = MockConfig(n_sightlines=args.n, kappa_true=KAPPA * args.kappa_scale, alpha_noise_sigma=args.noise_sigma)
    print(
        f"V3.1 mock α-drift pipeline: N={cfg.n_sightlines}  "
        f"κ_true={cfg.kappa_true:.1e} (scale {args.kappa_scale:.1e}×physical)  "
        f"noise σ={cfg.alpha_noise_sigma:.1e}"
    )

    # Single-trial sanity check.
    trial_h1 = run_trial(cfg, h0=False)
    trial_h0 = run_trial(cfg, h0=True)
    print(f"  Single trial  H1: r={trial_h1['r']:+.3f}  p={trial_h1['p_value']:.2e}")
    print(f"  Single trial  H0: r={trial_h0['r']:+.3f}  p={trial_h0['p_value']:.2e}")

    pa = power_analysis(cfg, seeds=args.seeds)
    print("  Power analysis (seeds={}):".format(args.seeds))
    for name, d in pa["thresholds"].items():
        print(f"    {name:>8s}  α={d['alpha']:.2e}  FPR={d['false_positive_rate']:.3f}  power={d['power']:.3f}")
    roc = roc_curve(cfg, seeds=min(args.seeds, 200))
    payload = {
        "config": asdict(cfg),
        "kappa_scale_vs_physical": args.kappa_scale,
        "single_trial_h0": trial_h0,
        "single_trial_h1": trial_h1,
        "power": pa,
        "roc": roc,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
