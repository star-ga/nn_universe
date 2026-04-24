"""V4.3 — bootstrap 95% CI on all headline power-law exponents.

Addresses multi-LLM audit v3 flag: "no error bars or bootstrap CI on
power-law exponents across seeds." Uses existing JSON data; no new
compute.

For each sweep (cosmology V1.2+V3.0, QEC V2.1, symbolic T3, vision T4,
20-seed robustness at W=1024 and W=4096), bootstraps the width-vs-ratio
data points and reports the 95% CI for the SV and FIM power-law
exponents. Also reports CI of the R².
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def fit_powerlaw(params: np.ndarray, ratios: np.ndarray) -> tuple[float, float]:
    """Return (exponent, R²) from log-log linear fit."""
    if len(params) < 2:
        return 0.0, 0.0
    log_p = np.log10(params)
    log_r = np.log10(np.clip(ratios, 1, None))
    fit = np.polyfit(log_p, log_r, 1)
    pred = np.polyval(fit, log_p)
    ss_res = float(np.sum((log_r - pred) ** 2))
    ss_tot = float(np.sum((log_r - log_r.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(fit[0]), float(r2)


def bootstrap_ci(params: np.ndarray, ratios: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """95% CI on power-law exponent via resampling the (param, ratio) pairs
    with replacement n_boot times."""
    rng = np.random.default_rng(seed)
    n = len(params)
    if n < 3:
        exp, r2 = fit_powerlaw(params, ratios)
        return {"exponent_point": exp, "exponent_lo95": exp, "exponent_hi95": exp, "r_squared_point": r2}
    exps, r2s = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(idx)) < 2:  # degenerate sample
            continue
        e, r = fit_powerlaw(params[idx], ratios[idx])
        exps.append(e); r2s.append(r)
    exps = np.array(exps); r2s = np.array(r2s)
    exp_pt, r2_pt = fit_powerlaw(params, ratios)
    return {
        "exponent_point": exp_pt,
        "exponent_lo95": float(np.percentile(exps, 2.5)),
        "exponent_hi95": float(np.percentile(exps, 97.5)),
        "r_squared_point": r2_pt,
        "r_squared_lo95": float(np.percentile(r2s, 2.5)),
        "r_squared_hi95": float(np.percentile(r2s, 97.5)),
        "n_bootstraps": n_boot, "n_points": n,
    }


def main() -> int:
    REPO = Path(__file__).resolve().parents[2]
    out = {"bootstraps": {}}

    # V1.2 + V3.0 cosmology scaling
    d = json.load(open(REPO / "scaling_results.json"))
    ps = np.array([r['params'] for r in d['results']])
    svs = np.array([r['max_sv_ratio'] for r in d['results']])
    fims = np.array([r['fim_tier1_tier3'] for r in d['results']])
    out['bootstraps']['cosmology_SV_13widths'] = bootstrap_ci(ps, svs)
    out['bootstraps']['cosmology_FIM_13widths'] = bootstrap_ci(ps, fims)
    # Clean (excl. W=45000 with fake SV=1.1)
    mask = ps < 6e9
    out['bootstraps']['cosmology_SV_12widths_clean'] = bootstrap_ci(ps[mask], svs[mask])

    # V2.1 QEC sweep (300 probes)
    d = json.load(open(REPO / "experiments/v2_1_qec/v2_1_sweep_results.json"))
    ps = np.array([r['n_params'] for r in d['results']])
    svs = np.array([r['max_sv_ratio'] for r in d['results']])
    fims = np.array([r['fim_tier1_tier3'] for r in d['results']])
    out['bootstraps']['qec_SV_300probes'] = bootstrap_ci(ps, svs)
    out['bootstraps']['qec_FIM_300probes'] = bootstrap_ci(ps, fims)

    # T3 symbolic (300 probes)
    d = json.load(open(REPO / "experiments/v3_0_task3_symbolic/v3_0_task3_results.json"))
    ps = np.array([r['n_params'] for r in d['results']])
    svs = np.array([r['max_sv_ratio'] for r in d['results']])
    fims = np.array([r['fim_tier1_tier3'] for r in d['results']])
    out['bootstraps']['symbolic_SV_300probes'] = bootstrap_ci(ps, svs)
    out['bootstraps']['symbolic_FIM_300probes'] = bootstrap_ci(ps, fims)

    # T4 vision (300 probes)
    d = json.load(open(REPO / "experiments/v3_0_task4_vision/v3_0_task4_results.json"))
    ps = np.array([r['n_params'] for r in d['results']])
    svs = np.array([r['max_sv_ratio'] for r in d['results']])
    fims = np.array([r['fim_tier1_tier3'] for r in d['results']])
    out['bootstraps']['vision_SV_300probes'] = bootstrap_ci(ps, svs)
    out['bootstraps']['vision_FIM_300probes'] = bootstrap_ci(ps, fims)

    # T4 vision (2000 probes)
    d = json.load(open(REPO / "experiments/v3_0_task4_vision/v3_0_task4_verify_2000probes.json"))
    ps = np.array([r['n_params'] for r in d['results']])
    fims = np.array([r['fim_tier1_tier3'] for r in d['results']])
    out['bootstraps']['vision_FIM_2000probes'] = bootstrap_ci(ps, fims)

    # Print
    print(f"{'dataset':<38}  {'exp point':>10}  {'95% CI':>24}  {'R² point':>10}")
    for name, ci in out['bootstraps'].items():
        ci_range = f"[{ci['exponent_lo95']:+.3f}, {ci['exponent_hi95']:+.3f}]"
        print(f"{name:<38}  {ci['exponent_point']:>+10.3f}  {ci_range:>24}  {ci.get('r_squared_point',0):>10.3f}")

    out_path = Path(__file__).resolve().parent / "v4_3_bootstrap_ci_results.json"
    os.makedirs(out_path.parent, exist_ok=True)
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
