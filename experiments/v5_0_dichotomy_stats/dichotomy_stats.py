"""
dichotomy_stats.py
==================
Bootstrap 95% CIs and Mann-Whitney U test for the V5.0 dichotomy claim:
"deep layered sequential" systems (T1/T3 >= 10^3) vs "everything else" (<= 104).

Data sources:
  - v4_learning_baselines_results.json   : linear/kernel/logistic/GP (5 seeds)
  - v5_0_lattice_u1_results.json         : U(1) lattice (3 seeds)
  - v4_0_trained_vs_untrained.json       : trained & untrained NN, 4 widths x 5 seeds
  - v4_0_uniqueness_results.json         : NN, random matrix, Ising, harmonic, CA, boolean circuit (6 seeds)

Method:
  - Log-transformed bootstrap (2000 resamples) for all systems.
  - Systems with < 3 seeds: normal approximation on log scale noted.
  - Threshold: T1/T3 = 100 (CI crossing noted).
  - Mann-Whitney U: deep-layered vs rest, on log(T1/T3) pooled seed values.
"""

import json
import math
import pathlib
import numpy as np
from scipy.stats import mannwhitneyu, norm

SEED = 42
N_BOOT = 2000
THRESHOLD = 100.0   # dichotomy boundary
rng = np.random.default_rng(SEED)

BASE = pathlib.Path("/home/n/nn_universe/experiments")

# ── load raw per-seed T1/T3 values ───────────────────────────────────────────

def load_baselines():
    """linear, logistic, kernel_ridge, gaussian_process — 5 seeds each."""
    path = BASE / "v4_0_uniqueness/v4_learning_baselines_results.json"
    d = json.loads(path.read_text())
    systems = {}
    for name, blob in d["baselines"].items():
        systems[name] = [r["tier_ratio"] for r in blob["per_seed"]]
    return systems

def load_u1():
    """U(1) lattice — 3 seeds."""
    path = BASE / "v5_0_lattice_qcd/v5_0_lattice_u1_results.json"
    d = json.loads(path.read_text())
    return {"u1_lattice": [r["fim_tier1_tier3"] for r in d["results"]]}


def load_su2():
    """SU(2) non-abelian lattice — 3 seeds."""
    path = BASE / "v7_0_lattice_su2/v7_0_lattice_su2_results.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {"su2_lattice": [r["fim_tier1_tier3"] for r in d["results"]]}

def load_trained_untrained():
    """Trained & untrained NN across 4 widths — pool all 20 seeds per class."""
    path = BASE / "v4_0_uniqueness/v4_0_trained_vs_untrained.json"
    d = json.loads(path.read_text())
    trained, untrained = [], []
    for rec in d["results"]:
        trained.extend(rec["trained"]["per_seed"])
        untrained.extend(rec["untrained"]["per_seed"])
    return {"nn_trained_pooled": trained, "nn_untrained_pooled": untrained}

def load_uniqueness():
    """NN, random_matrix, ising_chain, harmonic_chain, boolean_circuit, cellular_automaton — 6 seeds."""
    path = BASE / "v4_0_uniqueness/v4_0_uniqueness_results.json"
    d = json.loads(path.read_text())
    systems = {}
    for name, blob in d["baselines"].items():
        systems[name] = [r["tier_ratio"] for r in blob["per_seed"]]
    return systems

# ── bootstrap CI (log-transformed) ───────────────────────────────────────────

def bootstrap_log_ci(values, n_boot=N_BOOT, alpha=0.05):
    """
    Bootstrap on log(values), exponentiate CI bounds.
    Returns (point_estimate, ci_low, ci_high, method).
    Uses normal approximation when n < 3.
    """
    vals = np.array(values, dtype=float)
    log_vals = np.log(vals[vals > 0])  # guard against exact-zero tier3 edge cases
    n = len(log_vals)

    point = float(np.exp(np.mean(log_vals)))

    if n < 3:
        # Normal approximation on log scale
        se = float(np.std(log_vals, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        z = norm.ppf(1 - alpha / 2)
        mean_log = float(np.mean(log_vals))
        ci_low = math.exp(mean_log - z * se)
        ci_high = math.exp(mean_log + z * se)
        method = "normal_approx"
    else:
        boot_means = np.array([
            np.mean(rng.choice(log_vals, size=n, replace=True))
            for _ in range(n_boot)
        ])
        ci_low = float(np.exp(np.percentile(boot_means, 100 * alpha / 2)))
        ci_high = float(np.exp(np.percentile(boot_means, 100 * (1 - alpha / 2))))
        method = "bootstrap"

    return point, ci_low, ci_high, method, n

# ── assemble all systems ──────────────────────────────────────────────────────

def build_system_table():
    all_data = {}
    all_data.update(load_baselines())
    all_data.update(load_u1())
    all_data.update(load_su2())
    all_data.update(load_trained_untrained())
    uniq = load_uniqueness()

    # boolean_circuit has some seeds with tier3=0 → extremely large ratios.
    # Use only finite non-inf values for the bootstrap; note the issue.
    bc_raw = uniq.pop("boolean_circuit")
    bc_finite = [v for v in bc_raw if np.isfinite(v) and v > 0]
    # Fail fast rather than silently drop BC from the Mann–Whitney test if
    # the source data ever has no finite ratios.
    if len(bc_finite) == 0:
        raise RuntimeError(
            "boolean_circuit has zero finite T1/T3 values — MW test would "
            "silently drop the BC group. Check v4_0_uniqueness_results.json."
        )
    all_data["boolean_circuit"] = bc_finite
    all_data["boolean_circuit_raw_n"] = len(bc_raw)
    all_data["boolean_circuit_finite_n"] = len(bc_finite)

    # Remove NN from uniqueness (it's superseded by the trained_vs_untrained pooled data)
    uniq.pop("neural_network", None)
    all_data.update(uniq)

    return all_data

# ── dichotomy group assignment ────────────────────────────────────────────────

DEEP_SEQUENTIAL = {
    "nn_trained_pooled",
    "nn_untrained_pooled",
    "boolean_circuit",
}

REST = {
    "linear_regression",
    "kernel_ridge",
    "logistic_regression",
    "gaussian_process",
    "u1_lattice",
    "su2_lattice",
    "ising_chain",
    "harmonic_chain",
    "cellular_automaton",
    "random_matrix",
}

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    all_data = build_system_table()

    # Remove metadata keys
    systems = {k: v for k, v in all_data.items()
               if isinstance(v, list) and k not in ("boolean_circuit_raw_n",)}

    results = {}
    n_systems = len(systems)
    bonferroni_alpha = 0.05 / n_systems  # FWER-corrected per-system alpha
    for name, vals in systems.items():
        point, ci_low, ci_high, method, n = bootstrap_log_ci(vals)
        # Bonferroni-corrected CI at alpha = 0.05 / n_systems.
        _, bonf_low, bonf_high, _, _ = bootstrap_log_ci(vals, alpha=bonferroni_alpha)
        results[name] = {
            "n_seeds": n,
            "point_estimate": round(point, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "ci_bonferroni_low": round(bonf_low, 4),
            "ci_bonferroni_high": round(bonf_high, 4),
            "method": method,
            "ci_crosses_threshold_100": ci_low < THRESHOLD < ci_high,
            "ci_bonferroni_crosses_threshold_100": bonf_low < THRESHOLD < bonf_high,
            "above_threshold_1000": point >= 1000.0,
            "group": "deep_sequential" if name in DEEP_SEQUENTIAL else "rest",
        }

    # ── Mann-Whitney U test on log(T1/T3) ─────────────────────────────────────
    deep_vals = []
    rest_vals = []
    for name, vals in systems.items():
        log_vals = np.log([v for v in vals if v > 0]).tolist()
        if name in DEEP_SEQUENTIAL:
            deep_vals.extend(log_vals)
        elif name in REST:
            rest_vals.extend(log_vals)

    stat, p_value = mannwhitneyu(deep_vals, rest_vals, alternative="greater")
    effect_size_r = stat / (len(deep_vals) * len(rest_vals))   # rank-biserial r

    # ── output ────────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "n_bootstrap": N_BOOT,
            "threshold": THRESHOLD,
            "alpha": 0.05,
            "n_systems": n_systems,
            "bonferroni_alpha": bonferroni_alpha,
            "bonferroni_note": (
                f"Family-wise error rate controlled by Bonferroni correction "
                f"alpha=0.05/{n_systems}={bonferroni_alpha:.5f}. The headline "
                "Mann-Whitney U test on group separation tests a single hypothesis "
                "and does not require correction; the per-system CIs are reported "
                "at both 95% and Bonferroni-corrected levels for transparency."
            ),
            "ci_method": "log_transformed_bootstrap_percentile (normal_approx if n<3)",
            "test": "Mann-Whitney U (one-sided: deep > rest) on log(T1/T3)",
            "boolean_circuit_note": (
                f"BC: {all_data['boolean_circuit_finite_n']} finite seeds used "
                f"(raw n={all_data['boolean_circuit_raw_n']}; seeds with T3=0 "
                "produce infinite ratios — excluded from log-bootstrap but verified "
                "genuine in v4_0_uniqueness_results.json boolean_circuit_verification block)"
            ),
        },
        "systems": results,
        "mann_whitney": {
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size_r": round(float(effect_size_r), 4),
            "n_deep_obs": len(deep_vals),
            "n_rest_obs": len(rest_vals),
            "interpretation": (
                "Reject H0 (deep == rest) at alpha=0.05"
                if p_value < 0.05 else
                "Fail to reject H0 at alpha=0.05"
            ),
        },
    }

    out_path = pathlib.Path(__file__).parent / "dichotomy_stats_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n=== V5.0 DICHOTOMY — 95% CI TABLE ===")
    print(f"{'System':<30} {'Group':<17} {'n':>3}  {'Point':>12}  {'CI 95% low':>12}  {'CI 95% high':>12}  {'CI > 100':>9}  Method")
    print("-" * 115)
    order = sorted(results.items(), key=lambda x: -x[1]["point_estimate"])
    for name, r in order:
        flag = "YES" if r["ci_95_low"] > THRESHOLD else ("CROSS" if r["ci_crosses_threshold_100"] else "no")
        print(f"{name:<30} {r['group']:<17} {r['n_seeds']:>3}  {r['point_estimate']:>12.2f}  "
              f"{r['ci_95_low']:>12.4f}  {r['ci_95_high']:>12.2f}  {flag:>9}  {r['method']}")

    mw = output["mann_whitney"]
    print(f"\n=== MANN-WHITNEY U TEST ===")
    print(f"  U = {mw['statistic']:.0f},  p = {mw['p_value']:.2e},  r = {mw['effect_size_r']:.4f}")
    print(f"  n(deep) = {mw['n_deep_obs']},  n(rest) = {mw['n_rest_obs']}")
    print(f"  {mw['interpretation']}")

    return output

if __name__ == "__main__":
    main()
