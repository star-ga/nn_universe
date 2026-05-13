"""V12 item 1 — partition-invariant statistics across the full 13-substrate panel.

Closes the audit gap: §4.5 partition-invariant verification (Gini /
effective rank / top-1 % mass) is currently only computed for untrained
MLP depth sweeps + baselines.  This script re-runs the §4.5 panel,
retains the raw FIM diagonal arrays, and computes the three
partition-free statistics per (substrate, seed).

Usage:
    python3 experiments/v12_partition_invariant/run.py \
        --substrate <substrate_id> \
        --seed <int> \
        --out  results/<substrate>_seed<n>.json \
        --raw-fim-out results/<substrate>_seed<n>_raw.npy

Substrate IDs (must match the work-queue manifest in
scripts/run_v12_cluster.sh):
    mlp_trained_w200, cnn_trained_w200, vit_trained_w200,
    mlp_untrained_w200, boolean_circuit,
    linear_regression, logistic_regression, kernel_ridge, gaussian_process,
    u1_lattice_L8, su2_lattice_L3,
    ising_chain_N256, harmonic_chain_N256, cellular_automaton_R110,
    random_matrix_GOE
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def gini(x: np.ndarray) -> float:
    """Gini coefficient of nonnegative values (0 = uniform, 1 = single-spike)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def effective_rank(x: np.ndarray) -> float:
    """Effective rank = (sum x)^2 / sum(x^2). Normalised by n in caller."""
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0]
    if x.size == 0:
        return 0.0
    s1 = x.sum()
    s2 = (x ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def top_k_mass(x: np.ndarray, k_frac: float = 0.01) -> float:
    """Fraction of total mass in top-k entries."""
    x = np.asarray(x, dtype=np.float64)
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    k = max(1, int(np.ceil(k_frac * x.size)))
    return float(np.sort(x)[-k:].sum() / x.sum())


def t1_t3(x: np.ndarray) -> float:
    """V1.0 partition: top 1% / bottom 50% mean ratio (load-bearing observable)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[x >= 0]
    if x.size < 100:
        return float("nan")
    x = np.sort(x)
    n = x.size
    bottom_50 = x[: n // 2].mean()
    top_1 = x[-max(1, n // 100):].mean()
    return float(top_1 / bottom_50) if bottom_50 > 0 else float("inf")


# ---------------------------------------------------------- substrate dispatch
def fim_for_substrate(substrate: str, seed: int) -> np.ndarray:
    """Return the FIM-diagonal (or FIM-analog) array for the given substrate.

    For neural substrates this is E[(d loss / d theta_i)^2] over $\geq 200$
    Gaussian probes; for boolean circuits this is the finite-difference
    sensitivity-analog described in §3.3.
    """
    rng = np.random.default_rng(seed)

    if substrate.startswith("mlp_trained") or substrate.startswith("mlp_untrained"):
        from experiments.v1_baseline.toy_experiment import fim_diagonal as _fim_mlp
        return _fim_mlp(width=200, depth=5, seed=seed, n_probes=200,
                        trained=substrate.startswith("mlp_trained"))

    if substrate.startswith("cnn_trained"):
        from experiments.v3_0_arch_baselines.cnn_baseline import fim_diagonal_cnn
        return fim_diagonal_cnn(seed=seed, n_probes=200, trained=True)

    if substrate.startswith("vit_trained"):
        from experiments.v3_0_arch_baselines.vit_baseline import fim_diagonal_vit
        return fim_diagonal_vit(seed=seed, n_probes=200, trained=True)

    if substrate == "boolean_circuit":
        from experiments.v4_0_uniqueness.boolean_circuit import sensitivity_analog
        return sensitivity_analog(seed=seed, depth=8, width=128, n_probes=200)

    if substrate == "linear_regression":
        from experiments.v4_0_uniqueness.learning_baselines import fim_linear
        return fim_linear(seed=seed, n_features=3000, n_probes=200)
    if substrate == "logistic_regression":
        from experiments.v4_0_uniqueness.learning_baselines import fim_logistic
        return fim_logistic(seed=seed, n_features=3000, n_probes=200)
    if substrate == "kernel_ridge":
        from experiments.v4_0_uniqueness.learning_baselines import fim_kernel_ridge
        return fim_kernel_ridge(seed=seed, n_features=3000, n_probes=200)
    if substrate == "gaussian_process":
        from experiments.v4_0_uniqueness.learning_baselines import fim_gp
        return fim_gp(seed=seed, n_features=3000, n_probes=200)

    if substrate == "u1_lattice_L8":
        from experiments.v5_0_lattice_qcd.lattice_u1 import fim_diagonal_u1
        return fim_diagonal_u1(L=8, beta=2.0, seed=seed, n_probes=200)
    if substrate == "su2_lattice_L3":
        from experiments.v7_0_lattice_su2.lattice_su2 import fim_diagonal_su2
        return fim_diagonal_su2(L=3, beta=2.4, seed=seed, n_probes=200)

    if substrate == "ising_chain_N256":
        from experiments.v4_0_uniqueness.dynamical_controls import fim_ising
        return fim_ising(N=256, T=2.27, seed=seed, n_probes=200)
    if substrate == "harmonic_chain_N256":
        from experiments.v4_0_uniqueness.dynamical_controls import fim_harmonic
        return fim_harmonic(N=256, seed=seed, n_probes=200)
    if substrate == "cellular_automaton_R110":
        from experiments.v4_0_uniqueness.dynamical_controls import fim_cellular
        return fim_cellular(rule=110, N=128, seed=seed, n_probes=200)

    if substrate == "random_matrix_GOE":
        # GOE: parameters are entries of a symmetric Gaussian random matrix
        N = 3003  # matches §4.5 panel
        M = rng.standard_normal((N, N))
        M = (M + M.T) / np.sqrt(2 * N)
        # FIM-analog: per-entry sensitivity = (eigenvector amplitude squared) at top eig
        evals, evecs = np.linalg.eigh(M)
        top = np.argmax(np.abs(evals))
        return evecs[:, top] ** 2

    raise ValueError(f"Unknown substrate: {substrate}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--raw-fim-out", required=True,
                    help="Output .npy path for raw FIM-diagonal array")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.raw_fim_out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fim = fim_for_substrate(args.substrate, args.seed)
    elapsed = time.time() - t0

    np.save(args.raw_fim_out, fim.astype(np.float32))  # half the disk vs float64

    summary = {
        "substrate":          args.substrate,
        "seed":                args.seed,
        "n_params":            int(fim.size),
        "n_probes":            200,
        "elapsed_s":           round(elapsed, 1),
        "t1_t3":               t1_t3(fim),
        "gini":                gini(fim),
        "effective_rank":      effective_rank(fim),
        "effective_rank_n":    effective_rank(fim) / fim.size if fim.size else 0.0,
        "top_1pct_mass":       top_k_mass(fim, 0.01),
        "raw_fim_path":        args.raw_fim_out,
        "fim_min":             float(np.min(fim)),
        "fim_max":             float(np.max(fim)),
        "fim_mean":            float(np.mean(fim)),
        "fim_nonzero_frac":    float(np.mean(fim > 0)),
        "schema_version":      "v12.1",
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"OK  {args.substrate} seed={args.seed} "
          f"t1_t3={summary['t1_t3']:.3g} gini={summary['gini']:.3f} "
          f"r_eff/n={summary['effective_rank_n']:.3g} "
          f"top1%={summary['top_1pct_mass']:.3f} "
          f"({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
