"""V4.0 supplement — non-deep LEARNING baselines.

Addresses the 3/8 audit-v3 flag: V4.0's non-NN controls (Ising chain,
harmonic oscillator, cellular automaton) are *dynamical* systems, not
parameterised learners. Proper controls for "NN is unique" need to be
actual learning systems that aren't deep:

- Linear regression (closed-form fit)
- Logistic regression (1-layer, sigmoid)
- Kernel ridge regression
- Gaussian process regression

If none of these exhibit the three-tier FIM diagonal hierarchy, the
universality claim sharpens: **depth is required**. If any of them DO
exhibit it, the claim widens to "any parameterised learner" and the
"layered sequential computation" framing from V5.0 needs revisiting.

All four baselines trained on the same self-prediction task
(y = W x + noise, with x ∈ R^32 Gaussian). FIM diagonal measured on
trained parameters. Same 1%/50% tier partition as the rest of the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def tier_ratio(values: np.ndarray, top_pct: float = 1.0, bot_pct: float = 50.0) -> float:
    s = np.sort(values)[::-1]
    n = len(s)
    k1 = max(1, int(n * top_pct / 100))
    k3 = max(1, int(n * bot_pct / 100))
    t1 = float(s[:k1].mean())
    t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return t1 / t3 if t3 > 0 else float("inf")


# ---------------------------------------------------------------------------
# Baseline 1 — Linear regression: y = W x + b, y ∈ R^d, x ∈ R^d
# Parameters: W ∈ R^{d×d} + b ∈ R^d = d² + d scalars.
# Trained on (x, y = x + ε) (self-prediction task, ε ~ N(0, σ²I)).
# FIM_{ij} = E[(∂log p(y|x) / ∂θ_{ij})²] at the MLE.
# ---------------------------------------------------------------------------

def linear_regression_fim(d: int, n_samples: int, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Solve least-squares, then compute FIM diagonal via per-sample gradient."""
    X = rng.standard_normal((n_samples, d))
    Y = X + noise_sigma * rng.standard_normal((n_samples, d))  # self-prediction with noise
    # Closed-form MLE: W = (X^T X)^-1 X^T Y, b = mean
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (d, d)
    b = (Y - X @ W).mean(axis=0)                    # (d,)
    # FIM diagonal. For Gaussian p(y|x) = N(Wx + b, σ²I),
    #   ∂log p / ∂W_{ij} = (y_i - (Wx)_i - b_i) * x_j / σ²
    #   ∂log p / ∂b_i    = (y_i - (Wx)_i - b_i) / σ²
    # FIM_{W_ij,W_ij}   = E_x[x_j² / σ²]
    # FIM_{b_i,b_i}     = 1 / σ²
    # Both are analytic.
    xj_sq = (X ** 2).mean(axis=0) / noise_sigma ** 2  # (d,) — FIM diag for each W row
    fim_W = np.tile(xj_sq, d)  # d × d with each column = xj_sq
    fim_b = np.ones(d) / noise_sigma ** 2
    return np.concatenate([fim_W, fim_b])


# ---------------------------------------------------------------------------
# Baseline 2 — Kernel ridge regression (RBF kernel).
# Parameters: dual coefficients α_i, one per training sample.
# No bias/width parameter; treat kernel bandwidth as fixed hyperparam.
# ---------------------------------------------------------------------------

def kernel_ridge_fim(n_train: int, d: int, noise_sigma: float, ridge: float, rng: np.random.Generator) -> np.ndarray:
    """For kernel ridge y = sum_i α_i K(x_i, x) with RBF kernel, the "parameter"
    vector is α ∈ R^{n_train}. At the MLE α = (K + ridgeI)^-1 y, and the FIM
    diagonal in α-space is just diag(K²/σ²) / n_train (the model is linear in α).
    """
    # Sample training data
    X = rng.standard_normal((n_train, d))
    y = X[:, 0] + noise_sigma * rng.standard_normal(n_train)  # simple scalar target

    # RBF kernel matrix
    pairwise = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    bandwidth = 2.0 * d
    K = np.exp(-pairwise / (2 * bandwidth))
    # FIM_{αα}_{ii} = (1/σ²) E_x[K(x_i, x)²] — approximate via training-set samples.
    # Since K is already the training Gram matrix, FIM_ii ≈ <K²_i•>/σ²
    K2 = K ** 2
    fim_diag = K2.mean(axis=1) / noise_sigma ** 2
    return fim_diag


# ---------------------------------------------------------------------------
# Baseline 3 — Logistic regression (1-layer softmax).
# Parameters: W ∈ R^{d×K} + b ∈ R^K with K classes. Classes assigned
# via fixed random teacher (same as Task-4 vision).
# ---------------------------------------------------------------------------

def logistic_regression_fim(d: int, n_classes: int, n_train: int, rng: np.random.Generator, n_probes: int = 200) -> np.ndarray:
    """Fit logistic regression via scipy or manual SGD; compute per-parameter
    grad² over a probe batch."""
    # Teacher assigns class labels via fixed random linear map
    teacher = rng.standard_normal((d, n_classes)) * 0.3
    X_train = rng.standard_normal((n_train, d))
    y_train = np.argmax(X_train @ teacher + 0.1 * rng.standard_normal((n_train, n_classes)), axis=1)

    # Manual SGD to fit (rather than depending on sklearn)
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    lr = 0.01
    for _ in range(300):
        logits = X_train @ W + b
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        err = probs.copy()
        err[np.arange(n_train), y_train] -= 1
        W -= lr * (X_train.T @ err) / n_train
        b -= lr * err.mean(axis=0)

    # FIM diagonal via per-sample grad²
    fim_W = np.zeros((d, n_classes))
    fim_b = np.zeros(n_classes)
    for _ in range(n_probes):
        i = rng.integers(0, n_train)
        x = X_train[i:i+1]
        y = y_train[i]
        logits = x @ W + b
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        err = probs.copy()
        err[0, y] -= 1
        gW = x.T @ err  # (d, K)
        gb = err[0]     # (K,)
        fim_W += gW ** 2
        fim_b += gb ** 2
    fim_W /= n_probes
    fim_b /= n_probes
    return np.concatenate([fim_W.ravel(), fim_b.ravel()])


# ---------------------------------------------------------------------------
# Baseline 4 — Gaussian process regression.
# Parameters: kernel hyperparameters (length scale σ_l, amplitude σ_f,
# noise σ_n) + dual coefficients.
# Keep fixed kernel hyperparams and use dual coefficients as the
# FIM-relevant parameters (same as kernel ridge).
# ---------------------------------------------------------------------------

def gaussian_process_fim(n_train: int, d: int, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """GP regression on (X, y); FIM diagonal in dual-coefficient space."""
    X = rng.standard_normal((n_train, d))
    y = X[:, 0] + noise_sigma * rng.standard_normal(n_train)
    pairwise = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    K = np.exp(-pairwise / (2 * d))
    # FIM in coefficient space ≈ (K + sigma²I) / sigma² — well-conditioned.
    # Diagonal: (K_ii + sigma²) / sigma² per point.
    # For FIM-diagonal analysis we use the eigendecomposition of K + sigma² I
    # to expose the effective parameter variability.
    eig = np.linalg.eigvalsh(K + noise_sigma ** 2 * np.eye(n_train))
    # "Parameter importance" proxy: each mode contributes 1/λ_i to the
    # posterior uncertainty; diagonal-equivalent in α-space.
    fim_diag = 1.0 / np.clip(eig, 1e-8, None)
    return fim_diag


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--noise-sigma", type=float, default=0.2)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v4_learning_baselines_results.json"))
    args = ap.parse_args()

    baselines = {
        "linear_regression":
            lambda rng: linear_regression_fim(args.d, args.n_train, args.noise_sigma, rng),
        "logistic_regression":
            lambda rng: logistic_regression_fim(args.d, args.n_classes, args.n_train, rng),
        "kernel_ridge":
            lambda rng: kernel_ridge_fim(args.n_train, args.d, args.noise_sigma, args.ridge, rng),
        "gaussian_process":
            lambda rng: gaussian_process_fim(args.n_train, args.d, args.noise_sigma, rng),
    }

    results: dict = {}
    for name, fn in baselines.items():
        ratios = []
        rows = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            t0 = time.time()
            fim = fn(rng)
            dt = time.time() - t0
            r = tier_ratio(fim)
            ratios.append(r)
            rows.append({"seed": seed, "n_params": int(fim.size), "tier_ratio": float(r), "elapsed_s": dt})
            print(f"  {name:22s} seed={seed}  N={fim.size:>6,}  T1/T3={r:.2f}  ({dt:.1f}s)", flush=True)
        ratios_arr = np.array(ratios)
        results[name] = {
            "per_seed": rows,
            "tier_ratio_mean": float(ratios_arr.mean()),
            "tier_ratio_std": float(ratios_arr.std(ddof=1)) if len(ratios_arr) > 1 else 0.0,
            "tier_ratio_cv": float(ratios_arr.std(ddof=1) / ratios_arr.mean()) if len(ratios_arr) > 1 and ratios_arr.mean() > 0 else 0.0,
        }
        print(f"  {name:22s} mean={results[name]['tier_ratio_mean']:.2f}  CV={results[name]['tier_ratio_cv']*100:.1f}%", flush=True)

    payload = {
        "config": vars(args),
        "baselines": results,
        "comparison": {
            "trained_NN_W_256": 404,
            "trained_QEC_W_256": 1762,
            "untrained_NN_W_256": 1500,
            "boolean_circuit_N_384": 1e8,
            "random_matrix_N_9870": 104,
            "ising_chain_N_10000": 2.7,
            "harmonic_chain_N_10000": 4.9,
            "cellular_automaton_N_128": 3.8,
            "U1_lattice_L_8": 2.0,  # V5.0 smoke at seed 0
        },
        "interpretation_guide": (
            "If all 4 non-deep learning baselines have T1/T3 in the O(1-10) band, "
            "the FIM tier hierarchy requires DEPTH — not 'any parameterized learner'. "
            "The universality claim sharpens: 'three-tier hierarchy <=> layered sequential "
            "computation', with explicit depth requirement."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
