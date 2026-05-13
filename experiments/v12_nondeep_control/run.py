"""V12 item 5 — parameter-matched non-deep production-scale control.

Closes the audit gap: every production-scale checkpoint measured so
far (ResNet-50, ViT-L/16, GPT-2-large, Pythia-6.9B, OLMoE, Mamba) is
deep-sequential.  No production-scale *non-deep* comparator exists,
so reviewers can argue the dichotomy might be an effect of model size
alone, not depth + sequential composition.

This script measures the FIM diagonal of a **300M-parameter random-
feature ridge regression** (RFF kernel ridge): a shallow learner
parameter-matched to ViT-L/16 (304M params).  No depth.  No
sequential composition.  If the dichotomy survives, the audit gap
closes; if it fails, we narrow the claim honestly.

Hardware: fits in 12 GB VRAM (FP32 ops, no model weights to store —
just the regression coefficients).  Wall-clock ~30 minutes per seed
on RTX 4070.

Usage:
    python3 experiments/v12_nondeep_control/run.py \
        --substrate rff_kernel_ridge_300m \
        --seed 0 \
        --out  results/<substrate>_seed<n>.json \
        --raw-fim-out results/<substrate>_seed<n>_raw.npy
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64); x = x[x >= 0]
    if x.size == 0 or x.sum() == 0: return 0.0
    x = np.sort(x); n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def effective_rank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64); x = x[x > 0]
    if x.size == 0: return 0.0
    return float((x.sum() ** 2) / (x ** 2).sum())


def top_k_mass(x: np.ndarray, k_frac: float = 0.01) -> float:
    x = np.asarray(x, dtype=np.float64); x = x[x >= 0]
    if x.size == 0 or x.sum() == 0: return 0.0
    k = max(1, int(np.ceil(k_frac * x.size)))
    return float(np.sort(x)[-k:].sum() / x.sum())


def t1_t3(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64); x = x[x >= 0]
    if x.size < 100: return float("nan")
    x = np.sort(x); n = x.size
    bot = x[: n // 2].mean()
    top = x[-max(1, n // 100):].mean()
    return float(top / bot) if bot > 0 else float("inf")


def rff_kernel_ridge_fim(seed: int, n_params: int = 300_000_000,
                         n_probes: int = 200, regularisation: float = 1e-3):
    """FIM diagonal of a 300M-param random-feature ridge regression.

    Model: y_hat = w^T phi(x), phi(x) = cos(W x + b) random Fourier features.
    The "parameters" are the n_params entries of w (RFF coefficients).
    NB: W and b are fixed random features, NOT parameters — phi(x) is a
    fixed lookup table from x to a random Fourier embedding.

    FIM diagonal under MSE loss:
        F_ii = E_x[ phi(x)_i^2 ]  (since d/dw_i (y_hat - y)^2 = 2 (y_hat - y) phi(x)_i,
                                   and at random init E[(y_hat - y)^2] is bounded)
    We compute it directly by averaging phi(x)_i^2 over n_probes Gaussian probes.

    Memory plan: 300M-entry vector × FP32 = 1.2 GB. Fits trivially.
    Time: ~30 min on RTX 4070 (matrix-free streaming RFF).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # We compute one parameter slice at a time to avoid materialising
    # the 300M-element RFF feature vector in memory all at once.
    SLICE = 1_000_000  # 1M-entry slices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fim_acc = torch.zeros(n_params, dtype=torch.float32, device="cpu")

    INPUT_DIM = 1024  # matches ViT-L/16 input embedding dim
    print(f"  rff_kernel_ridge: n_params={n_params:,}, input_dim={INPUT_DIM}, n_probes={n_probes}", flush=True)

    for slice_start in range(0, n_params, SLICE):
        slice_end = min(slice_start + SLICE, n_params)
        slice_n = slice_end - slice_start

        # Per-slice fixed random projection W: shape (slice_n, INPUT_DIM)
        slice_rng = np.random.default_rng(seed * 2**32 + slice_start)
        W = torch.tensor(slice_rng.standard_normal((slice_n, INPUT_DIM)).astype(np.float32),
                         device=device)
        b = torch.tensor(slice_rng.uniform(0, 2 * np.pi, slice_n).astype(np.float32),
                         device=device)

        slice_acc = torch.zeros(slice_n, dtype=torch.float32, device=device)
        for p in range(n_probes):
            x_rng = np.random.default_rng(seed + p)
            x = torch.tensor(x_rng.standard_normal(INPUT_DIM).astype(np.float32),
                             device=device)
            phi = torch.cos(W @ x + b)                # (slice_n,)
            slice_acc += phi ** 2

        slice_acc /= n_probes
        fim_acc[slice_start:slice_end] = slice_acc.cpu()
        if slice_start % (10 * SLICE) == 0:
            print(f"    slice {slice_start:,}/{n_params:,} done", flush=True)
        del W, b, slice_acc

    return fim_acc.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="rff_kernel_ridge_300m")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-fim-out", required=True)
    ap.add_argument("--n-params", type=int, default=300_000_000,
                    help="Number of RFF coefficients (param-matched to ViT-L/16=304M)")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.raw_fim_out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fim = rff_kernel_ridge_fim(seed=args.seed, n_params=args.n_params, n_probes=200)
    elapsed = time.time() - t0

    np.save(args.raw_fim_out, fim.astype(np.float32))

    summary = {
        "substrate":          args.substrate,
        "seed":               args.seed,
        "n_params":           int(fim.size),
        "n_probes":           200,
        "elapsed_s":          round(elapsed, 1),
        "t1_t3":              t1_t3(fim),
        "log10_t1_t3":        float(np.log10(t1_t3(fim))) if np.isfinite(t1_t3(fim)) else None,
        "gini":               gini(fim),
        "effective_rank_n":   effective_rank(fim) / fim.size,
        "top_1pct_mass":      top_k_mass(fim, 0.01),
        "raw_fim_path":       args.raw_fim_out,
        "param_match_note":   "300M params = ViT-L/16 (304M) parameter-matched control; shallow non-deep architecture",
        "predicted_t1_t3":    "< 6 (matching V4.5 shallow learners)",
        "predicted_band":     "rest (T_1/T_3 < 100)",
        "falsifier":          "T_1/T_3 > 100 falsifies the depth + sequential composition primitive hypothesis",
        "schema_version":     "v12.1",
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"OK  {args.substrate} seed={args.seed} t1_t3={summary['t1_t3']:.3g} "
          f"(predicted < 6, falsifies if > 100) ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
