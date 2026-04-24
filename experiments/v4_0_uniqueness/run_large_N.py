"""V4.0 large-N uniqueness sweep — rule out the small-param-artifact critique.

The original V4.0 sweep (2026-04-23) measured the FIM tier ratio for each
baseline at n_params ~ 3000. A reviewer might object that the trained
NN (n_params 3500) shows tier ratio ~26000x simply because it's the
only baseline that *scales* to learning a non-trivial function, while the
5 non-NN baselines happen to be at smaller effective dimension.

This script re-runs the 5 non-learning baselines (random matrix,
Ising chain, harmonic chain, boolean circuit, cellular automaton) at
matched-or-larger n_params, so the "size explains it" argument fails.
Deliberately skips the NeuralNetwork baseline (already measured in
V4.0; its 20k-step training inside __init__ makes it the slowest item
and blocked the driver phase-7 attempt on 2026-04-23).

Per-baseline timeout defaults to 300s; explicit OMP thread budget
prevents CPU thrash. Seeds 0-5 match V4.0 default for direct comparison.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _run_one(spec_name: str, n_params: int, seed: int, n_probes: int, timeout: float) -> dict:
    """Run a single baseline/seed combination with a wall-clock timeout."""
    from baselines import make  # imported here so OMP env is already set

    t0 = time.time()
    sys_obj = make(spec_name, seed)
    # Override n_params if the baseline supports it. Most baselines
    # accept n_params through their __init__; see baselines.py REGISTRY.
    if hasattr(sys_obj, "n_params") and sys_obj.n_params < n_params:
        sys_obj = _make_with_size(spec_name, n_params, seed)
    imp = sys_obj.parameter_importance(n_probes=n_probes)
    elapsed = time.time() - t0
    if elapsed > timeout:
        raise RuntimeError(f"exceeded {timeout}s")
    # Compute tier ratio from importance vector.
    vals = np.sort(imp)[::-1]
    n = len(vals)
    if n < 100:
        raise RuntimeError(f"too few parameters ({n}) for meaningful tier stats")
    k1 = max(1, n // 100)
    k2 = n // 2
    tier1_mean = float(vals[:k1].mean())
    tier3_mean = float(vals[k2:].mean())
    # Guard underflow
    if tier3_mean <= 0:
        nonzero = vals[vals > 0]
        tier3_mean = float(nonzero[-max(len(nonzero)//10, 1):].mean()) if len(nonzero) else 1e-30
    ratio = tier1_mean / tier3_mean
    top1_mass = float(vals[:k1].sum() / max(vals.sum(), 1e-30))
    return {
        "seed": seed,
        "n_params_actual": int(n),
        "tier1_mean": tier1_mean,
        "tier3_mean": tier3_mean,
        "tier_ratio": ratio,
        "top1pct_mass": top1_mass,
        "elapsed_s": elapsed,
    }


def _make_with_size(name: str, n_params: int, seed: int):
    """Instantiate a baseline at a specific n_params, bypassing the registry default."""
    # The Baseline subclasses each take (n_params, seed) as __init__ args.
    from baselines import (
        RandomMatrix,
        IsingChain,
        HarmonicOscillator,
        BooleanCircuit,
        CellularAutomaton,
    )
    cls_map = {
        "random_matrix": RandomMatrix,
        "ising_chain": IsingChain,
        "harmonic_chain": HarmonicOscillator,
        "boolean_circuit": BooleanCircuit,
        "cellular_automaton": CellularAutomaton,
    }
    cls = cls_map[name]
    return cls(n_params, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--target-params", type=int, default=1_000_000,
                    help="target n_params per baseline; default 1M")
    ap.add_argument("--per-baseline-timeout", type=float, default=300.0)
    ap.add_argument("--omp-threads", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default=str(HERE / "v4_0_large_N_results.json"))
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp_threads)

    # 5 non-NN baselines; NN is already measured in the V4.0 (small-N) run.
    baselines = [
        "random_matrix",
        "ising_chain",
        "harmonic_chain",
        "boolean_circuit",
        "cellular_automaton",
    ]

    all_results: dict[str, dict] = {}
    for name in baselines:
        print(f"=== {name}  (target n_params ~ {args.target_params:,}) ===", flush=True)
        rows = []
        for seed in args.seeds:
            try:
                r = _run_one(
                    spec_name=name,
                    n_params=args.target_params,
                    seed=seed,
                    n_probes=args.n_probes,
                    timeout=args.per_baseline_timeout,
                )
            except Exception as exc:
                print(f"  seed={seed}: FAILED ({type(exc).__name__}: {exc})", flush=True)
                continue
            print(
                f"  seed={seed}  n_params_actual={r['n_params_actual']:>9,}  "
                f"t1/t3={r['tier_ratio']:>12,.2f}  top1%_mass={r['top1pct_mass']:.4f}  "
                f"({r['elapsed_s']:.1f}s)",
                flush=True,
            )
            rows.append(r)
        if not rows:
            all_results[name] = {"error": "all seeds failed"}
            continue
        ratios = np.array([r["tier_ratio"] for r in rows])
        all_results[name] = {
            "n_params_target": args.target_params,
            "per_seed": rows,
            "tier_ratio_mean": float(ratios.mean()),
            "tier_ratio_std": float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0,
            "tier_ratio_cv": float(ratios.std(ddof=1) / ratios.mean()) if len(ratios) > 1 and ratios.mean() > 0 else 0.0,
        }

    payload = {
        "baselines": all_results,
        "meta": {
            "seeds": args.seeds,
            "probes": args.n_probes,
            "target_params": args.target_params,
            "per_baseline_timeout": args.per_baseline_timeout,
            "omp_threads": args.omp_threads,
            "nn_baseline_skipped": True,
            "nn_baseline_skipped_reason": "already measured in V4.0 small-N run; 20k-step __init__ training causes driver stall",
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
