"""V4.0 supplement — trained NN vs untrained NN at matched N.

The original V4.0 critique ("small-param artifact") is answered most
cleanly by comparing the SAME network at the SAME parameter count,
differing only in whether gradient descent has been applied. If the
trained network develops a 10³-10⁴× tier ratio while the untrained
twin does not, size is ruled out as the explanation.

Five widths, 5 seeds each, both trained (20k SGD steps) and untrained
(init-only). Same 5-layer 256-neuron ReLU self-prediction architecture
used in V1.0. Runs fully on CPU in a few minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments/v2_1_qec"))
from analyze import fim_diagonal, tier_partition  # noqa: E402


def make_net(width: int, dim: int = 16) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(dim, width), nn.ReLU()]
    for _ in range(3):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers.append(nn.Linear(width, dim))
    return nn.Sequential(*layers)


def train(net: nn.Module, dim: int, steps: int, batch: int) -> None:
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for _ in range(steps):
        x = torch.randn(batch, dim)
        y = net(x)
        loss = 0.5 * (y - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def measure(net: nn.Module, dim: int, n_probes: int) -> dict:
    def sample_loss():
        x = torch.randn(1, dim)
        y = net(x)
        return 0.5 * (y - x).pow(2).mean()

    fim = fim_diagonal(net, sample_loss, n_samples=n_probes)
    tiers = tier_partition(fim)
    return {
        "n_params": sum(p.numel() for p in net.parameters()),
        "tier1_mean": tiers["tier1"]["mean"],
        "tier3_mean": tiers["tier3"]["mean"],
        "tier_ratio": tiers["ratio_tier1_tier3"],
    }


def run_one(width: int, seed: int, steps: int, probes: int, dim: int) -> tuple[dict, dict]:
    # Untrained (same init as trained — so we compare ONLY the effect of training).
    torch.manual_seed(seed)
    untrained_net = make_net(width, dim)
    untrained = measure(untrained_net, dim, probes)
    # Trained — independent build so FIM diagonal state is clean.
    torch.manual_seed(seed)
    trained_net = make_net(width, dim)
    train(trained_net, dim, steps, batch=128)
    trained = measure(trained_net, dim, probes)
    return untrained, trained


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--probes", type=int, default=200)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--omp-threads", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default=str(HERE / "v4_0_trained_vs_untrained.json"))
    args = ap.parse_args()

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.omp_threads)

    rows: list[dict] = []
    for w in args.widths:
        untrained_ratios = []
        trained_ratios = []
        n_params = None
        for s in args.seeds:
            t0 = time.time()
            u, t = run_one(w, s, args.steps, args.probes, args.dim)
            dt = time.time() - t0
            n_params = u["n_params"]
            untrained_ratios.append(u["tier_ratio"])
            trained_ratios.append(t["tier_ratio"])
            print(
                f"  width={w}  seed={s}  n_params={n_params:>8,}  "
                f"untrained_t1/t3={u['tier_ratio']:>8,.1f}  trained_t1/t3={t['tier_ratio']:>10,.1f}  ({dt:.1f}s)",
                flush=True,
            )
        u_arr = np.array(untrained_ratios)
        t_arr = np.array(trained_ratios)
        rows.append({
            "width": w,
            "n_params": n_params,
            "untrained": {
                "per_seed": untrained_ratios,
                "mean": float(u_arr.mean()),
                "std": float(u_arr.std(ddof=1)) if len(u_arr) > 1 else 0.0,
                "cv": float(u_arr.std(ddof=1) / u_arr.mean()) if len(u_arr) > 1 and u_arr.mean() > 0 else 0.0,
            },
            "trained": {
                "per_seed": trained_ratios,
                "mean": float(t_arr.mean()),
                "std": float(t_arr.std(ddof=1)) if len(t_arr) > 1 else 0.0,
                "cv": float(t_arr.std(ddof=1) / t_arr.mean()) if len(t_arr) > 1 and t_arr.mean() > 0 else 0.0,
            },
            "trained_over_untrained": float(t_arr.mean() / u_arr.mean()) if u_arr.mean() > 0 else float("inf"),
        })

    payload = {
        "config": vars(args),
        "results": rows,
        "interpretation": (
            "If trained/untrained ratio >> 1 at every width, the FIM tier "
            "hierarchy is learning-induced, not size-induced. Falsification: "
            "if trained/untrained < 5 at some width, size explains most of it."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
