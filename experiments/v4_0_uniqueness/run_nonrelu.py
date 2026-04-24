"""V4.1.1 — does the init-induced FIM hierarchy persist for non-ReLU activations?

V4.1 established that untrained 5-layer 256-ReLU MLPs have FIM T1/T3 in the
10^3–10^4 range and training reduces this by 4-24×. This extension asks:
is the effect ReLU-specific, or does it hold for GELU and tanh too?

For each activation, run 5 widths × 5 seeds in a matched trained-vs-untrained
comparison. Output JSON at `v4_0_nonrelu_results.json`.
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


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


def make_net(width: int, activation: str, dim: int = 16) -> nn.Module:
    act_cls = _ACTIVATIONS[activation]
    layers: list[nn.Module] = [nn.Linear(dim, width), act_cls()]
    for _ in range(3):
        layers += [nn.Linear(width, width), act_cls()]
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


def measure(net: nn.Module, dim: int, probes: int) -> dict:
    def sample_loss():
        x = torch.randn(1, dim)
        return 0.5 * (net(x) - x).pow(2).mean()
    fim = fim_diagonal(net, sample_loss, n_samples=probes)
    tiers = tier_partition(fim)
    return {
        "n_params": sum(p.numel() for p in net.parameters()),
        "tier_ratio": tiers["ratio_tier1_tier3"],
    }


def run_one(activation: str, width: int, seed: int, steps: int, probes: int, dim: int) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    untrained_net = make_net(width, activation, dim)
    untrained = measure(untrained_net, dim, probes)
    torch.manual_seed(seed)
    trained_net = make_net(width, activation, dim)
    train(trained_net, dim, steps, batch=128)
    trained = measure(trained_net, dim, probes)
    return untrained, trained


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", nargs="+", default=["relu", "gelu", "tanh"])
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--probes", type=int, default=200)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--omp-threads", type=int, default=8)
    ap.add_argument("--out", type=str,
                    default=str(HERE / "v4_0_nonrelu_results.json"))
    args = ap.parse_args()

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(args.omp_threads)

    all_results: list[dict] = []
    for act in args.activations:
        print(f"=== activation: {act} ===", flush=True)
        for w in args.widths:
            u_list = []; t_list = []
            for s in args.seeds:
                t0 = time.time()
                u, t = run_one(act, w, s, args.steps, args.probes, args.dim)
                dt = time.time() - t0
                u_list.append(u["tier_ratio"]); t_list.append(t["tier_ratio"])
                print(
                    f"  {act} W={w} seed={s}  N={u['n_params']:>8,}  "
                    f"untrained={u['tier_ratio']:>10,.1f}  trained={t['tier_ratio']:>10,.1f}  "
                    f"({dt:.1f}s)",
                    flush=True,
                )
            ua = np.array(u_list); ta = np.array(t_list)
            all_results.append({
                "activation": act,
                "width": w,
                "n_params": u["n_params"],
                "untrained_mean": float(ua.mean()),
                "untrained_std": float(ua.std(ddof=1)) if len(ua) > 1 else 0.0,
                "trained_mean": float(ta.mean()),
                "trained_std": float(ta.std(ddof=1)) if len(ta) > 1 else 0.0,
                "training_reduction": float(ua.mean() / ta.mean()) if ta.mean() > 0 else float("inf"),
            })
    payload = {"config": vars(args), "results": all_results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
