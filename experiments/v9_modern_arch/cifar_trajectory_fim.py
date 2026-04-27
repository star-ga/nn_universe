"""V9.2c — CIFAR-10 ResNet-18 FIM trajectory study.

Reviewer's request: "from-scratch trajectory evidence for at least one
medium-scale model, not just pretrained checkpoints and endpoint
evaluations."

This script trains a CIFAR-10 ResNet-18 from random init and measures
T1/T3, Gini, top-1% mass at epochs 0, 1, 3, 10 of a 10-epoch training
run, on the SAME network (no re-init). This shows the hierarchy
trajectory across training, complementing V4.1 (init vs endpoint) with
intermediate samples.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.v9_modern_arch.cifar_resnet18_fim import (
    ResNet18Cifar, train_epoch, evaluate, fim_diagonal, tier_ratio,
    gini, effective_rank, top_1pct_mass, get_cifar,
)


def measure(net, test_loader, device, n_probes=200):
    fim = fim_diagonal(net, test_loader, device, n_probes)
    t1m, t3m, ratio = tier_ratio(fim)
    g = gini(fim); rn = effective_rank(fim) / fim.size; tp1 = top_1pct_mass(fim)
    return {"T1T3": ratio, "gini": g, "eff_rank_n": rn, "top_1pct_mass": tp1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--checkpoint-epochs", type=int, nargs="+", default=[0, 1, 3, 5, 10])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_2c_cifar_trajectory_results.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_loader, test_loader = get_cifar(batch=args.batch)
    net = ResNet18Cifar(num_classes=10).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ResNet-18-CIFAR-10 params: {n_params:,}", flush=True)

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    trajectory = []

    # Epoch 0: at init, before any training
    if 0 in args.checkpoint_epochs:
        print(f"\n[epoch 0 / pre-training] measuring FIM at random init...", flush=True)
        t0 = time.time()
        m = measure(net, test_loader, device, args.n_probes)
        m["epoch"] = 0; m["test_accuracy"] = float(evaluate(net, test_loader, device))
        m["fim_time_s"] = time.time() - t0
        trajectory.append(m)
        print(f"  T1/T3={m['T1T3']:.3e} Gini={m['gini']:.4f} top1%={m['top_1pct_mass']:.4f} acc={m['test_accuracy']:.3f}", flush=True)

    # Train + measure at checkpoints
    for ep in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(net, train_loader, opt, device)
        sched.step()
        dt = time.time() - t0
        ep_idx = ep + 1
        if ep_idx in args.checkpoint_epochs:
            print(f"\n[epoch {ep_idx} / {args.epochs}] train_acc={train_acc:.3f} train_loss={train_loss:.3f} ({dt:.1f}s); measuring FIM...", flush=True)
            t1 = time.time()
            m = measure(net, test_loader, device, args.n_probes)
            m["epoch"] = ep_idx
            m["test_accuracy"] = float(evaluate(net, test_loader, device))
            m["train_accuracy"] = float(train_acc)
            m["train_loss"] = float(train_loss)
            m["fim_time_s"] = time.time() - t1
            m["epoch_time_s"] = dt
            trajectory.append(m)
            print(f"  T1/T3={m['T1T3']:.3e} Gini={m['gini']:.4f} top1%={m['top_1pct_mass']:.4f} train_acc={m['train_accuracy']:.3f} test_acc={m['test_accuracy']:.3f}", flush=True)

    # Summary: trajectory of log10(T1/T3)
    log10_T1T3 = [(m["epoch"], np.log10(max(m["T1T3"], 1e-30))) for m in trajectory]
    log10_T1T3_drop = log10_T1T3[0][1] - log10_T1T3[-1][1] if len(log10_T1T3) >= 2 else 0
    payload = {
        "config": vars(args),
        "n_params": int(n_params),
        "trajectory": trajectory,
        "log10_T1T3_init_vs_final": {
            "epoch_0": log10_T1T3[0][1] if log10_T1T3 else None,
            "epoch_final": log10_T1T3[-1][1] if log10_T1T3 else None,
            "log10_drop_during_training": log10_T1T3_drop,
            "T1T3_drop_factor": 10 ** log10_T1T3_drop if log10_T1T3 else None,
        },
        "interpretation": (
            "FIM tier hierarchy trajectory across CIFAR-10 ResNet-18 from-scratch training "
            "(10 epochs cosine SGD). Measures T1/T3 at random init AND at multiple training "
            "checkpoints to characterise the V4.1 'training reduces but does not eliminate "
            "the hierarchy' finding with intermediate samples (not just init vs endpoint)."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n=== Trajectory summary ===")
    for m in trajectory:
        print(f"  epoch={m['epoch']:>2} log10(T1/T3)={np.log10(max(m['T1T3'],1e-30)):.3f} Gini={m['gini']:.3f} test_acc={m['test_accuracy']:.3f}")
    print(f"\nT1/T3 reduction during training: {log10_T1T3_drop:.3f} log units (factor {10**log10_T1T3_drop:.1f}×)")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
