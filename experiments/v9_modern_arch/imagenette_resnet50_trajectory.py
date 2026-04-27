"""V9.5b — Imagenette ResNet-50 from-scratch trajectory study.

Reviewer's request beyond V9.2c CIFAR-10 trajectory: medium-scale
real-image dataset, real ImageNet-class images. Imagenette is fastai's
10-class subset of ImageNet (13K images, 320x320 native), a standard
medium-scale natural-image benchmark.

ResNet-50 from random init, 10 epochs SGD, FIM measured at epochs
0/1/3/5/10. Demonstrates the trajectory at production-scale architecture
(25.6 M params) on real natural images, not just toy CIFAR.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.datasets import Imagenette
from torchvision.models import resnet50


def gini(values):
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v >= 0]
    if v.size == 0 or v.sum() == 0: return 0.0
    v.sort(); n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def effective_rank(values):
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v > 0]
    if v.size == 0: return 0.0
    return float((v.sum() ** 2) / (v.size * (v ** 2).sum()))


def top_1pct_mass(values):
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v > 0]
    if v.size == 0: return 0.0
    s = np.sort(v)[::-1]
    k1 = max(1, int(s.size * 0.01))
    return float(s[:k1].sum() / s.sum())


def tier_ratio(values):
    s = np.sort(values)[::-1]
    n = len(s)
    k1 = max(1, int(n * 0.01))
    k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean()); t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return (t1 / t3 if t3 > 0 else float("inf"))


def get_imagenette(batch=128, root="/data/datasets/imagenette"):
    Path(root).mkdir(parents=True, exist_ok=True)
    transform_train = T.Compose([
        T.RandomResizedCrop(160),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = T.Compose([
        T.Resize(192),
        T.CenterCrop(160),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    trainset = Imagenette(root=root, split="train", size="160px", download=True, transform=transform_train)
    testset = Imagenette(root=root, split="val", size="160px", transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_epoch(net, loader, opt, device):
    net.train()
    correct = 0; total = 0; loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        out = net(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (net(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


def fim_diagonal(net, loader, device, n_probes):
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    iter_loader = iter(loader)
    seen = 0
    while seen < n_probes:
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            x, y = next(iter_loader)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        for i in range(min(x.size(0), n_probes - seen)):
            xi, yi = x[i:i+1], y[i:i+1]
            out = net(xi)
            loss = F.cross_entropy(out, yi)
            net.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in net.named_parameters():
                if p.grad is not None:
                    fim[n] += p.grad.data.double() ** 2
            seen += 1
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def measure(net, test_loader, device, n_probes):
    fim = fim_diagonal(net, test_loader, device, n_probes)
    ratio = tier_ratio(fim)
    return {
        "T1T3": ratio,
        "log10_T1T3": float(np.log10(max(ratio, 1e-30))),
        "gini": gini(fim),
        "eff_rank_n": effective_rank(fim) / fim.size,
        "top_1pct_mass": top_1pct_mass(fim),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--checkpoint-epochs", type=int, nargs="+", default=[0, 1, 3, 5, 10])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_5b_imagenette_resnet50_trajectory_results.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_loader, test_loader = get_imagenette(batch=args.batch)
    # 10-class Imagenette: replace fc head
    net = resnet50(num_classes=10).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ResNet-50-Imagenette params: {n_params:,}", flush=True)

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    trajectory = []

    if 0 in args.checkpoint_epochs:
        print(f"\n[epoch 0 / pre-training] random-init FIM measurement...", flush=True)
        t0 = time.time()
        m = measure(net, test_loader, device, args.n_probes)
        m["epoch"] = 0; m["test_accuracy"] = float(evaluate(net, test_loader, device))
        m["fim_time_s"] = time.time() - t0
        trajectory.append(m)
        print(f"  T1/T3={m['T1T3']:.3e} Gini={m['gini']:.4f} top1%={m['top_1pct_mass']:.4f} acc={m['test_accuracy']:.3f} ({m['fim_time_s']:.1f}s)", flush=True)

    for ep in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(net, train_loader, opt, device)
        sched.step()
        dt = time.time() - t0
        ep_idx = ep + 1
        if ep_idx in args.checkpoint_epochs:
            print(f"\n[epoch {ep_idx} / {args.epochs}] train_acc={train_acc:.3f} ({dt:.1f}s); FIM...", flush=True)
            t1 = time.time()
            m = measure(net, test_loader, device, args.n_probes)
            m["epoch"] = ep_idx
            m["test_accuracy"] = float(evaluate(net, test_loader, device))
            m["train_accuracy"] = float(train_acc)
            m["train_loss"] = float(train_loss)
            m["fim_time_s"] = time.time() - t1
            m["epoch_time_s"] = dt
            trajectory.append(m)
            print(f"  T1/T3={m['T1T3']:.3e} Gini={m['gini']:.4f} top1%={m['top_1pct_mass']:.4f} test_acc={m['test_accuracy']:.3f} ({m['fim_time_s']:.1f}s)", flush=True)

    log10 = [(m["epoch"], m["log10_T1T3"]) for m in trajectory]
    drop = log10[0][1] - log10[-1][1] if len(log10) >= 2 else 0
    payload = {
        "config": vars(args), "n_params": int(n_params), "trajectory": trajectory,
        "log10_T1T3_init_vs_final": {
            "epoch_0": log10[0][1] if log10 else None,
            "epoch_final": log10[-1][1] if log10 else None,
            "log10_drop_during_training": drop,
            "T1T3_drop_factor": 10 ** drop if log10 else None,
        },
        "interpretation": (
            "Imagenette ResNet-50 (25.6M params, 10-class real-image subset of ImageNet) "
            "from-scratch trajectory. 10 epochs SGD, FIM measured at epochs 0/1/3/5/10. "
            "Real natural-image classification, production-scale architecture, from random "
            "init — extends V9.2c CIFAR-10 ResNet-18 trajectory to real-ImageNet-class images "
            "and a deeper architecture."
        ),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n=== Imagenette ResNet-50 trajectory ===")
    for m in trajectory:
        print(f"  epoch={m['epoch']:>2} log10(T1/T3)={m['log10_T1T3']:.3f} Gini={m['gini']:.3f} test_acc={m['test_accuracy']:.3f}")
    print(f"\nT1/T3 reduction: {drop:.3f} log units (factor {10**drop:.1f}×)")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
