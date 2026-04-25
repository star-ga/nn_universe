"""V9.2 — CIFAR-10 ResNet-18 FIM measurement.

NeurIPS reviewers flagged the synthetic-task / toy-architecture coverage.
This experiment is the smallest credible real-data + real-architecture
pairing that fits a single RTX 3080 10 GB:

  - Architecture: ResNet-18 (PyTorch torchvision-style, 11.2M params).
  - Dataset: CIFAR-10 (50k train, 10k test, 32x32x3, 10 classes).
  - Training: 10 epochs SGD momentum=0.9 lr=0.1 → 0.001 cosine.
  - FIM measurement: per-sample squared gradient on the test set, 200
    probes, float64 accumulation. Match the paper protocol.

We report:
  - Final test accuracy (sanity check that training worked).
  - T1/T3 tier ratio.
  - Partition-invariant Gini, effective rank, top-1% mass.

Goal: confirm that the FIM tier hierarchy is present on a real-data
benchmark with a real depth-18 architecture, addressing the 'synthetic
tasks only' reviewer concern.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gini(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v >= 0]
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v.sort()
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def effective_rank(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v > 0]
    n = v.size
    if n == 0:
        return 0.0
    return float((v.sum() ** 2) / (n * (v ** 2).sum()))


def top_1pct_mass(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    s = np.sort(v)[::-1]
    k = max(1, int(0.01 * v.size))
    return float(s[:k].sum() / s.sum())


def tier_ratio(values: np.ndarray):
    s = np.sort(values)[::-1]; n = len(s)
    k1 = max(1, int(n * 0.01)); k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean()); t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1/t3 if t3 > 0 else float("inf"))


# ============================================================
# Compact ResNet-18 (CIFAR variant — 3x3 first conv, no maxpool)
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)
    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = F.avg_pool2d(out, 4).flatten(1)
        return self.linear(out)


# ============================================================
# Training + FIM
# ============================================================

def get_cifar(batch=128, root="/tmp/cifar10"):
    import torchvision
    import torchvision.transforms as T
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_epoch(net, loader, opt, device):
    net.train()
    correct = 0; total = 0; loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
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
        x, y = x.to(device), y.to(device)
        out = net(x)
        correct += (out.argmax(1) == y).sum().item()
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
        x, y = x.to(device), y.to(device)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_2_cifar_resnet18_fim.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_loader, test_loader = get_cifar(batch=args.batch)
    net = ResNet18Cifar(num_classes=10).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ResNet-18-CIFAR params: {n_params:,}", flush=True)

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print("Training...", flush=True)
    for ep in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(net, train_loader, opt, device)
        test_acc = evaluate(net, test_loader, device)
        sched.step()
        dt = time.time() - t0
        print(f"  epoch {ep+1:>2}/{args.epochs}  train_loss={train_loss:.3f}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  ({dt:.1f}s)",
              flush=True)

    print("\nMeasuring FIM diagonal on test set...", flush=True)
    t0 = time.time()
    fim = fim_diagonal(net, test_loader, device, args.n_probes)
    t_fim = time.time() - t0
    t1m, t3m, ratio = tier_ratio(fim)
    g = gini(fim); rn = effective_rank(fim) / fim.size; tp1 = top_1pct_mass(fim)

    print(f"\n=== ResNet-18 + CIFAR-10 FIM ===")
    print(f"  N params       = {n_params:,}")
    print(f"  T1/T3          = {ratio:.3e}")
    print(f"  Gini           = {g:.4f}")
    print(f"  r_eff/n        = {rn:.5f}")
    print(f"  top-1% mass    = {tp1:.4f}")
    print(f"  test accuracy  = {test_acc:.4f}")
    print(f"  FIM time       = {t_fim:.1f}s")

    payload = {
        "config": vars(args),
        "n_params": int(n_params),
        "test_accuracy": float(test_acc),
        "tier1_mean": t1m, "tier3_mean": t3m, "tier_ratio": ratio,
        "partition_invariant": {
            "gini": g,
            "effective_rank_normalised": rn,
            "top_1pct_mass": tp1,
        },
        "fim_measurement_s": t_fim,
        "interpretation": (
            "FIM tier hierarchy on a real-data benchmark (CIFAR-10) with a "
            "real architecture (ResNet-18, 11M params). All four observables "
            "(T1/T3, Gini, r_eff/n, top-1% mass) are reported; the deep-"
            "sequential prediction is high T1/T3 + Gini→1 + r_eff/n→0 + "
            "top-1% mass → significant fraction."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
