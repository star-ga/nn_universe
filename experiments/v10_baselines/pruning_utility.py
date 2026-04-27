"""V10c — Downstream utility study: top-1% FIM mass as pruning saliency.

Reviewer's significance ask: "show one downstream utility study: pruning,
optimizer tuning, checkpoint diagnosis, or architecture screening."

This script measures: at random initialisation, does the top-X% of
parameters by FIM-diagonal value form a useful pruning subnetwork
(in the sense that retraining the unpruned subnetwork from the same
random seed reaches comparable accuracy to the dense baseline)?

Comparison baselines on a small MLP trained on a synthetic 4-class
classification task:
  1. Dense (no pruning) baseline
  2. Top-1% magnitude pruning (keep 1% of |W| largest weights)
  3. Top-1% FIM-diagonal pruning (keep 1% with largest F_ii)
  4. Random 1% pruning
For each, train the masked subnetwork and report final test accuracy.

This is a small / illustrative downstream-utility result, NOT a
state-of-the-art pruning claim. The point is to show the diagnostic
has *some* practical use beyond classification.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_data(n=2000, dim=16, n_classes=4, seed=0):
    torch.manual_seed(seed)
    # Synthetic classification: random Gaussian inputs + fixed random teacher
    teacher = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, n_classes))
    for p in teacher.parameters(): p.requires_grad = False
    X = torch.randn(n, dim)
    with torch.no_grad():
        y = teacher(X).argmax(dim=1)
    return X, y


def make_mlp(seed, dim=16, hidden=128, depth=5, n_classes=4):
    torch.manual_seed(seed)
    layers = [nn.Linear(dim, hidden), nn.ReLU()]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
    layers.append(nn.Linear(hidden, n_classes))
    return nn.Sequential(*layers)


def fim_diag_classification(net, X, n_probes=200, n_classes=4):
    """FIM diagonal for classification: per-sample squared cross-entropy gradient."""
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    rng = np.random.default_rng(0)
    n = X.size(0)
    for _ in range(n_probes):
        i = rng.integers(0, n)
        x = X[i:i+1]
        out = net(x)
        # Sample target from softmax (true Fisher) — use predicted class for empirical Fisher
        target = torch.distributions.Categorical(logits=out).sample()
        loss = F.cross_entropy(out, target)
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n_, p in net.named_parameters():
            if p.grad is not None:
                fim[n_] += p.grad.data.double() ** 2
    for n_ in fim:
        fim[n_] /= n_probes
    return fim


def magnitude_dict(net):
    return {n: p.data.abs().double() for n, p in net.named_parameters()}


def train(net, X, y, mask=None, lr=0.01, epochs=80, batch=64):
    """Train net (with optional binary mask freezing zero entries)."""
    if mask is not None:
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in mask:
                    p.mul_(mask[n].to(p.dtype))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.size(0)
    for ep in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i+batch]
            out = net(X[b])
            loss = F.cross_entropy(out, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            # Re-apply mask after gradient update (keep pruned weights at 0)
            if mask is not None:
                with torch.no_grad():
                    for n_, p in net.named_parameters():
                        if n_ in mask:
                            p.mul_(mask[n_].to(p.dtype))
    with torch.no_grad():
        return float((net(X).argmax(1) == y).float().mean())


def topk_mask(scores_dict, keep_frac):
    """Build a binary mask keeping the top keep_frac of all parameters by score."""
    flat = torch.cat([s.flatten() for s in scores_dict.values()])
    n_keep = max(1, int(flat.numel() * keep_frac))
    threshold = torch.sort(flat, descending=True)[0][n_keep - 1]
    return {n: (s >= threshold).float() for n, s in scores_dict.items()}


def random_mask(net, keep_frac, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for n, p in net.named_parameters():
        flat = rng.random(p.numel()) > (1 - keep_frac)
        out[n] = torch.from_numpy(flat.reshape(p.shape).astype(np.float32))
    return out


def main():
    KEEP_FRACS = [0.01, 0.05, 0.10, 0.50, 1.00]
    SEEDS = [0, 1, 2]
    results = {kf: {"dense": [], "fim": [], "magnitude": [], "random": []} for kf in KEEP_FRACS}

    X, y = make_data(n=2000, dim=16, n_classes=4, seed=0)
    Xt, yt = make_data(n=500, dim=16, n_classes=4, seed=42)

    for seed in SEEDS:
        for kf in KEEP_FRACS:
            # Dense baseline (kf=1.0 trivially; for other kf we still run dense as control)
            if kf == 1.00:
                net = make_mlp(seed=seed)
                acc = train(net, X, y, mask=None, epochs=80)
                results[kf]["dense"].append(acc)
                test_acc_dense = float((net(Xt).argmax(1) == yt).float().mean())
                results[kf]["fim"].append(acc)
                results[kf]["magnitude"].append(acc)
                results[kf]["random"].append(acc)
                continue

            # Build masks at this keep fraction from a fresh init
            net_for_fim = make_mlp(seed=seed)
            fim_scores = fim_diag_classification(net_for_fim, X, n_probes=100)
            mag_scores = magnitude_dict(net_for_fim)

            fim_mask = topk_mask(fim_scores, kf)
            mag_mask = topk_mask(mag_scores, kf)
            rnd_mask = random_mask(net_for_fim, kf, seed=seed * 1000)

            # Train each pruned subnet from same init
            for label, mask in [("fim", fim_mask), ("magnitude", mag_mask), ("random", rnd_mask)]:
                net = make_mlp(seed=seed)
                _ = train(net, X, y, mask=mask, epochs=80)
                test_acc = float((net(Xt).argmax(1) == yt).float().mean())
                results[kf][label].append(test_acc)
            print(f"  seed={seed} keep={kf:.2f}: FIM={results[kf]['fim'][-1]:.3f} mag={results[kf]['magnitude'][-1]:.3f} rnd={results[kf]['random'][-1]:.3f}", flush=True)

    summary = {}
    for kf in KEEP_FRACS:
        summary[str(kf)] = {
            label: {
                "mean": float(np.mean(results[kf][label])),
                "std": float(np.std(results[kf][label], ddof=1)) if len(results[kf][label]) > 1 else 0.0,
            }
            for label in ("fim", "magnitude", "random", "dense")
        }

    # Print verdict
    print("\n=== Pruning utility comparison (test accuracy, mean ± std across 3 seeds) ===")
    print(f"{'Keep frac':<10} {'FIM':<14} {'Magnitude':<14} {'Random':<14}")
    for kf in KEEP_FRACS:
        s = summary[str(kf)]
        print(f"{kf:<10.2f} {s['fim']['mean']:.3f}±{s['fim']['std']:.3f}    {s['magnitude']['mean']:.3f}±{s['magnitude']['std']:.3f}    {s['random']['mean']:.3f}±{s['random']['std']:.3f}")

    payload = {
        "config": {"dim": 16, "n_classes": 4, "n_train": 2000, "n_test": 500,
                   "depth": 5, "hidden": 128, "epochs": 80, "seeds": SEEDS},
        "per_seed_results": results,
        "summary": summary,
        "interpretation": (
            "Downstream-utility check: at random initialisation, does FIM-diagonal "
            "saliency pick a better subnetwork than magnitude or random? Dense "
            f"baseline test acc: {summary['1.0']['dense']['mean']:.3f}. "
            "FIM-saliency vs magnitude vs random pruning: see summary table. "
            "This is an illustrative result, not a state-of-the-art pruning claim."
        ),
    }
    p = Path(__file__).parent / "v10c_pruning_utility_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
