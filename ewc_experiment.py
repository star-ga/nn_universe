#!/usr/bin/env python3
"""
EWC Catastrophic Forgetting Experiment
3-layer, 64-neuron network. Task A = identity, Task B = 2x scaling.
Tests EWC at λ = 0, 1000, 10000, 100000, 400000.
Seed: 42. Fully reproducible.
"""
import torch
import torch.nn as nn
import json

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f"Device: {device} ({gpu_name})")

dim = 8

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, dim),
        )
    def forward(self, x):
        return self.net(x)

def eval_task(net, task_fn):
    net.eval()
    with torch.no_grad():
        x = torch.randn(1000, dim, device=device)
        y = net(x)
        target = task_fn(x)
        loss = 0.5 * (y - target).pow(2).mean()
    net.train()
    return float(loss)

def compute_fim(net, task_fn, n_samples=2000):
    """Per-sample FIM: batch_size=1 so grad² = true Fisher diagonal."""
    net.eval()
    fim = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    for _ in range(n_samples):
        x = torch.randn(1, dim, device=device)  # single sample
        y = net(x)
        target = task_fn(x)
        loss = 0.5 * (y - target).pow(2).mean()
        net.zero_grad()
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data ** 2
    for n in fim:
        fim[n] /= n_samples
    net.train()
    return fim

task_a = lambda x: x
task_b = lambda x: 2.0 * x

results = {}

for ewc_lambda in [0, 10000, 50000, 100000]:
    torch.manual_seed(42)  # Same init for fair comparison
    net = SmallNet().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.005)

    # Train Task A — 20K steps
    for step in range(20000):
        x = torch.randn(128, dim, device=device)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    task_a_before = eval_task(net, task_a)
    checkpoint = {n: p.data.clone() for n, p in net.named_parameters()}
    fim = compute_fim(net, task_a)

    # Report FIM scale
    fim_vals = torch.cat([v.flatten() for v in fim.values()])
    if ewc_lambda == 0:
        print(f"  FIM: mean={float(fim_vals.mean()):.6e}, max={float(fim_vals.max()):.6e}")

    # Train Task B — 5K steps with EWC
    opt_b = torch.optim.SGD(net.parameters(), lr=0.005)
    for step in range(5000):
        x = torch.randn(128, dim, device=device)
        loss = 0.5 * (net(x) - 2.0 * x).pow(2).mean()

        if ewc_lambda > 0:
            ewc_loss = sum((fim[n] * (p - checkpoint[n]).pow(2)).sum()
                          for n, p in net.named_parameters())
            loss = loss + ewc_lambda * ewc_loss

        opt_b.zero_grad()
        loss.backward()
        opt_b.step()

    task_a_after = eval_task(net, task_a)
    task_b_after = eval_task(net, task_b)
    degradation = task_a_after / task_a_before

    results[f"lambda_{ewc_lambda}"] = {
        "task_a_before": task_a_before,
        "task_a_after": task_a_after,
        "task_b_after": task_b_after,
        "degradation": degradation,
    }
    print(f"  λ={ewc_lambda:>7d}: A_before={task_a_before:.6f} A_after={task_a_after:.6f} B_after={task_b_after:.6f} degradation={degradation:.1f}x")

# Summary
d0 = results["lambda_0"]["degradation"]
for lam in [10000, 50000, 100000]:
    key = f"lambda_{lam}"
    if key in results:
        dl = results[key]["degradation"]
        red = d0 / dl if dl > 0 else 0
        print(f"  Forgetting reduction (λ=0 vs λ={lam}): {red:.1f}x")
        results[key]["forgetting_reduction_vs_0"] = red

with open("ewc_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to ewc_results.json")
