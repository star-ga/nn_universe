#!/usr/bin/env python3
"""
Toy Neural-Network Cosmology Experiment
Runs on RTX 3080. Tests 3 predictions from the FIM-Onsager framework:
1. Spontaneous symmetry breaking under self-supervised training
2. FIM eigenvalue hierarchy (3-tier structure)
3. Catastrophic forgetting resistance via EWC regularization

Architecture: 5-layer FC, 256 neurons/layer, ReLU, ~340K params
Training: SGD with FIM-diagonal approximation, 50K iterations
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    gpu_name = 'Apple Silicon (MPS)'
else:
    device = torch.device('cpu')
    gpu_name = 'CPU'
print(f"Device: {device} ({gpu_name})")

# --- Architecture ---
class SelfPredictNet(nn.Module):
    def __init__(self, dim=64, hidden=256, layers=5):
        super().__init__()
        mods = [nn.Linear(dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.ReLU()]
        mods.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

dim = 64
net = SelfPredictNet(dim=dim, hidden=256, layers=5).to(device)
n_params = sum(p.numel() for p in net.parameters())
print(f"Parameters: {n_params:,}")

# --- Phase 1: Self-supervised training (self-prediction fixed point) ---
print("\n=== Phase 1: Self-Supervised Training ===")
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
losses = []
t0 = time.time()

for step in range(50000):
    x = torch.randn(128, dim, device=device)
    y = net(x)
    loss = 0.5 * (y - x).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5000 == 0:
        losses.append(float(loss))
        print(f"  Step {step:5d}: loss={float(loss):.6f}")

print(f"Training time: {time.time()-t0:.1f}s")

# --- Measurement 1: Symmetry Breaking ---
print("\n=== Measurement 1: Symmetry Breaking ===")
sv_stats = []
for name, param in net.named_parameters():
    if 'weight' in name and param.dim() == 2:
        U, S, V = torch.linalg.svd(param.data)
        top3 = S[:3].tolist()
        ratio = float(S[0] / S[-1]) if S[-1] > 1e-8 else float('inf')
        sv_stats.append({
            'layer': name, 'shape': list(param.shape),
            'top3_sv': [round(s, 4) for s in top3],
            'sv_ratio': round(ratio, 2),
            'sv_std': round(float(S.std()), 4)
        })
        print(f"  {name}: top3={[round(s,3) for s in top3]}, ratio={ratio:.1f}, std={float(S.std()):.4f}")

# --- Measurement 2: FIM Diagonal ---
print("\n=== Measurement 2: FIM Diagonal Approximation ===")
# Empirical FIM diagonal: F_ii = E[(d log p / d theta_i)^2].
# Accumulate in float64: grad^2 for tier-3 params can fall below the float32
# minimum normal (~1.2e-38) and round to zero, which inflates the reported
# T1/T3 ratio. See experiments/v2_1_qec/analyze.py for the canonical version.
net.eval()
fim_diag = {
    name: torch.zeros_like(p, dtype=torch.float64) for name, p in net.named_parameters()
}

n_samples = 1000
for _ in range(n_samples):
    x = torch.randn(32, dim, device=device)
    y = net(x)
    loss = 0.5 * (y - x).pow(2).mean()
    net.zero_grad()
    loss.backward()
    for name, p in net.named_parameters():
        if p.grad is not None:
            fim_diag[name] += p.grad.data.double() ** 2

# Normalize
for name in fim_diag:
    fim_diag[name] /= n_samples

# Collect all FIM diagonal values
all_fim = torch.cat([v.flatten() for v in fim_diag.values()]).cpu().numpy()
all_fim_sorted = np.sort(all_fim)[::-1]  # descending

# Compute tier structure
total = len(all_fim_sorted)
tier1_threshold = np.percentile(all_fim_sorted, 99)  # top 1%
tier2_threshold = np.percentile(all_fim_sorted, 50)  # top 50%

n_tier1 = int(np.sum(all_fim_sorted >= tier1_threshold))
n_tier2 = int(np.sum((all_fim_sorted >= tier2_threshold) & (all_fim_sorted < tier1_threshold)))
n_tier3 = total - n_tier1 - n_tier2

tier1_mean = float(np.mean(all_fim_sorted[:n_tier1])) if n_tier1 > 0 else 0
tier2_mean = float(np.mean(all_fim_sorted[n_tier1:n_tier1+n_tier2])) if n_tier2 > 0 else 0
tier3_mean = float(np.mean(all_fim_sorted[n_tier1+n_tier2:])) if n_tier3 > 0 else 0

print(f"  Total parameters: {total:,}")
print(f"  Tier 1 (top 1%, 'physical constants'): {n_tier1} params, mean FIM={tier1_mean:.6f}")
print(f"  Tier 2 (1-50%, 'coupling constants'): {n_tier2} params, mean FIM={tier2_mean:.6f}")
print(f"  Tier 3 (bottom 50%, 'gauge DOF'): {n_tier3} params, mean FIM={tier3_mean:.8f}")
print(f"  Tier1/Tier3 ratio: {tier1_mean/tier3_mean:.1f}x" if tier3_mean > 0 else "  Tier3 is zero")
print(f"  Top 10 FIM values: {[round(float(v), 6) for v in all_fim_sorted[:10]]}")

# --- Measurement 3: Catastrophic Forgetting ---
print("\n=== Measurement 3: Catastrophic Forgetting (EWC vs No-EWC) ===")

# Save checkpoint for EWC
checkpoint = {name: p.data.clone() for name, p in net.named_parameters()}

def train_new_task(net, task_transform, steps=10000, ewc_lambda=0.0,
                   fim=None, checkpoint=None):
    """Train on a new self-consistency task."""
    opt = torch.optim.SGD(net.parameters(), lr=5e-4, momentum=0.9)
    for step in range(steps):
        x = torch.randn(128, dim, device=device)
        x_transformed = task_transform(x)
        y = net(x_transformed)
        loss = 0.5 * (y - x_transformed).pow(2).mean()

        # EWC penalty
        if ewc_lambda > 0 and fim is not None and checkpoint is not None:
            ewc_loss = 0
            for name, p in net.named_parameters():
                ewc_loss += (fim[name] * (p - checkpoint[name]).pow(2)).sum()
            loss = loss + ewc_lambda * ewc_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

def eval_task(net, task_transform=None):
    """Evaluate on original self-prediction task."""
    net.eval()
    with torch.no_grad():
        x = torch.randn(1000, dim, device=device)
        if task_transform:
            x = task_transform(x)
        y = net(x)
        loss = 0.5 * (y - x).pow(2).mean()
    net.train()
    return float(loss)

# Define 3 new tasks (rotated self-prediction)
tasks = [
    lambda x: x @ torch.eye(dim, device=device).roll(1, 0),  # permutation
    lambda x: x * torch.linspace(0.5, 1.5, dim, device=device),  # scaling
    lambda x: torch.relu(x),  # rectification
]

# Test WITHOUT EWC
print("  Without EWC (lambda=0):")
net_no_ewc = SelfPredictNet(dim=dim, hidden=256, layers=5).to(device)
net_no_ewc.load_state_dict(net.state_dict())
baseline_loss = eval_task(net_no_ewc)
print(f"    Baseline (original task): {baseline_loss:.6f}")

no_ewc_results = [baseline_loss]
for i, task in enumerate(tasks):
    train_new_task(net_no_ewc, task, steps=5000, ewc_lambda=0)
    orig_loss = eval_task(net_no_ewc)  # check original task
    no_ewc_results.append(orig_loss)
    print(f"    After task {i+1}: original_loss={orig_loss:.6f} (degradation={orig_loss/baseline_loss:.1f}x)")

# Test WITH EWC
print("  With EWC (lambda=100):")
net_ewc = SelfPredictNet(dim=dim, hidden=256, layers=5).to(device)
net_ewc.load_state_dict(net.state_dict())
baseline_loss_ewc = eval_task(net_ewc)
print(f"    Baseline (original task): {baseline_loss_ewc:.6f}")

ewc_results = [baseline_loss_ewc]
for i, task in enumerate(tasks):
    train_new_task(net_ewc, task, steps=5000, ewc_lambda=100.0,
                   fim=fim_diag, checkpoint=checkpoint)
    orig_loss = eval_task(net_ewc)
    ewc_results.append(orig_loss)
    print(f"    After task {i+1}: original_loss={orig_loss:.6f} (degradation={orig_loss/baseline_loss_ewc:.1f}x)")

# --- Save results ---
results = {
    'device': str(device),
    'n_params': n_params,
    'training_losses': losses,
    'sv_stats': sv_stats,
    'fim': {
        'total_params': total,
        'tier1': {'count': n_tier1, 'mean': tier1_mean},
        'tier2': {'count': n_tier2, 'mean': tier2_mean},
        'tier3': {'count': n_tier3, 'mean': tier3_mean},
        'ratio_tier1_tier3': tier1_mean / tier3_mean if tier3_mean > 0 else None,
        'top10': [float(v) for v in all_fim_sorted[:10]],
        'spectrum_percentiles': {
            'p99': float(np.percentile(all_fim_sorted, 99)),
            'p95': float(np.percentile(all_fim_sorted, 95)),
            'p50': float(np.percentile(all_fim_sorted, 50)),
            'p10': float(np.percentile(all_fim_sorted, 10)),
            'p1': float(np.percentile(all_fim_sorted, 1)),
        }
    },
    'forgetting': {
        'no_ewc': no_ewc_results,
        'ewc_lambda100': ewc_results,
        'no_ewc_final_degradation': no_ewc_results[-1] / no_ewc_results[0] if no_ewc_results[0] > 0 else None,
        'ewc_final_degradation': ewc_results[-1] / ewc_results[0] if ewc_results[0] > 0 else None,
    }
}

with open('toy_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== SUMMARY ===")
print(f"Symmetry breaking: SV ratio range {min(s['sv_ratio'] for s in sv_stats):.1f}x - {max(s['sv_ratio'] for s in sv_stats):.1f}x")
print(f"FIM hierarchy: Tier1/Tier3 = {tier1_mean/tier3_mean:.0f}x" if tier3_mean > 0 else "FIM hierarchy: extreme")
print(f"Forgetting (no EWC): {no_ewc_results[-1]/no_ewc_results[0]:.1f}x degradation")
print(f"Forgetting (EWC):    {ewc_results[-1]/ewc_results[0]:.1f}x degradation")
print(f"Results saved to toy_experiment_results.json")
