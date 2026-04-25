#!/usr/bin/env python3
"""
Intermediate-Scale Experiment — SV ratio + FIM hierarchy across 6 orders of magnitude.
Widths: 16, 64, 256, 1024, 4096, 8192
All 5-layer, I/O=32, seed 42. Reports R^2 for power-law fits.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f"Device: {device} ({gpu})")

dim = 32
widths = [16, 64, 256, 1024, 4096]

# Try 8192 if enough VRAM
try:
    test = torch.randn(1, 8192, device=device)
    del test
    widths.append(8192)
    print("Including width=8192")
except:
    print("Skipping width=8192 (OOM)")

results = []

for width in widths:
    torch.manual_seed(42)
    t0 = time.time()

    # Build network
    layers = [nn.Linear(dim, width), nn.ReLU()]
    for _ in range(3):  # 3 more hidden layers = 5 total with output
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers.append(nn.Linear(width, dim))
    net = nn.Sequential(*layers).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    # Train — self-prediction
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for step in range(20000):
        x = torch.randn(128, dim, device=device)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    train_time = time.time() - t0

    # SVD analysis
    max_sv_ratio = 0
    for name, param in net.named_parameters():
        if 'weight' in name and param.dim() == 2:
            S = torch.linalg.svdvals(param.data)
            ratio = float(S[0] / S[-1]) if S[-1] > 1e-10 else float('inf')
            max_sv_ratio = max(max_sv_ratio, ratio)

    # FIM diagonal (per-sample)
    net.eval()
    fim_vals = []
    fim_diag = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
    for _ in range(500):
        x = torch.randn(1, dim, device=device)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        net.zero_grad()
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim_diag[n] += p.grad.data ** 2
    for n in fim_diag:
        fim_diag[n] /= 500

    all_fim = torch.cat([v.flatten() for v in fim_diag.values()]).cpu().numpy()
    all_fim_sorted = np.sort(all_fim)[::-1]
    t1_thresh = np.percentile(all_fim_sorted, 99)
    t2_thresh = np.percentile(all_fim_sorted, 50)
    n_t1 = int(np.sum(all_fim_sorted >= t1_thresh))
    n_t3 = int(np.sum(all_fim_sorted < t2_thresh))
    t1_mean = float(np.mean(all_fim_sorted[:n_t1])) if n_t1 > 0 else 0
    t3_mean = float(np.mean(all_fim_sorted[-n_t3:])) if n_t3 > 0 else 1e-20
    fim_ratio = t1_mean / t3_mean if t3_mean > 0 else float('inf')

    r = {
        'width': width,
        'params': n_params,
        'max_sv_ratio': round(max_sv_ratio, 1),
        'fim_tier1_tier3': round(fim_ratio, 1),
        'train_time': round(train_time, 1),
    }
    results.append(r)
    print(f"  width={width:5d} params={n_params:>10,} SV={max_sv_ratio:>10.1f}x FIM={fim_ratio:>12.1f}x ({train_time:.1f}s)")
    del net, opt
    torch.cuda.empty_cache()

# Power-law fits (log-log)
params = np.array([r['params'] for r in results])
sv = np.array([r['max_sv_ratio'] for r in results])
fim = np.array([r['fim_tier1_tier3'] for r in results])

log_p = np.log10(params)
log_sv = np.log10(np.clip(sv, 1, None))
log_fim = np.log10(np.clip(fim, 1, None))

sv_fit = np.polyfit(log_p, log_sv, 1)
sv_pred = np.polyval(sv_fit, log_p)
ss_res = np.sum((log_sv - sv_pred)**2)
ss_tot = np.sum((log_sv - np.mean(log_sv))**2)
sv_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

fim_fit = np.polyfit(log_p, log_fim, 1)
fim_pred = np.polyval(fim_fit, log_p)
ss_res = np.sum((log_fim - fim_pred)**2)
ss_tot = np.sum((log_fim - np.mean(log_fim))**2)
fim_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

print(f"\nSV power-law: exponent={sv_fit[0]:.3f}, R²={sv_r2:.3f}")
print(f"FIM power-law: exponent={fim_fit[0]:.3f}, R²={fim_r2:.3f}")

output = {
    'device': str(device),
    'gpu': gpu,
    'dim': dim,
    'hidden_layers': 5,
    'results': results,
    'sv_power_law': {'exponent': round(sv_fit[0], 3), 'r_squared': round(sv_r2, 3)},
    'fim_power_law': {'exponent': round(fim_fit[0], 3), 'r_squared': round(fim_r2, 3)},
}
with open('scaling_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to scaling_results.json")
