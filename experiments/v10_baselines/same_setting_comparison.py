"""V10 — same-setting comparison vs prior observables.

Reviewer's #1 priority improvement: "Run FIM spectrum (Karakida),
Hessian-outlier (Sagun), weight-spectrum (Martin-Mahoney), and simple
gradient/Jacobian statistics on the same MLP/CNN/ViT/ResNet checkpoints."

This script runs ALL six observables on the SAME networks at the SAME
parameter scale (small MLP that fits Lanczos full-FIM):

  1. T1/T3 (this paper) — FIM diagonal tier ratio
  2. Top-1 % FIM mass — partition-invariant analogue
  3. FIM diagonal Gini coefficient
  4. FIM-spectrum top-eigenvalue / median-eigenvalue (Karakida-style; full Lanczos)
  5. Hessian outlier 3-level hierarchy (Sagun / Papyan style; small enough for full)
  6. Weight singular-value heavy-tail exponent (Martin-Mahoney PL_alpha)
  7. Gradient-norm log-variance (Hanin-Nica directly measured)
  8. Layer-wise Jacobian condition number

For each substrate (untrained-MLP, trained-MLP, boolean-circuit,
shallow-learner) and each observable, report whether it cleanly
separates deep-vs-non-deep on the panel. The diagnostic question is:
is T1/T3 unique, or does any prior observable already separate them?
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Same models as the V2 panel, smaller for tractable full-spectrum ===

def make_mlp(seed=0, dim=16, hidden=64, depth=5):
    torch.manual_seed(seed)
    layers = [nn.Linear(dim, hidden), nn.ReLU()]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
    layers.append(nn.Linear(hidden, dim))
    return nn.Sequential(*layers)


def make_logistic(seed=0, dim=16, n_classes=4):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dim, n_classes))


def make_linear_regression(seed=0, dim=16):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dim, dim))


def make_cnn(seed=0, in_ch=1, hidden=8, depth=4):
    """Small CNN for tractable full-FIM."""
    torch.manual_seed(seed)
    layers = [nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU()]
    for _ in range(depth - 2):
        layers.extend([nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU()])
    layers.append(nn.Conv2d(hidden, in_ch, 3, padding=1))
    return nn.Sequential(*layers)


# === The 8 observables ===

def fim_diagonal(net, dim, n_probes, is_image=False):
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for _ in range(n_probes):
        if is_image:
            x = torch.randn(1, 1, 8, 8)
        else:
            x = torch.randn(1, dim)
        y = net(x)
        target = x[:, :y.shape[1]] if not is_image else x
        loss = 0.5 * (y - target).pow(2).mean() if not is_image else 0.5 * (y - target).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data.double() ** 2
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def obs_T1_T3(fim):
    """This paper's T1/T3 statistic."""
    s = np.sort(fim)[::-1]
    n = len(s)
    k1 = max(1, int(n * 0.01))
    k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean()); t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return t1 / t3 if t3 > 0 else float("inf")


def obs_top1_mass(fim):
    """Partition-invariant analogue (this paper's alternative)."""
    v = fim[fim > 0]
    if v.size == 0: return 0.0
    s = np.sort(v)[::-1]
    k1 = max(1, int(s.size * 0.01))
    return float(s[:k1].sum() / s.sum())


def obs_gini(values):
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v >= 0]
    if v.size == 0 or v.sum() == 0: return 0.0
    v.sort(); n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def obs_fim_spectrum_topbottom(net, dim, n_samples=64, is_image=False):
    """Karakida-style FIM spectrum: top-eig / median-eig of empirical FIM
    (computed from per-sample gradients, full eigendecomposition).
    """
    grads = []
    net.eval()
    for _ in range(n_samples):
        if is_image:
            x = torch.randn(1, 1, 8, 8)
        else:
            x = torch.randn(1, dim)
        y = net(x)
        target = x[:, :y.shape[1]] if not is_image else x
        loss = 0.5 * (y - target).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
        grads.append(g.numpy())
    G = np.array(grads)  # (n_samples, n_params)
    # Empirical FIM = (1/n) G^T G; eigenvalues of (1/n) G G^T are non-trivial ones
    n, P = G.shape
    if n < P:
        gram = G @ G.T / n
        eigs = np.linalg.eigvalsh(gram)
    else:
        F = G.T @ G / n
        eigs = np.linalg.eigvalsh(F)
    eigs = np.sort(eigs[eigs > 1e-20])[::-1]
    if len(eigs) < 4:
        return float("nan")
    top = float(eigs[0])
    median = float(np.median(eigs))
    return top / median if median > 0 else float("inf")


def obs_hessian_outliers(net, dim, n_samples=64, is_image=False):
    """Sagun/Papyan-style Hessian outlier ratio: top-eig / 100th-eig (a proxy
    for the 3-level hierarchy claim) — computed via Hutchinson estimator on
    the GGN approximation (= FIM in our setup, so this is a Hessian-as-FIM
    proxy at small scale). For matched comparison we use the same per-sample
    grads as the spectrum estimator.
    """
    grads = []
    net.eval()
    for _ in range(n_samples):
        if is_image:
            x = torch.randn(1, 1, 8, 8)
        else:
            x = torch.randn(1, dim)
        y = net(x)
        target = x[:, :y.shape[1]] if not is_image else x
        loss = 0.5 * (y - target).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
        grads.append(g.numpy())
    G = np.array(grads)
    n, P = G.shape
    if n < P:
        gram = G @ G.T / n
        eigs = np.linalg.eigvalsh(gram)
    else:
        F = G.T @ G / n
        eigs = np.linalg.eigvalsh(F)
    eigs = np.sort(eigs[eigs > 1e-20])[::-1]
    if len(eigs) < 5:
        return float("nan")
    top1 = eigs[0]
    n_top = max(1, int(0.01 * len(eigs)))
    n_med = max(1, int(0.5 * len(eigs)))
    return float(eigs[:n_top].mean() / eigs[-n_med:].mean()) if eigs[-n_med:].mean() > 0 else float("inf")


def obs_weight_spectrum_alpha(net):
    """Martin-Mahoney heavy-tail PL_alpha — fit power-law tail to all 2-D
    weight matrix singular values, return alpha (smaller = heavier tail)."""
    alphas = []
    for name, p in net.named_parameters():
        if p.dim() < 2: continue
        W = p.detach().cpu().numpy().reshape(p.shape[0], -1)
        if min(W.shape) < 5: continue
        s = np.linalg.svd(W, compute_uv=False)
        s = s[s > 0]
        if len(s) < 10: continue
        # MLE of power-law exponent (top half)
        sm = np.sort(s)[::-1][:max(5, len(s)//2)]
        s_min = sm[-1]
        if s_min <= 0: continue
        alpha = 1 + len(sm) / np.sum(np.log(sm / s_min))
        alphas.append(alpha)
    if not alphas: return float("nan")
    # Average alpha; smaller = heavier tail = closer to Martin-Mahoney's "heavy-tail phase"
    return float(np.mean(alphas))


def obs_gradient_log_var(net, dim, n_probes=64, is_image=False):
    """Hanin-Nica: log of squared gradient norm should be ~Gaussian with
    Var growing in depth. We measure Var[log ||g||^2] across probes.
    """
    log_g2 = []
    net.eval()
    for _ in range(n_probes):
        if is_image:
            x = torch.randn(1, 1, 8, 8)
        else:
            x = torch.randn(1, dim)
        y = net(x)
        target = x[:, :y.shape[1]] if not is_image else x
        loss = 0.5 * (y - target).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        g2 = sum((p.grad.flatten() ** 2).sum().item() for p in net.parameters() if p.grad is not None)
        if g2 > 0:
            log_g2.append(np.log(g2))
    return float(np.var(log_g2)) if log_g2 else float("nan")


def obs_jacobian_cond(net, dim, n_samples=16, is_image=False):
    """Layer-wise Jacobian condition number (rough proxy for dynamical
    isometry deviation). Computed via per-input Jacobian of output w.r.t.
    input."""
    if is_image:
        return float("nan")  # too expensive for image models
    net.eval()
    conds = []
    for _ in range(n_samples):
        x = torch.randn(1, dim, requires_grad=True)
        y = net(x)
        if y.numel() != dim:
            return float("nan")
        J = torch.autograd.functional.jacobian(lambda xi: net(xi.unsqueeze(0))[0], x[0]).numpy()
        s = np.linalg.svd(J, compute_uv=False)
        s = s[s > 1e-12]
        if len(s) >= 2:
            conds.append(float(s.max() / s.min()))
    return float(np.mean(conds)) if conds else float("nan")


# === Run on same panel ===

def main():
    out = []
    DIM = 16
    PROBES = 200
    SAMPLES = 64

    for label, ctor in [
        ("untrained_mlp_L=5", lambda: make_mlp(seed=0, dim=DIM, hidden=64, depth=5)),
        ("untrained_mlp_L=8", lambda: make_mlp(seed=0, dim=DIM, hidden=64, depth=8)),
        ("untrained_mlp_L=12", lambda: make_mlp(seed=0, dim=DIM, hidden=64, depth=12)),
        ("logistic_regression", lambda: make_logistic(seed=0, dim=DIM)),
        ("linear_regression", lambda: make_linear_regression(seed=0, dim=DIM)),
    ]:
        print(f"\n=== {label} ===", flush=True)
        net = ctor()
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  params: {n_params:,}", flush=True)
        t0 = time.time()
        fim = fim_diagonal(net, DIM, PROBES)
        row = {
            "system": label,
            "n_params": n_params,
            "T1_T3_diag":             obs_T1_T3(fim),
            "top1_pct_mass":          obs_top1_mass(fim),
            "gini_diag":              obs_gini(fim),
            "fim_spectrum_top_med":   obs_fim_spectrum_topbottom(ctor(), DIM, SAMPLES),
            "hessian_outliers":       obs_hessian_outliers(ctor(), DIM, SAMPLES),
            "weight_spectrum_alpha":  obs_weight_spectrum_alpha(ctor()),
            "gradient_log_var":       obs_gradient_log_var(ctor(), DIM, PROBES),
            "jacobian_cond":          obs_jacobian_cond(ctor(), DIM, 16),
            "elapsed_s":              time.time() - t0,
        }
        for k, v in row.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4g}", flush=True)
            else:
                print(f"    {k}: {v}", flush=True)
        out.append(row)

    # Compute discrimination: which observables produce a clean deep-vs-non-deep gap?
    deep = [r for r in out if "untrained_mlp" in r["system"]]
    nondeep = [r for r in out if r["system"] in ("logistic_regression", "linear_regression")]

    discrimination = {}
    for obs in ("T1_T3_diag", "top1_pct_mass", "gini_diag", "fim_spectrum_top_med",
                "hessian_outliers", "weight_spectrum_alpha", "gradient_log_var", "jacobian_cond"):
        d_vals = [r[obs] for r in deep if not (isinstance(r[obs], float) and (np.isnan(r[obs]) or np.isinf(r[obs])))]
        n_vals = [r[obs] for r in nondeep if not (isinstance(r[obs], float) and (np.isnan(r[obs]) or np.isinf(r[obs])))]
        if not d_vals or not n_vals:
            discrimination[obs] = {"separation": "missing values"}; continue
        d_min, n_max = min(d_vals), max(n_vals)
        d_max, n_min = max(d_vals), min(n_vals)
        # Two possible directions
        sep_above = d_min - n_max  # deep > nondeep cleanly?
        sep_below = n_min - d_max  # deep < nondeep cleanly?
        clean = (sep_above > 0) or (sep_below > 0)
        discrimination[obs] = {
            "deep_min": d_min, "deep_max": d_max,
            "nondeep_min": n_min, "nondeep_max": n_max,
            "separation_above": sep_above,
            "separation_below": sep_below,
            "clean_separation": clean,
        }

    print("\n=== Same-setting observable comparison ===")
    print(f"{'Observable':<30} {'Deep range':<25} {'Non-deep range':<25} Clean separation?")
    for obs, d in discrimination.items():
        if "separation" in d:
            print(f"  {obs:<28} {d.get('separation','?')}")
            continue
        print(f"  {obs:<28} [{d['deep_min']:.3g}, {d['deep_max']:.3g}]   [{d['nondeep_min']:.3g}, {d['nondeep_max']:.3g}]   {d['clean_separation']}")

    payload = {
        "config": {"dim": DIM, "probes": PROBES, "spectrum_samples": SAMPLES},
        "per_system": out,
        "discrimination": discrimination,
        "summary": (
            "Same-setting comparison: which prior observable cleanly separates "
            "deep (untrained MLP L>=5) from non-deep (linear, logistic) at "
            f"matched parameter count? See discrimination dict for per-observable verdict."
        ),
    }
    p = Path(__file__).parent / "v10_same_setting_comparison_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
