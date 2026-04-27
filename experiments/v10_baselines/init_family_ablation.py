"""V10b — Initialization-family ablation (dynamical isometry falsifier).

Reviewer's #2 ablation request: the paper itself identifies dynamical
isometry as a clean falsifier — but doesn't run it. Run it here.

Hypothesis: dynamical-isometry initialization (orthogonal weights with
gain matched to activation) should suppress the FIM tier hierarchy
because Hanin-Nica's log-normal accumulation requires Var[log F] grow
in depth, which is exactly what isometric Jacobians prevent.

Init families compared at fixed depth L=8, width=128:
  1. Kaiming normal (default; far from isometry) — should give T1/T3 >> 100
  2. Xavier normal (fan-in/fan-out balanced) — moderate
  3. Orthogonal (gain=sqrt(2) for ReLU, near isometry) — should suppress
  4. Identity-perturbation (perfect isometry plus epsilon) — should fully suppress

Falsifier outcome: if orthogonal init has T1/T3 << 100 (far below 100
threshold), the dynamical-isometry prediction is confirmed. If
orthogonal init still has T1/T3 > 100, the mechanism story is wrong.
"""
from __future__ import annotations
import json, time, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_mlp_with_init(init_name, depth=8, width=128, dim=16, seed=0):
    torch.manual_seed(seed)
    layers = []
    in_d = dim
    for i in range(depth):
        out_d = width if i < depth - 1 else dim
        L = nn.Linear(in_d, out_d)
        if init_name == "kaiming":
            nn.init.kaiming_normal_(L.weight, nonlinearity='relu')
            nn.init.zeros_(L.bias)
        elif init_name == "xavier":
            nn.init.xavier_normal_(L.weight)
            nn.init.zeros_(L.bias)
        elif init_name == "orthogonal":
            # Orthogonal init with sqrt(2) gain (ReLU-corrected dynamical isometry)
            nn.init.orthogonal_(L.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(L.bias)
        elif init_name == "identity_eps":
            # Identity perturbation: W = I + eps*N(0,1)/sqrt(n)
            n = max(L.weight.shape)
            eps = 0.01
            with torch.no_grad():
                if L.weight.shape[0] == L.weight.shape[1]:
                    L.weight.copy_(torch.eye(L.weight.shape[0]) + eps * torch.randn_like(L.weight) / math.sqrt(n))
                else:
                    nn.init.orthogonal_(L.weight, gain=1.0)
                    L.weight.add_(eps * torch.randn_like(L.weight) / math.sqrt(n))
                nn.init.zeros_(L.bias)
        layers.append(L)
        if i < depth - 1:
            layers.append(nn.ReLU())
        in_d = out_d
    return nn.Sequential(*layers)


def fim_diagonal(net, dim, n_probes=200):
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for _ in range(n_probes):
        x = torch.randn(1, dim)
        y = net(x)
        target = x[:, :y.shape[1]]
        loss = 0.5 * (y - target).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data.double() ** 2
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def stats(fim):
    s = np.sort(fim)[::-1]
    n = len(s)
    k1 = max(1, int(n * 0.01))
    k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean()); t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    ratio = t1 / t3 if t3 > 0 else float("inf")
    v = fim[fim > 0]
    if v.size == 0:
        gini, top1 = 0.0, 0.0
    else:
        v_sorted = np.sort(v)
        n_v = v_sorted.size
        cum = np.cumsum(v_sorted)
        gini = float((n_v + 1 - 2 * np.sum(cum) / cum[-1]) / n_v)
        top1 = float(np.sort(v)[::-1][:max(1, int(v.size*0.01))].sum() / v.sum())
    return {"T1T3": ratio, "gini": gini, "top1_pct_mass": top1}


def main():
    out = []
    DEPTH = 8; WIDTH = 128; DIM = 16
    inits = ["kaiming", "xavier", "orthogonal", "identity_eps"]
    SEEDS = [0, 1, 2]

    for init_name in inits:
        per_seed = []
        for s in SEEDS:
            net = make_mlp_with_init(init_name, depth=DEPTH, width=WIDTH, dim=DIM, seed=s)
            fim = fim_diagonal(net, DIM, n_probes=200)
            r = stats(fim)
            r["seed"] = s
            per_seed.append(r)
        T1T3_vals = [r["T1T3"] for r in per_seed]
        gini_vals = [r["gini"] for r in per_seed]
        top1_vals = [r["top1_pct_mass"] for r in per_seed]
        log_T1T3 = [math.log10(max(v, 1e-30)) for v in T1T3_vals]
        agg = {
            "init": init_name,
            "depth": DEPTH, "width": WIDTH,
            "per_seed": per_seed,
            "T1T3_mean": float(np.mean(T1T3_vals)),
            "T1T3_std": float(np.std(T1T3_vals, ddof=1)),
            "log10_T1T3_mean": float(np.mean(log_T1T3)),
            "log10_T1T3_std": float(np.std(log_T1T3, ddof=1)),
            "gini_mean": float(np.mean(gini_vals)),
            "top1_mean": float(np.mean(top1_vals)),
            "above_threshold_100": all(v > 100 for v in T1T3_vals),
        }
        print(f"  {init_name:<14} log10(T1/T3)={agg['log10_T1T3_mean']:6.3f}±{agg['log10_T1T3_std']:.3f}  Gini={agg['gini_mean']:.3f}  top1={agg['top1_mean']:.3f}  above-100={agg['above_threshold_100']}", flush=True)
        out.append(agg)

    # Falsifier verdict
    kaiming = [r for r in out if r["init"] == "kaiming"][0]
    orth    = [r for r in out if r["init"] == "orthogonal"][0]
    iso_eps = [r for r in out if r["init"] == "identity_eps"][0]
    log_drop_orth = kaiming["log10_T1T3_mean"] - orth["log10_T1T3_mean"]
    log_drop_iso = kaiming["log10_T1T3_mean"] - iso_eps["log10_T1T3_mean"]
    verdict = {
        "kaiming_log10_T1T3": kaiming["log10_T1T3_mean"],
        "orthogonal_log10_T1T3": orth["log10_T1T3_mean"],
        "identity_eps_log10_T1T3": iso_eps["log10_T1T3_mean"],
        "log10_drop_kaiming_to_orthogonal": log_drop_orth,
        "log10_drop_kaiming_to_identity": log_drop_iso,
        "orthogonal_below_threshold_100": not orth["above_threshold_100"],
        "identity_below_threshold_100": not iso_eps["above_threshold_100"],
        "falsifier_outcome": (
            "CONFIRMED: orthogonal init drops T1/T3 below 100 threshold — "
            "dynamical isometry suppresses the hierarchy as predicted by "
            "Hanin-Nica mechanism"
            if not orth["above_threshold_100"] else
            "PARTIAL: orthogonal init reduces T1/T3 but not below 100 threshold; "
            "hierarchy partially survives near-isometric init (mechanism may need refinement)"
            if log_drop_orth > 0 else
            "FALSIFIED: orthogonal init does NOT reduce T1/T3 — Hanin-Nica "
            "log-normal mechanism is wrong (or finite-width corrections dominate)"
        ),
    }
    payload = {
        "config": {"depth": DEPTH, "width": WIDTH, "dim": DIM, "seeds": SEEDS, "n_probes": 200},
        "per_init": out,
        "falsifier_verdict": verdict,
    }
    p = Path(__file__).parent / "v10b_init_family_ablation_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n=== Falsifier verdict ===")
    print(f"  Kaiming      log10(T1/T3) = {kaiming['log10_T1T3_mean']:.3f}")
    print(f"  Orthogonal   log10(T1/T3) = {orth['log10_T1T3_mean']:.3f}  (drop {log_drop_orth:+.3f})")
    print(f"  Identity+eps log10(T1/T3) = {iso_eps['log10_T1T3_mean']:.3f}  (drop {log_drop_iso:+.3f})")
    print(f"  Verdict: {verdict['falsifier_outcome']}")
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
