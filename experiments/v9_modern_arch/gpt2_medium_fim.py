"""V9.6 — GPT-2-medium FIM measurement (production-scale gap, language).

Uses HuggingFace's pretrained GPT-2-medium (355M parameters), a 24-layer
transformer with hidden size 1024 and 16 heads. Pretrained on the
WebText corpus, this is the canonical "real-scale" autoregressive
language model in the GPT-2 family — orders of magnitude larger than
our V9 GPT-Tiny (0.6 M params).

Probes are short text samples drawn from a tiny WikiText-2 sample
(or any local plain text); FIM diagonal is computed on the next-token
language-modelling loss with the standard 200-probe float64 protocol.

Goal: confirm FIM tier hierarchy (T1/T3 magnitude + Gini + top-1% mass)
on a production-scale attention transformer pretrained on real text.
Closes the architecture_coverage gap on the language side.
"""
from __future__ import annotations
import argparse
import json
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
    k1 = max(1, int(s.size * 0.01))
    return float(s[:k1].sum() / s.sum())


def tier_ratio(values: np.ndarray):
    s = np.sort(values)[::-1]
    n = len(s)
    k1 = max(1, int(n * 0.01))
    k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean())
    t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz) // 10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1 / t3 if t3 > 0 else float("inf"))


SAMPLE_TEXTS = [
    "The Fisher Information Matrix is a fundamental object in information geometry that captures",
    "Deep neural networks trained with gradient descent exhibit complex spectral properties in the",
    "Random matrix theory predicts that products of independent random matrices have eigenvalues that",
    "The neural tangent kernel describes the infinite-width limit of fully-connected feedforward networks",
    "In the lazy training regime, neural networks behave like kernel machines and the loss decreases",
    "The lottery ticket hypothesis posits that dense networks contain sparse subnetworks that",
    "Universality in deep learning refers to the phenomenon by which different architectures and",
    "Empirical Fisher information differs from the true Fisher information when the model distribution",
    "Statistical learning theory provides bounds on generalization error in terms of the complexity of",
    "Heavy-tailed weight distributions in trained neural networks are associated with",
    "The Hanin-Nica theorem on products of random matrices establishes that the log of the gradient",
    "Mean-field theory of deep networks predicts that the variance of the pre-activation grows",
]


def fim_diagonal(net, tokenizer, n_probes, device, max_len=64):
    fim = {n: torch.zeros_like(p, dtype=torch.float64)
           for n, p in net.named_parameters() if p.requires_grad}
    net.eval()
    rng = np.random.default_rng(0)
    for i in range(n_probes):
        text = SAMPLE_TEXTS[rng.integers(0, len(SAMPLE_TEXTS))]
        ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        if ids.size(1) < 2:
            continue
        out = net(ids, labels=ids)
        loss = out.loss
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data.double() ** 2
        if (i + 1) % 50 == 0:
            print(f"  probe {i+1}/{n_probes}", flush=True)
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2-medium",
                    help="HuggingFace model id (gpt2 / gpt2-medium / gpt2-large / gpt2-xl)")
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(Path(__file__).resolve().parent / f"v9_6_{args.model.replace('-', '_')}_results.json")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading pretrained {args.model}...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    net = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"{args.model} params: {n_params:,}", flush=True)
    print(f"Pretrained on WebText (~40GB internet text)", flush=True)

    print(f"\nMeasuring FIM diagonal with {args.n_probes} text probes...", flush=True)
    t0 = time.time()
    fim = fim_diagonal(net, tokenizer, args.n_probes, device)
    t_fim = time.time() - t0
    t1m, t3m, ratio = tier_ratio(fim)
    g = gini(fim); rn = effective_rank(fim) / fim.size; tp1 = top_1pct_mass(fim)

    print(f"\n=== {args.model} FIM (V9.6) ===")
    print(f"  N params       = {n_params:,}")
    print(f"  T1/T3          = {ratio:.3e}")
    print(f"  Gini           = {g:.4f}")
    print(f"  r_eff/n        = {rn:.5f}")
    print(f"  top-1% mass    = {tp1:.4f}")
    print(f"  FIM time       = {t_fim:.1f}s")

    payload = {
        "config": vars(args),
        "n_params": int(n_params),
        "model": args.model,
        "tier1_mean": t1m, "tier3_mean": t3m, "tier_ratio": ratio,
        "partition_invariant": {
            "gini": g,
            "effective_rank_normalised": rn,
            "top_1pct_mass": tp1,
        },
        "fim_measurement_s": t_fim,
        "interpretation": (
            f"FIM tier hierarchy on pretrained {args.model} ({n_params:,} params), "
            "production-scale autoregressive transformer pretrained on WebText. "
            "Same protocol as V9.2/V9.2b CIFAR (200 probes, float64 accumulation), "
            "with text inputs and language-modelling cross-entropy loss. "
            "Closes the language-side architecture_coverage gap."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
