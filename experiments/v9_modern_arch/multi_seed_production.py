"""V9.x-multi-seed — production-scale FIM with seed uncertainty.

Reviewer concern: production-scale points lack visible multi-seed
treatment. The pretrained weights are deterministic, so "seed" varies
the random Gaussian probe set used to estimate FIM_ii = E[(d_l/dtheta_i)^2].
We use 5 different probe seeds for each model and report mean / std /
95 % bootstrap CI on T1/T3 + Gini + top-1% mass.

Models covered:
  - ResNet-50 V1 (ImageNet-1K-V1, 76.13% top-1)
  - ResNet-50 V2 (ImageNet-1K-V2, 80.86% top-1)
  - ViT-L/16 SWAG-LINEAR (79.66% top-1)
  - GPT-2-medium (WebText)
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gini(values):
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v >= 0]
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v.sort()
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def effective_rank(values):
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v > 0]
    n = v.size
    if n == 0:
        return 0.0
    return float((v.sum() ** 2) / (n * (v ** 2).sum()))


def top_1pct_mass(values):
    v = np.asarray(values, dtype=np.float64).flatten()
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    s = np.sort(v)[::-1]
    k1 = max(1, int(s.size * 0.01))
    return float(s[:k1].sum() / s.sum())


def tier_ratio(values):
    s = np.sort(values)[::-1]
    n = len(s)
    k1 = max(1, int(n * 0.01))
    k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean())
    t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz) // 10, 1):].mean()) if len(nz) else 1e-30
    return (t1 / t3 if t3 > 0 else float("inf"))


def imagenet_probe(device, seed, batch=1, size=224):
    g = torch.Generator(device=device).manual_seed(seed)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.randn(batch, 3, size, size, device=device, generator=g)
    return (x - mean) / std


SAMPLE_TEXTS = [
    "The Fisher Information Matrix is a fundamental object in information geometry that captures",
    "Deep neural networks trained with gradient descent exhibit complex spectral properties in the",
    "Random matrix theory predicts that products of independent random matrices have eigenvalues that",
    "The neural tangent kernel describes the infinite-width limit of fully-connected feedforward networks",
    "In the lazy training regime, neural networks behave like kernel machines and the loss decreases",
    "Universality in deep learning refers to the phenomenon by which different architectures",
    "Empirical Fisher information differs from the true Fisher information when the model distribution",
    "Statistical learning theory provides bounds on generalization error in terms of the complexity of",
    "Heavy-tailed weight distributions in trained neural networks are associated with",
    "The Hanin-Nica theorem on products of random matrices establishes that the log of the gradient",
]


def fim_for_image_model(net, device, n_probes, seed, input_size=224, n_classes=1000):
    g = torch.Generator(device=device).manual_seed(seed * 31337 + 1)
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    for i in range(n_probes):
        x = imagenet_probe(device, seed=seed * 100 + i, size=input_size)
        out = net(x)
        target = torch.randint(0, out.shape[1], (1,), device=device, generator=g)
        loss = F.cross_entropy(out, target)
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim[n] += p.grad.data.double() ** 2
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def fim_for_gpt2(net, tokenizer, device, n_probes, seed, max_len=64):
    rng = np.random.default_rng(seed * 31337 + 1)
    fim = {n: torch.zeros_like(p, dtype=torch.float64)
           for n, p in net.named_parameters() if p.requires_grad}
    net.eval()
    for _ in range(n_probes):
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
    for n in fim:
        fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def stats_for_seed(fim):
    return {
        "T1T3": tier_ratio(fim),
        "log10_T1T3": float(np.log10(max(tier_ratio(fim), 1e-30))),
        "gini": gini(fim),
        "effective_rank_n": effective_rank(fim) / fim.size,
        "top_1pct_mass": top_1pct_mass(fim),
    }


def run_model(model_name, n_probes, seeds, device):
    print(f"\n=== {model_name} multi-seed ({len(seeds)} seeds, {n_probes} probes each) ===", flush=True)
    out_per_seed = []
    if model_name == "resnet50_v1":
        from torchvision.models import resnet50, ResNet50_Weights
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
            t0 = time.time()
            fim = fim_for_image_model(net, device, n_probes, s)
            stats = stats_for_seed(fim); stats["elapsed_s"] = time.time() - t0
            out_per_seed.append({"seed": s, **stats})
            print(f"  seed={s:>2} T1/T3={stats['T1T3']:.3e} Gini={stats['gini']:.3f} top1%={stats['top_1pct_mass']:.3f} ({stats['elapsed_s']:.1f}s)", flush=True)
            del net; torch.cuda.empty_cache()
    elif model_name == "resnet50_v2":
        from torchvision.models import resnet50, ResNet50_Weights
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
            t0 = time.time()
            fim = fim_for_image_model(net, device, n_probes, s)
            stats = stats_for_seed(fim); stats["elapsed_s"] = time.time() - t0
            out_per_seed.append({"seed": s, **stats})
            print(f"  seed={s:>2} T1/T3={stats['T1T3']:.3e} Gini={stats['gini']:.3f} top1%={stats['top_1pct_mass']:.3f} ({stats['elapsed_s']:.1f}s)", flush=True)
            del net; torch.cuda.empty_cache()
    elif model_name == "vit_l16":
        from torchvision.models import vit_l_16, ViT_L_16_Weights
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            net = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1).to(device)
            t0 = time.time()
            fim = fim_for_image_model(net, device, n_probes, s)
            stats = stats_for_seed(fim); stats["elapsed_s"] = time.time() - t0
            out_per_seed.append({"seed": s, **stats})
            print(f"  seed={s:>2} T1/T3={stats['T1T3']:.3e} Gini={stats['gini']:.3f} top1%={stats['top_1pct_mass']:.3f} ({stats['elapsed_s']:.1f}s)", flush=True)
            del net; torch.cuda.empty_cache()
    elif model_name == "gpt2_medium":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            net = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
            t0 = time.time()
            fim = fim_for_gpt2(net, tokenizer, device, n_probes, s)
            stats = stats_for_seed(fim); stats["elapsed_s"] = time.time() - t0
            out_per_seed.append({"seed": s, **stats})
            print(f"  seed={s:>2} T1/T3={stats['T1T3']:.3e} Gini={stats['gini']:.3f} top1%={stats['top_1pct_mass']:.3f} ({stats['elapsed_s']:.1f}s)", flush=True)
            del net; torch.cuda.empty_cache()
    return out_per_seed


def aggregate(per_seed_list):
    arr = lambda k: np.array([s[k] for s in per_seed_list])
    out = {}
    for key in ("T1T3", "log10_T1T3", "gini", "effective_rank_n", "top_1pct_mass"):
        v = arr(key)
        # Bootstrap CI from per-seed values (n=5 -> small but illustrative)
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(2000)])
        out[key] = {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()),
            "max": float(v.max()),
            "ci95_low": float(np.percentile(bs, 2.5)),
            "ci95_high": float(np.percentile(bs, 97.5)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, nargs="+",
                    default=["resnet50_v1", "resnet50_v2", "vit_l16", "gpt2_medium"])
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "v9_multi_seed_production_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    payload = {"config": vars(args), "results": {}}
    for model_name in args.models:
        per_seed = run_model(model_name, args.n_probes, args.seeds, device)
        agg = aggregate(per_seed)
        payload["results"][model_name] = {
            "per_seed": per_seed,
            "aggregate": agg,
        }
        print(f"\n  {model_name} aggregate:", flush=True)
        print(f"    log10(T1/T3): mean={agg['log10_T1T3']['mean']:.3f}  std={agg['log10_T1T3']['std']:.3f}  CI95=[{agg['log10_T1T3']['ci95_low']:.3f}, {agg['log10_T1T3']['ci95_high']:.3f}]", flush=True)
        print(f"    Gini:         mean={agg['gini']['mean']:.4f}  std={agg['gini']['std']:.4f}", flush=True)
        print(f"    top-1% mass:  mean={agg['top_1pct_mass']['mean']:.4f}  std={agg['top_1pct_mass']['std']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
