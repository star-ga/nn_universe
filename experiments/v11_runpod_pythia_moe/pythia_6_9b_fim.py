"""V11 — Pythia-6.9B FIM measurement (production-scale gap, dense LM).

Closes the upper end of the Pythia-family FIM tier-hierarchy curve:
V8 1.4B and V9.10 2.8B are already on disk; this is the 6.9B point.

Schema (must match V9.10 Pythia-2.8B exactly, byte-for-byte keys):
    {
      "model_id":       "EleutherAI/pythia-6.9b",
      "n_params":       <int>,
      "n_probes":       200,
      "seed":           42,
      "T1T3":           <float>,
      "log10_T1T3":     <float>,
      "gini":           <float>,
      "gini_sample_size": 100000000,
      "eff_rank_n":     <float>,
      "top_1pct_mass":  <float>
    }

Memory plan (A100 80 GB):
  Model fp16              ~13.8 GB
  Per-probe acts/grads    ~6-10 GB (freed after backward)
  FIM accumulators (fp32) on **CPU** host RAM (~28 GB)
  ⇒ GPU peak < 30 GB; CPU peak < 60 GB. A100 80 GB pod has ≥ 200 GB host RAM.

Metric protocol (matches V9.10):
  * 200 text probes, float64 accumulation post-grad², seed 42.
  * Effective rank computed on positive-FIM entries with the full
    flat distribution; eff_rank_n = eff_rank / total_n.
  * Gini / top-1% mass / T1/T3 computed on a uniformly sampled
    100,000,000-element sub-vector (since the full 6.9B-element
    flat sort would not fit in RAM and matches V9.10 sampling).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


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


def _accumulate_fim_cpu(net, tokenizer, n_probes, device, seed, max_len=64):
    """Accumulate per-parameter grad-squared on CPU (saves GPU RAM)."""
    fim_cpu: dict[str, torch.Tensor] = {}
    for n, p in net.named_parameters():
        if p.requires_grad:
            fim_cpu[n] = torch.zeros(p.shape, dtype=torch.float32, device="cpu")
    net.eval()
    rng = np.random.default_rng(seed)
    for i in range(n_probes):
        text = SAMPLE_TEXTS[rng.integers(0, len(SAMPLE_TEXTS))]
        ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_len
        ).to(device)
        if ids.size(1) < 2:
            continue
        out = net(ids, labels=ids)
        loss = out.loss
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None and n in fim_cpu:
                fim_cpu[n] += (p.grad.detach().float() ** 2).cpu()
        if (i + 1) % 20 == 0:
            print(f"  probe {i+1}/{n_probes}", flush=True)
    for n in fim_cpu:
        fim_cpu[n] /= n_probes
    return fim_cpu


def _streaming_stats_and_sample(fim_cpu, sample_size, seed):
    """Single pass: total stats + uniform sample for sort-dependent metrics."""
    layer_sizes = [v.numel() for v in fim_cpu.values()]
    total_n = sum(layer_sizes)
    if total_n == 0:
        raise RuntimeError("FIM is empty — model has no trainable params?")

    pos_n = 0
    pos_sum = 0.0
    pos_sum_sq = 0.0
    sample_pieces: list[np.ndarray] = []
    rng = np.random.default_rng(seed + 1)

    actual_sample = min(sample_size, total_n)
    remaining = actual_sample
    for i, (name, v) in enumerate(fim_cpu.items()):
        flat = v.flatten().to(torch.float64).numpy()
        positive = flat[flat > 0]
        pos_n += positive.size
        pos_sum += float(positive.sum())
        pos_sum_sq += float((positive ** 2).sum())

        is_last = (i == len(fim_cpu) - 1)
        if is_last:
            k = remaining
        else:
            k = max(1, int(round(actual_sample * flat.size / total_n))) if flat.size else 0
            k = min(k, remaining)
        if k > 0 and flat.size > 0:
            idx = rng.integers(0, flat.size, size=k)
            sample_pieces.append(flat[idx])
            remaining -= k
        if remaining <= 0:
            break
    sample = (
        np.concatenate(sample_pieces) if sample_pieces else np.empty(0, dtype=np.float64)
    )
    return total_n, pos_n, pos_sum, pos_sum_sq, sample, actual_sample


def _gini_on_positive(values: np.ndarray) -> float:
    v = values[values >= 0].astype(np.float64)
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v.sort()
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def _top_1pct_mass(values: np.ndarray) -> float:
    v = values[values > 0].astype(np.float64)
    if v.size == 0:
        return 0.0
    s = np.sort(v)[::-1]
    k = max(1, int(s.size * 0.01))
    return float(s[:k].sum() / s.sum())


def _tier_ratio(values: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    s = np.sort(v)[::-1]
    n = s.size
    k1 = max(1, int(n * 0.01))
    k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean())
    t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz) // 10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1 / t3 if t3 > 0 else float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="EleutherAI/pythia-6.9b")
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gini-sample-size", type=int, default=100_000_000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Forward/backward dtype (FIM still accumulated in fp32 on CPU).",
    )
    args = ap.parse_args()

    if args.out is None:
        slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
        args.out = str(Path(__file__).resolve().parent / f"v11_{slug}_results.json")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} in {args.dtype}...", flush=True)
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    net = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device)
    print(f"  load time: {time.time() - t_load:.1f}s", flush=True)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  params: {n_params:,}", flush=True)

    print(f"\nMeasuring FIM diagonal with {args.n_probes} probes...", flush=True)
    t0 = time.time()
    fim_cpu = _accumulate_fim_cpu(net, tokenizer, args.n_probes, device, args.seed)
    t_fim = time.time() - t0
    print(f"  FIM accumulation: {t_fim:.1f}s", flush=True)

    # Free GPU model before sampling stats — saves headroom for downstream pods.
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Computing streaming stats + uniform sample...", flush=True)
    t1 = time.time()
    total_n, pos_n, pos_sum, pos_sum_sq, sample, actual_sample = (
        _streaming_stats_and_sample(fim_cpu, args.gini_sample_size, args.seed)
    )
    print(
        f"  total params seen: {total_n:,}; positive: {pos_n:,}; "
        f"sample size: {actual_sample:,}",
        flush=True,
    )

    eff_rank = (
        (pos_sum ** 2) / (pos_n * pos_sum_sq)
        if (pos_sum_sq > 0 and pos_n > 0)
        else 0.0
    )
    eff_rank_n = eff_rank / total_n if total_n > 0 else 0.0

    t1m, t3m, tier = _tier_ratio(sample)
    g = _gini_on_positive(sample)
    tp1 = _top_1pct_mass(sample)
    log10_t = float(np.log10(tier)) if tier > 0 and np.isfinite(tier) else float("inf")
    print(f"  streaming-stats time: {time.time() - t1:.1f}s", flush=True)

    print(f"\n=== {args.model} FIM (V11) ===")
    print(f"  N params       = {n_params:,}")
    print(f"  T1/T3          = {tier:.6e}")
    print(f"  log10(T1/T3)   = {log10_t:.6f}")
    print(f"  Gini           = {g:.6f}")
    print(f"  eff_rank_n     = {eff_rank_n:.6e}")
    print(f"  top-1% mass    = {tp1:.6f}")
    print(f"  FIM time       = {t_fim:.1f}s")

    payload = {
        "model_id": args.model,
        "n_params": int(n_params),
        "n_probes": int(args.n_probes),
        "seed": int(args.seed),
        "T1T3": tier,
        "log10_T1T3": log10_t,
        "gini": g,
        "gini_sample_size": int(actual_sample),
        "eff_rank_n": eff_rank_n,
        "top_1pct_mass": tp1,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
