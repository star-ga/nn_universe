"""V12 item 2 — FIM-diagonal under real LM cross-entropy loss.

Closes the audit gap: V11.x measured Pythia / OLMoE / Mamba under
**Gaussian-probe self-prediction**, not under the actual next-token
cross-entropy loss that defines the model.  GPT-5.5 critique: "LM
evaluations appear not to use language-modeling loss".

This script re-measures the same checkpoints under
  loss = -log P(x_t | x_<t),  x ~ Pile validation
with $\ge 200$ probes from the Pile validation split.

Predicted (V12 prereg): every measured LM stays in the deep-sequential
band (T_1/T_3 > 100), with magnitude probably 0.3-1 dex lower than
Gaussian-probe (because real text has stronger structure than random
ids).
Falsifier: any LM drops out of the deep-sequential band under real
LM-loss probes.

Hardware:
  Pythia-1.4B (FP16):   ~3 GB    → fits 4070 (12 GB)
  Pythia-2.8B (FP16):   ~5.6 GB  → fits 4070
  Mamba-790M (FP16):    ~1.6 GB  → fits 4070
  OLMoE-1B-7B (FP16):   ~14 GB   → INT4 fallback on 4070; FP16 on H100/H200
  Pythia-6.9B (FP16):   ~14 GB   → INT4 fallback on 4070; FP16 on H100/H200

Usage:
    python3 experiments/v12_lm_loss_fim/run.py \
        --substrate pythia14b_pile_loss \
        --seed 0 \
        --out  results/<substrate>_seed<n>.json \
        --raw-fim-out results/<substrate>_seed<n>_raw.npy
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


REGISTRY = {
    "pythia14b_pile_loss":          {"model_id": "EleutherAI/pythia-1.4b",       "quant": "fp16"},
    "pythia28b_pile_loss":          {"model_id": "EleutherAI/pythia-2.8b",       "quant": "fp16"},
    "pythia69b_pile_loss":          {"model_id": "EleutherAI/pythia-6.9b",       "quant": "fp16"},   # H100/H200 only
    "mamba_790m_pile_loss":         {"model_id": "state-spaces/mamba-790m-hf",   "quant": "fp16"},
    "olmoe_1b7b_pile_loss":         {"model_id": "allenai/OLMoE-1B-7B-0924",     "quant": "fp16"},
    "olmoe_1b7b_pile_loss_int4":    {"model_id": "allenai/OLMoE-1B-7B-0924",     "quant": "int4"},
    "pythia69b_pile_loss_int4":     {"model_id": "EleutherAI/pythia-6.9b",       "quant": "int4"},
}


def t1_t3(x):
    x = np.asarray(x, dtype=np.float64); x = x[x >= 0]
    if x.size < 100: return float("nan")
    x = np.sort(x); n = x.size
    bot = x[: n // 2].mean(); top = x[-max(1, n // 100):].mean()
    return float(top / bot) if bot > 0 else float("inf")


def gini(x):
    x = np.asarray(x, dtype=np.float64); x = x[x >= 0]
    if x.size == 0 or x.sum() == 0: return 0.0
    x = np.sort(x); n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def effective_rank(x):
    x = np.asarray(x, dtype=np.float64); x = x[x > 0]
    if x.size == 0: return 0.0
    return float((x.sum() ** 2) / (x ** 2).sum())


def top_k_mass(x, k_frac=0.01):
    x = np.asarray(x, dtype=np.float64); x = x[x >= 0]
    if x.size == 0 or x.sum() == 0: return 0.0
    k = max(1, int(np.ceil(k_frac * x.size)))
    return float(np.sort(x)[-k:].sum() / x.sum())


def load_pile_probes(tokenizer, n_probes=200, seed=42, max_length=512):
    """Load $n_probes$ samples from the Pile validation set.

    Falls back to:
      1. HF Datasets `monology/pile-uncopyrighted` (validation split)
      2. Local cache at $PILE_VAL_CACHE
      3. Gaussian-text probes (with explicit JSON flag)
    """
    rng = np.random.default_rng(seed)

    # Path 1: HF Datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("monology/pile-uncopyrighted", split="validation", streaming=True)
        texts = []
        for i, item in enumerate(ds):
            if i >= n_probes * 3:  # oversample for filtering
                break
            t = item.get("text", "")
            if len(t) >= 200:
                texts.append(t[:4000])
        if len(texts) >= n_probes:
            chosen = rng.choice(len(texts), size=n_probes, replace=False)
            texts = [texts[i] for i in chosen]
            probes = []
            for t in texts:
                ids = tokenizer(t, return_tensors="pt", truncation=True,
                                max_length=max_length).input_ids
                if ids.size(1) >= 16:
                    probes.append(ids)
            if len(probes) >= n_probes:
                return probes[:n_probes], "pile_uncopyrighted_val"
    except Exception as e:
        print(f"  [warn] HF Pile load failed: {e}", flush=True)

    # Path 2: Local cache
    cache = os.environ.get("PILE_VAL_CACHE", "")
    if cache and Path(cache).exists():
        # one document per line, JSONL with {"text": ...}
        probes = []
        with open(cache) as f:
            lines = [ln for ln in f if len(ln) >= 200]
        rng.shuffle(lines)
        for ln in lines[: n_probes * 3]:
            try:
                t = json.loads(ln).get("text", "")[:4000]
                ids = tokenizer(t, return_tensors="pt", truncation=True,
                                max_length=max_length).input_ids
                if ids.size(1) >= 16:
                    probes.append(ids)
                if len(probes) >= n_probes:
                    break
            except Exception:
                continue
        if probes:
            return probes[:n_probes], "pile_val_local_cache"

    # Path 3: Gaussian-text fallback (matches V11.x convention, NOT real LM loss)
    print("  [warn] Falling back to Gaussian-text probes — NOT real LM loss", flush=True)
    vocab = tokenizer.vocab_size
    probes = []
    for i in range(n_probes):
        g = torch.Generator().manual_seed(seed + i)
        ids = torch.randint(0, vocab, (1, 64), generator=g)
        probes.append(ids)
    return probes, "gaussian_text_fallback"


def load_model(model_id: str, quant: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if quant == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    elif quant == "int4":
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb)
        except Exception as e:
            print(f"  [warn] INT4 failed ({e}), falling back to FP16", flush=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tok


def fim_diagonal_lm_loss(model, probes, device, n_probes=200):
    """Accumulate per-parameter g^2 under next-token cross-entropy loss.

    CPU FP32 accumulator. Streams probes — peak GPU mem matches a
    single forward+backward.
    """
    flat_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fim_acc = torch.zeros(flat_n, dtype=torch.float32, device="cpu")

    model.eval()  # but grad still flows
    for i, ids in enumerate(probes):
        if i >= n_probes: break
        ids = ids.to(device)
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        out.loss.backward()
        offset = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                n = p.numel()
                fim_acc[offset:offset + n] += (p.grad.detach() ** 2).flatten().to("cpu", dtype=torch.float32)
                offset += n
        if (i + 1) % 50 == 0:
            print(f"  probe {i+1}/{n_probes}", flush=True)

    return (fim_acc / n_probes).numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-fim-out", required=True)
    ap.add_argument("--n-probes", type=int, default=200)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.raw_fim_out).parent.mkdir(parents=True, exist_ok=True)

    reg = REGISTRY[args.substrate]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    model, tok = load_model(reg["model_id"], reg["quant"])
    model = model.to(device)
    load_s = time.time() - t0
    print(f"  loaded {reg['model_id']} ({reg['quant']}) in {load_s:.1f}s on {device}", flush=True)

    probes, probe_dist = load_pile_probes(tok, n_probes=args.n_probes, seed=args.seed)
    print(f"  loaded {len(probes)} probes ({probe_dist})", flush=True)

    t1 = time.time()
    fim = fim_diagonal_lm_loss(model, probes, device, n_probes=args.n_probes)
    fim_s = time.time() - t1

    np.save(args.raw_fim_out, fim.astype(np.float32))

    out = {
        "substrate":          args.substrate,
        "seed":               args.seed,
        "model_id":           reg["model_id"],
        "quantization":       reg["quant"],
        "loss_type":          "next_token_cross_entropy",
        "probe_distribution": probe_dist,
        "n_params":           int(fim.size),
        "n_probes":           args.n_probes,
        "load_time_s":        round(load_s, 1),
        "fim_compute_time_s": round(fim_s, 1),
        "t1_t3":              t1_t3(fim),
        "log10_t1_t3":        float(np.log10(t1_t3(fim))) if np.isfinite(t1_t3(fim)) else None,
        "gini":               gini(fim),
        "effective_rank_n":   effective_rank(fim) / fim.size,
        "top_1pct_mass":      top_k_mass(fim, 0.01),
        "raw_fim_path":       args.raw_fim_out,
        "schema_version":     "v12.1",
        "predicted":          "T_1/T_3 > 100, ~0.3-1 dex lower than Gaussian-probe V11.x",
        "falsifier":          "T_1/T_3 < 100 under real-LM-loss probes",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"OK  {args.substrate} seed={args.seed} "
          f"t1_t3={out['t1_t3']:.3g} ({reg['quant']}, {probe_dist}, total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
