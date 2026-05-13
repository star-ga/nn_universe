"""V12 item 4 — probe-convergence sweep at billion-param scale.

Closes the audit gap: V6.0b probe-convergence sweep is small-scale only
(MLPs ≤ 300k params).  This script repeats the sweep on Pythia-2.8B
(and 6.9B if memory permits) at probe counts {50, 100, 200, 400, 800,
1600} and reports T_1/T_3 stability as a function of n.

Predicted from V6.0b: stabilises by n=200 within ±5 % of n=1600.
Falsifier:           n=400 differs from n=1600 by > 5 % at any model.

Usage:
    python3 experiments/v12_probe_convergence_large.py \
        --model pythia-2.8b \
        --probe-counts 50,100,200,400,800,1600 \
        --seed 42 \
        --out results/probe_convergence_pythia28b.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def t1_t3(fim: np.ndarray) -> float:
    fim = np.asarray(fim, dtype=np.float64)
    fim = fim[fim >= 0]
    if fim.size < 100:
        return float("nan")
    fim = np.sort(fim)
    n = fim.size
    bottom_50 = fim[: n // 2].mean()
    top_1 = fim[-max(1, n // 100):].mean()
    return float(top_1 / bottom_50) if bottom_50 > 0 else float("inf")


def fim_diagonal_lm(model, tokenizer, n_probes: int, seed: int, device: str):
    """Incremental FIM-diagonal accumulator that streams probes.

    Returns: dict mapping n_probes_seen -> T_1/T_3 evaluated at that count.
    """
    fim_acc = None
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab_size = tokenizer.vocab_size
    snapshots = {}
    snapshot_at = set(n_probes if isinstance(n_probes, list) else [n_probes])
    max_probes = max(snapshot_at)

    model.eval()
    flat_param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model has {flat_param_size:,} trainable params", flush=True)

    if fim_acc is None:
        # CPU FP32 accumulator (vs ~22 GB GPU for 6.9B)
        fim_acc = torch.zeros(flat_param_size, dtype=torch.float32, device="cpu")

    for i in range(1, max_probes + 1):
        # Sample Gaussian-text probe at this seed
        probe_ids = torch.randint(0, vocab_size, (1, 64), generator=g).to(device)
        model.zero_grad(set_to_none=True)
        out = model(probe_ids, labels=probe_ids)
        out.loss.backward()

        # Accumulate g^2 into flat CPU buffer
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                fim_acc[offset:offset + n] += (p.grad.detach() ** 2).flatten().to("cpu", dtype=torch.float32)
                offset += n

        if i in snapshot_at:
            fim_arr = (fim_acc / i).numpy().astype(np.float64)
            snapshots[i] = {
                "n_probes": i,
                "t1_t3": t1_t3(fim_arr),
                "fim_mean": float(fim_arr.mean()),
                "fim_min": float(fim_arr.min()),
                "fim_max": float(fim_arr.max()),
            }
            print(f"  snapshot @ {i} probes: T1/T3 = {snapshots[i]['t1_t3']:.3g}", flush=True)

    return snapshots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["pythia-1.4b", "pythia-2.8b", "pythia-6.9b"])
    ap.add_argument("--probe-counts", default="50,100,200,400,800,1600",
                    help="Comma-separated probe counts to snapshot at")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    snapshot_counts = [int(x) for x in args.probe_counts.split(",")]

    print(f"== probe convergence at scale: {args.model} ==")
    print(f"  snapshot counts: {snapshot_counts}")
    print(f"  seed: {args.seed}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = f"EleutherAI/{args.model}"

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    load_s = time.time() - t0
    print(f"  loaded {model_id} in {load_s:.1f}s on {device}")

    t1 = time.time()
    snapshots = fim_diagonal_lm(model, tok, snapshot_counts, args.seed, device)
    fim_s = time.time() - t1

    # Convergence verdict: stable iff |T_1/T_3(n) - T_1/T_3(1600)| / T_1/T_3(1600) < 0.05 for all n >= 200
    ref = snapshots[max(snapshot_counts)]["t1_t3"]
    stable_at = {}
    for n in sorted(snapshot_counts):
        rel_err = abs(snapshots[n]["t1_t3"] - ref) / abs(ref) if ref != 0 else float("inf")
        stable_at[n] = bool(rel_err < 0.05)

    verdict = "STABLE_BY_200" if stable_at.get(200, False) else (
        "STABLE_BY_400" if stable_at.get(400, False) else "NOT_STABLE")

    out = {
        "model_id":          model_id,
        "seed":              args.seed,
        "load_time_s":       round(load_s, 1),
        "fim_compute_time_s": round(fim_s, 1),
        "snapshots":         snapshots,
        "convergence_stable_at_n": stable_at,
        "convergence_verdict": verdict,
        "schema_version":    "v12.1",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"OK  verdict = {verdict}")
    print(f"OK  total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
