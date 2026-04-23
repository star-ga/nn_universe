"""Train a neural toric-code decoder and measure its learned-weight spectra.

Mirrors the V1.0 cosmology-experiment protocol (5-layer 256-neuron ReLU MLP,
SGD+momentum, 50k steps) so the measured SV ratios and FIM tier hierarchy
are directly comparable across the two tasks.

Outputs `v2_1_qec_results.json` in the experiment directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import fim_diagonal, sv_per_layer, tier_partition  # noqa: E402
from decoder import MLPDecoder  # noqa: E402
from toric_code import compute_syndromes, logical_error_rate, make_batch, parity_check_matrix  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=5, help="toric code distance (L x L torus)")
    p.add_argument("--p", type=float, default=0.05, help="bit-flip error rate")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--hidden-layers", type=int, default=5)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--fim-samples", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(HERE / "v2_1_qec_results.json"))
    p.add_argument("--loss", choices=["mse", "bce"], default="bce")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu})")
    print(
        f"Config: L={args.L} p={args.p} width={args.width} layers={args.hidden_layers} "
        f"steps={args.steps} loss={args.loss} seed={args.seed}"
    )

    # Toric code setup.
    L = args.L
    H = parity_check_matrix(L)
    n_syndromes = H.shape[0]
    n_qubits = H.shape[1]
    H_torch = torch.from_numpy(H.astype(np.float32)).to(device)

    net = MLPDecoder(
        n_syndromes=n_syndromes,
        n_qubits=n_qubits,
        width=args.width,
        hidden_layers=args.hidden_layers,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {n_params:,}")

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    # -----------------  Training  -----------------
    t0 = time.time()
    training_losses: list[float] = []
    for step in range(args.steps):
        s_np, e_np = make_batch(L, args.p, args.batch, rng, H)
        s = torch.from_numpy(s_np).to(device)
        e = torch.from_numpy(e_np).to(device)

        logits = net(s)
        if args.loss == "bce":
            loss = bce(logits, e)
        else:
            loss = 0.5 * (torch.sigmoid(logits) - e).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(1, args.steps // 10) == 0:
            training_losses.append(float(loss))
            print(f"  step {step:5d}: loss={float(loss):.6f}")
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    # -----------------  Evaluation  -----------------
    net.eval()
    with torch.no_grad():
        s_np, e_np = make_batch(L, args.p, 5_000, rng, H)
        s = torch.from_numpy(s_np).to(device)
        e_true = torch.from_numpy(e_np).to(device)
        logits = net(s)
        pred = torch.sigmoid(logits).cpu().numpy()
        e_np_cpu = e_true.cpu().numpy()
    ler = logical_error_rate(pred, e_np_cpu, H, L)
    print(f"Residual-syndrome error rate: {ler:.4f}  (p_in={args.p})")

    # -----------------  SVD analysis  -----------------
    print("\n=== SVD analysis ===")
    sv_stats = sv_per_layer(net)
    for s in sv_stats:
        print(f"  {s['layer']}: top3={s['top3_sv']}, ratio={s['sv_ratio']:.1f}x, std={s['sv_std']}")
    max_sv = max(s["sv_ratio"] for s in sv_stats) if sv_stats else 0.0

    # -----------------  FIM analysis  -----------------
    print("\n=== FIM diagonal ===")
    bce_red = nn.BCEWithLogitsLoss()

    def sample_loss() -> torch.Tensor:
        s_np, e_np = make_batch(L, args.p, 32, rng, H)
        s = torch.from_numpy(s_np).to(device)
        e = torch.from_numpy(e_np).to(device)
        logits = net(s)
        return bce_red(logits, e)

    fim = fim_diagonal(net, sample_loss, n_samples=args.fim_samples)
    tiers = tier_partition(fim)
    print(
        f"  total={tiers['total_params']:,}  "
        f"tier1_count={tiers['tier1']['count']}  "
        f"tier1/tier3={tiers['ratio_tier1_tier3']:.1f}x"
    )

    # -----------------  Save  -----------------
    results = {
        "device": str(device),
        "gpu": gpu,
        "config": vars(args),
        "toric_code": {"L": L, "n_qubits": n_qubits, "n_syndromes": n_syndromes, "p": args.p},
        "n_params": n_params,
        "training_losses": training_losses,
        "train_time_sec": round(train_time, 1),
        "residual_syndrome_error_rate": ler,
        "sv_stats": sv_stats,
        "max_sv_ratio": max_sv,
        "fim": tiers,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
