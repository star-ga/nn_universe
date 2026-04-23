"""Tier-1 item 4: CNN + Transformer architectural baselines.

Trains three architectures on a shared synthetic self-prediction task
and measures the FIM tier ratio + SV ratio. Answers: does the FIM
three-tier hierarchy appear in CNN and Transformer weights, or is it
MLP-specific?

Architectures
-------------
- MLPCtrl: 5-layer 256-neuron ReLU (V1.0 baseline, for calibration)
- SmallCNN: 4-block conv encoder + mirror deconv decoder,
  ResNet-style residuals
- SmallViT: ViT-Tiny-like patch transformer autoencoder, 4 blocks,
  hidden 192, 3 attention heads

Task
----
Self-prediction autoencoding of 32x32x3 Gaussian-noise images. Output:
reconstruct the input. Loss: MSE. Optimizer: Adam. The task is
architecture-agnostic, matches V1.0's self-prediction formulation,
and allows direct comparison of learned-weight spectra.

Each architecture is trained to comparable loss, then all 2D weight
matrices (Conv weights are flattened to C_out x (C_in * k * k)) are
analyzed for SV ratio and the diagonal FIM tier partition.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments/v2_1_qec"))
from analyze import fim_diagonal, tier_partition  # noqa: E402


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class MLPCtrl(nn.Module):
    """V1.0-style 5-layer 256-neuron ReLU autoencoder.

    Accepts either flat (B, in_dim) or image-shape (B, C, H, W) input;
    flattens/restores internally so all three architectures can share
    the same training loop and sample_loss callback.
    """

    def __init__(self, in_dim: int = 3 * 32 * 32, width: int = 256, layers: int = 5):
        super().__init__()
        self.in_dim = in_dim
        mods: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(width, width), nn.ReLU()]
        mods.append(nn.Linear(width, in_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = self.net(x.reshape(shape[0], -1))
        return y.reshape(shape)


class SmallCNN(nn.Module):
    """4-block conv encoder + mirror deconv decoder.

    Input:  (B, 3, 32, 32)  Output: (B, 3, 32, 32).
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.Conv2d(c2, c3, 4, stride=2, padding=1), nn.ReLU(),   # 8x8
            nn.Conv2d(c3, c4, 4, stride=2, padding=1), nn.ReLU(),   # 4x4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1), nn.ReLU(),  # 8x8
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1), nn.ReLU(),  # 16x16
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1), nn.ReLU(),  # 32x32
            nn.Conv2d(c1, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.decoder(h)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch: int, in_ch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.n_patches = (img_size // patch) ** 2
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, n_patches, dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(x)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_mult), nn.GELU(), nn.Linear(dim * mlp_mult, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    """ViT-Tiny-like patch autoencoder. Input (B,3,32,32)."""

    def __init__(self, img_size: int = 32, patch: int = 4, dim: int = 192, depth: int = 4, heads: int = 3):
        super().__init__()
        self.patch = patch
        self.img_size = img_size
        self.embed = PatchEmbed(img_size, patch, 3, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.embed.n_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.unembed = nn.Linear(dim, 3 * patch * patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h)
        # (B, N, 3*p*p) -> (B, 3, H, W)
        p = self.patch
        s = self.img_size // p
        h = self.unembed(h)  # (B, n_patches, 3*p*p)
        h = h.reshape(B, s, s, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, s * p, s * p)
        return h


# ------------------------------------------------------------------
# Training + analysis
# ------------------------------------------------------------------

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def sv_per_2d_weight(model: nn.Module) -> list[dict]:
    stats: list[dict] = []
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:
            w = p.detach()
            # Flatten Conv filters to 2D: (C_out, C_in * k * k).
            if w.dim() > 2:
                w = w.reshape(w.shape[0], -1)
            S = torch.linalg.svdvals(w.float())
            if S.numel() < 2:
                continue
            ratio = float(S[0] / S[-1]) if S[-1] > 1e-10 else float("inf")
            stats.append({
                "layer": name,
                "shape": list(p.shape),
                "sv_ratio": round(ratio, 2),
                "top3": [round(float(s), 4) for s in S[:3]],
            })
    return stats


def train_one(model: nn.Module, batch: int, steps: int, device: torch.device) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    mse = nn.MSELoss()
    final = None
    for step in range(steps):
        x = torch.randn(batch, 3, 32, 32, device=device)
        y = model(x)
        loss = mse(y, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final = float(loss.detach())
    return final or 0.0


def analyze_arch(arch_name: str, model: nn.Module, batch: int, fim_samples: int, device: torch.device) -> dict:
    svs = sv_per_2d_weight(model)
    max_sv = max((s["sv_ratio"] for s in svs), default=0.0)
    mse = nn.MSELoss()

    def sample_loss():
        x = torch.randn(batch, 3, 32, 32, device=device)
        return mse(model(x), x)

    fim = fim_diagonal(model, sample_loss, n_samples=fim_samples)
    tiers = tier_partition(fim)
    return {
        "arch": arch_name,
        "params": count_params(model),
        "max_sv_ratio": round(max_sv, 1),
        "fim_tier1_tier3": round(tiers["ratio_tier1_tier3"], 1),
        "top_sv_layers": sorted(svs, key=lambda s: -s["sv_ratio"])[:5],
    }


def run_arch(name: str, device: torch.device, seed: int, steps: int, batch: int, fim_samples: int) -> dict:
    torch.manual_seed(seed)
    if name == "mlp":
        model = MLPCtrl(in_dim=3 * 32 * 32, width=256, layers=5).to(device)
    elif name == "cnn":
        model = SmallCNN(base_ch=32).to(device)
    elif name == "vit":
        model = SmallViT(img_size=32, patch=4, dim=192, depth=4, heads=3).to(device)
    else:
        raise ValueError(name)
    t0 = time.time()
    final_loss = train_one(model, batch=batch, steps=steps, device=device)
    train_time = time.time() - t0

    out = analyze_arch(name, model, batch=batch, fim_samples=fim_samples, device=device)
    out["train_time"] = round(train_time, 1)
    out["final_loss"] = round(final_loss, 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["mlp", "cnn", "vit"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fim-samples", type=int, default=300)
    ap.add_argument("--out", type=str, default=str(HERE / "v3_0_arch_baselines_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu})", flush=True)

    results = []
    for name in args.archs:
        print(f"[{time.strftime('%H:%M:%S')}] arch={name}", flush=True)
        try:
            r = run_arch(name, device=device, seed=args.seed, steps=args.steps, batch=args.batch, fim_samples=args.fim_samples)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at arch={name}")
            torch.cuda.empty_cache()
            continue
        print(
            f"  params={r['params']:,}  SV={r['max_sv_ratio']}x  "
            f"FIM={r['fim_tier1_tier3']}x  loss={r['final_loss']:.6f}  ({r['train_time']}s)",
            flush=True,
        )
        results.append(r)
        torch.cuda.empty_cache()

    payload = {
        "config": vars(args),
        "device": str(device),
        "gpu": gpu,
        "results": results,
        "interpretation": (
            "If SV and FIM tier ratios are similar across MLP, CNN, ViT at comparable "
            "parameter counts, the spectral hierarchy is architecture-universal (not "
            "MLP-specific). If they differ by orders of magnitude, it is task- or "
            "architecture-dependent."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
