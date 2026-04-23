#!/usr/bin/env python3
"""
V1.2 scaling — extend the SV-ratio / FIM-hierarchy sweep beyond 8192.

Reads existing ``scaling_results.json`` if present, runs one or more
additional widths, appends the results, and re-computes the power-law
fit. Safe to interrupt: each width is committed before moving on.

Recipe for 10^9 — 10^10 param runs on H200 NVL 141 GB::

    # 1B params (~width 14000)
    MIND_NN_WIDTH=14000 MIND_NN_BF16=1 MIND_NN_GRAD_CKPT=1 \\
        python3 scaling_experiment_extended.py

    # 10B params (~width 45000) — H200 80GB+ required
    MIND_NN_WIDTH=45000 MIND_NN_BF16=1 MIND_NN_GRAD_CKPT=1 \\
        python3 scaling_experiment_extended.py

Env knobs (all optional):
    MIND_NN_WIDTH        — width to run (default: next missing width from a fixed ladder)
    MIND_NN_STEPS        — training steps (default: 20000, ≥5000 recommended)
    MIND_NN_BATCH        — batch size (default: 128; drop on OOM)
    MIND_NN_BF16         — 1 to train in bf16 (halves activation memory)
    MIND_NN_GRAD_CKPT    — 1 to enable torch.utils.checkpoint on hidden blocks
    MIND_NN_DIM          — input/output dim (default: 32, kept constant)
    MIND_NN_SEED         — RNG seed (default: 42)
    MIND_NN_FIM_SAMPLES  — FIM diagonal samples (default: 500; drop for speed)
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

_DEFAULT_LADDER = [12000, 16384, 22000, 32000, 45000, 65536]
_DIM = int(os.environ.get("MIND_NN_DIM", "32"))
_SEED = int(os.environ.get("MIND_NN_SEED", "42"))
_STEPS = int(os.environ.get("MIND_NN_STEPS", "20000"))
_BATCH = int(os.environ.get("MIND_NN_BATCH", "128"))
_BF16 = os.environ.get("MIND_NN_BF16") == "1"
_GRAD_CKPT = os.environ.get("MIND_NN_GRAD_CKPT") == "1"
_FIM_SAMPLES = int(os.environ.get("MIND_NN_FIM_SAMPLES", "500"))
_RESULTS_PATH = "scaling_results.json"


def _pick_width() -> int:
    """Return MIND_NN_WIDTH if set; else the first ladder width not in the results file."""
    explicit = os.environ.get("MIND_NN_WIDTH")
    if explicit:
        return int(explicit)
    if not os.path.exists(_RESULTS_PATH):
        return _DEFAULT_LADDER[0]
    with open(_RESULTS_PATH) as f:
        existing = {r["width"] for r in json.load(f)["results"]}
    for w in _DEFAULT_LADDER:
        if w not in existing:
            return w
    raise RuntimeError("Every ladder width already has a result; set MIND_NN_WIDTH explicitly.")


class _CheckpointedFC(nn.Module):
    """5-layer ReLU FC network. Hidden blocks optionally grad-checkpointed."""

    def __init__(self, dim: int, width: int, grad_ckpt: bool) -> None:
        super().__init__()
        self.stem = nn.Linear(dim, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
        self.head = nn.Linear(width, dim)
        self.grad_ckpt = grad_ckpt

    def _hidden_forward(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return torch.relu(self.hidden[layer_idx](h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        for i in range(len(self.hidden)):
            if self.grad_ckpt and self.training:
                # use_reentrant=False is the safer PyTorch 2.1+ path.
                h = cp.checkpoint(self._hidden_forward, h, i, use_reentrant=False)
            else:
                h = self._hidden_forward(h, i)
        return self.head(h)


def _build_net(width: int) -> tuple[nn.Module, torch.device, torch.dtype]:
    torch.manual_seed(_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (_BF16 and device.type == "cuda") else torch.float32
    net = _CheckpointedFC(_DIM, width, grad_ckpt=_GRAD_CKPT).to(device=device, dtype=dtype)
    return net, device, dtype


def _train(net: nn.Module, device: torch.device, dtype: torch.dtype) -> float:
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    t0 = time.time()
    for step in range(_STEPS):
        x = torch.randn(_BATCH, _DIM, device=device, dtype=dtype)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return time.time() - t0


def _max_sv_ratio(net: nn.Module) -> float:
    max_ratio = 0.0
    for name, param in net.named_parameters():
        if "weight" in name and param.dim() == 2:
            # SVD needs fp32 for numerical stability on bf16 weights.
            S = torch.linalg.svdvals(param.data.float())
            if S[-1] > 1e-10:
                max_ratio = max(max_ratio, float(S[0] / S[-1]))
    return max_ratio


def _fim_tier_ratio(net: nn.Module, device: torch.device, dtype: torch.dtype) -> float:
    net.eval()
    fim_diag: dict[str, torch.Tensor] = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in net.named_parameters()}
    for _ in range(_FIM_SAMPLES):
        x = torch.randn(1, _DIM, device=device, dtype=dtype)
        loss = 0.5 * (net(x) - x).pow(2).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fim_diag[n] += p.grad.data.float() ** 2
    for n in fim_diag:
        fim_diag[n] /= _FIM_SAMPLES
    all_fim = torch.cat([v.flatten() for v in fim_diag.values()]).cpu().numpy()
    all_fim_sorted = np.sort(all_fim)[::-1]
    t1_thresh = np.percentile(all_fim_sorted, 99)
    t2_thresh = np.percentile(all_fim_sorted, 50)
    n_t1 = int(np.sum(all_fim_sorted >= t1_thresh))
    n_t3 = int(np.sum(all_fim_sorted < t2_thresh))
    t1_mean = float(np.mean(all_fim_sorted[:n_t1])) if n_t1 > 0 else 0.0
    t3_mean = float(np.mean(all_fim_sorted[-n_t3:])) if n_t3 > 0 else 1e-20
    return t1_mean / t3_mean if t3_mean > 0 else math.inf


def _load_results() -> dict:
    if os.path.exists(_RESULTS_PATH):
        with open(_RESULTS_PATH) as f:
            return json.load(f)
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "dim": _DIM,
        "hidden_layers": 5,
        "results": [],
        "sv_power_law": {"exponent": 0.0, "r_squared": 0.0},
        "fim_power_law": {"exponent": 0.0, "r_squared": 0.0},
    }


def _save_results(payload: dict) -> None:
    with open(_RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def _refit_power_law(results: list[dict]) -> tuple[dict, dict]:
    if len(results) < 2:
        return {"exponent": 0.0, "r_squared": 0.0}, {"exponent": 0.0, "r_squared": 0.0}
    params = np.array([r["params"] for r in results])
    sv = np.clip(np.array([r["max_sv_ratio"] for r in results]), 1, None)
    fim = np.clip(np.array([r["fim_tier1_tier3"] for r in results]), 1, None)
    log_p = np.log10(params)
    log_sv = np.log10(sv)
    log_fim = np.log10(fim)
    sv_fit = np.polyfit(log_p, log_sv, 1)
    fim_fit = np.polyfit(log_p, log_fim, 1)

    def _r2(y: np.ndarray, fit: np.ndarray) -> float:
        pred = np.polyval(fit, log_p)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return (
        {"exponent": round(float(sv_fit[0]), 3), "r_squared": round(_r2(log_sv, sv_fit), 3)},
        {"exponent": round(float(fim_fit[0]), 3), "r_squared": round(_r2(log_fim, fim_fit), 3)},
    )


def main() -> None:
    width = _pick_width()
    print(f"[{time.strftime('%H:%M:%S')}] running width={width}  bf16={_BF16}  grad_ckpt={_GRAD_CKPT}")

    try:
        net, device, dtype = _build_net(width)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  params = {n_params:,}  device={device}  dtype={dtype}")
        train_time = _train(net, device, dtype)
        sv_ratio = _max_sv_ratio(net)
        fim_ratio = _fim_tier_ratio(net, device, dtype)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"  OOM at width={width} — try a smaller MIND_NN_WIDTH or enable BF16/GRAD_CKPT. ({exc})")
        return
    finally:
        torch.cuda.empty_cache()

    result = {
        "width": width,
        "params": n_params,
        "max_sv_ratio": round(sv_ratio, 1),
        "fim_tier1_tier3": round(fim_ratio, 1),
        "train_time": round(train_time, 1),
        "bf16": bool(_BF16),
        "grad_ckpt": bool(_GRAD_CKPT),
    }
    print(f"  SV={sv_ratio:.1f}x  FIM tier1/tier3={fim_ratio:.1f}x  ({train_time:.1f}s)")

    payload = _load_results()
    # Replace an existing row for this width (re-runs should overwrite).
    payload["results"] = [r for r in payload["results"] if r["width"] != width]
    payload["results"].append(result)
    payload["results"].sort(key=lambda r: r["params"])
    payload["sv_power_law"], payload["fim_power_law"] = _refit_power_law(payload["results"])
    _save_results(payload)
    print(f"  saved → {_RESULTS_PATH}  (total widths: {len(payload['results'])})")
    print(f"  SV power law: N^{payload['sv_power_law']['exponent']}  R²={payload['sv_power_law']['r_squared']}")
    print(f"  FIM power law: N^{payload['fim_power_law']['exponent']}  R²={payload['fim_power_law']['r_squared']}")


if __name__ == "__main__":
    main()
