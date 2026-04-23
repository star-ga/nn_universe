"""Neural decoder architectures for V2.1.

We implement two families of decoder:

1. `MLPDecoder` — fully-connected 5-layer, 256-neuron, ReLU. This matches the
   V1.0 cosmology experiment architecture exactly. Used to answer: does the
   spectral hierarchy emerge under a genuinely different task (syndrome
   decoding) with the same architecture? If yes, the hierarchy is
   architectural universality, not self-prediction-specific.

2. `ScaledMLPDecoder` — same 5-layer topology but variable width. Used for
   the scaling-law sweep to check whether SV ~ N^alpha holds in QEC decoders.

Both decoders operate on flattened syndrome input and produce a logits vector
over qubit-flip correction bits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """5-layer 256-neuron decoder — mirrors V1.0 toy cosmology architecture."""

    def __init__(self, n_syndromes: int, n_qubits: int, width: int = 256, hidden_layers: int = 5):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_syndromes, width), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, n_qubits))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScaledMLPDecoder(MLPDecoder):
    """Thin alias for clarity in scaling sweeps."""

    pass
