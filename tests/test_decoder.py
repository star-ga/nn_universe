"""Unit tests for the V2.1 QEC decoder architectures."""
from __future__ import annotations

import pytest
import torch

from decoder import MLPDecoder


@pytest.mark.unit
def test_decoder_shape() -> None:
    n_syn = 25
    n_qub = 50
    net = MLPDecoder(n_syn, n_qub, width=64, hidden_layers=3)
    x = torch.randn(8, n_syn)
    y = net(x)
    assert y.shape == (8, n_qub)


@pytest.mark.unit
def test_decoder_param_count_matches_formula() -> None:
    # For 5 hidden layers of width W, input n_syn, output n_qub:
    # params = n_syn*W + W + 4*(W*W + W) + W*n_qub + n_qub
    n_syn, n_qub, W = 25, 50, 256
    net = MLPDecoder(n_syn, n_qub, width=W, hidden_layers=5)
    n = sum(p.numel() for p in net.parameters())
    expected = n_syn * W + W + 4 * (W * W + W) + W * n_qub + n_qub
    assert n == expected, f"expected {expected} params, got {n}"


@pytest.mark.unit
def test_decoder_gradient_flows() -> None:
    net = MLPDecoder(10, 20, width=32, hidden_layers=3)
    x = torch.randn(4, 10, requires_grad=False)
    y = net(x)
    y.sum().backward()
    # All weight parameters should have non-None gradients.
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name}: no gradient"
