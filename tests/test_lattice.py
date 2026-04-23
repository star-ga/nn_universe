"""Unit tests for the V2.0 lattice-refinement helpers."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lattice_refinement import LatticeEmbeddedNet, _radial_profile, cauchy_rate, make_lattice_coords


@pytest.mark.unit
def test_lattice_coords_spacing() -> None:
    coords = make_lattice_coords((4, 4), a=0.5)
    assert coords.shape == (16, 2)
    diffs = coords[1] - coords[0]
    assert abs(diffs[1].item() - 0.5) < 1e-6  # step along inner axis


@pytest.mark.unit
def test_lattice_net_forward() -> None:
    torch.manual_seed(0)
    net = LatticeEmbeddedNet(shape=(8, 8), r=2, a=1.0, d=2)
    coords = make_lattice_coords((8, 8), a=1.0)
    pts = coords[:10]
    y = net.eval_at(pts, coords)
    assert y.shape == (10,)


@pytest.mark.unit
def test_cauchy_rate_output_shape() -> None:
    lvl0 = {"level": 0, "fim_row_origin_to_distance": [{"r_lo": 0, "r_hi": 1, "mean": 1.0}, {"r_lo": 1, "r_hi": 2, "mean": 0.5}]}
    lvl1 = {"level": 1, "fim_row_origin_to_distance": [{"r_lo": 0, "r_hi": 1, "mean": 0.9}, {"r_lo": 1, "r_hi": 2, "mean": 0.45}]}
    rates = cauchy_rate([lvl0, lvl1])
    assert len(rates) == 1
    assert rates[0]["from_level"] == 0
    assert rates[0]["to_level"] == 1
    assert rates[0]["relative_error"] > 0


@pytest.mark.unit
def test_radial_profile_center_dominates() -> None:
    shape = (9, 9)
    spacing = 1.0
    fim_row = np.zeros(81)
    # Maximum at center site
    fim_row[4 * 9 + 4] = 10.0
    profile = _radial_profile(fim_row, shape, spacing)
    # first bin (r near 0) should have the largest mean
    means = [b["mean"] for b in profile]
    assert means[0] == max(means)
