"""Shared pytest configuration for nn_universe tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("", "experiments/v1_2_scaling", "experiments/v2_0_lattice", "experiments/v2_1_qec", "experiments/v3_1_alpha"):
    p = str(REPO / sub) if sub else str(REPO)
    if p not in sys.path:
        sys.path.insert(0, p)
