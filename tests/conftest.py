"""Shared pytest configuration for nn_universe tests.

``experiments/v2_1_qec`` and ``experiments/v4_0_uniqueness`` both define a
top-level ``analyze.py``; we resolve the collision by deciding per test
module which one to point at, and rewriting ``sys.path`` at collection
time via the ``pytest_collectstart`` hook so the right ``analyze``
module gets imported when the test file itself is loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_PATHS = (
    "",
    "experiments/v1_2_scaling",
    "experiments/v2_0_lattice",
    "experiments/v3_1_alpha",
)
for sub in BASE_PATHS:
    p = str(REPO / sub) if sub else str(REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


_FILE_TO_EXPERIMENT = {
    "test_analyze.py": "experiments/v2_1_qec",
    "test_decoder.py": "experiments/v2_1_qec",
    "test_toric_code.py": "experiments/v2_1_qec",
    "test_v4_uniqueness.py": "experiments/v4_0_uniqueness",
}

_SHARED_MODULE_NAMES = ("analyze", "decoder", "toric_code", "baselines", "run_uniqueness")


def pytest_collectstart(collector):
    """Before pytest imports a test module, point sys.path at the right experiment dir."""
    name = getattr(collector, "name", None)
    extra = _FILE_TO_EXPERIMENT.get(name)
    if extra is None:
        return
    abs_extra = str(REPO / extra)
    # Remove any competing experiment dirs from sys.path, then prepend ours.
    competitors = [
        str(REPO / "experiments/v2_1_qec"),
        str(REPO / "experiments/v4_0_uniqueness"),
    ]
    sys.path[:] = [p for p in sys.path if p not in competitors]
    sys.path.insert(0, abs_extra)
    # Clear any already-imported competing modules so the next `from analyze
    # import ...` inside the test file resolves against our injected path.
    for mod in _SHARED_MODULE_NAMES:
        sys.modules.pop(mod, None)
