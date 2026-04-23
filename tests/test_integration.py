"""End-to-end integration tests for all V1.0-V3.1 scripts.

These use tiny hyperparameters so they run in seconds; they verify that
the *pipelines* work without correctness-testing the physics.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], env: dict | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=str(REPO))


@pytest.mark.integration
def test_lattice_analytic_pipeline(tmp_path: Path) -> None:
    out = tmp_path / "lattice_analytic.json"
    r = _run(
        [sys.executable, "experiments/v2_0_lattice/lattice_analytic.py", "--d", "2", "--levels", "3", "--eval-density", "33", "--out", str(out)]
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(out.read_text())
    assert len(payload["levels"]) == 3
    errs = [l["abs_err"] for l in payload["levels"]]
    assert errs[-1] <= errs[0]  # Cauchy decrease in the aggregate


@pytest.mark.integration
def test_alpha_mock_pipeline(tmp_path: Path) -> None:
    out = tmp_path / "mock.json"
    r = _run(
        [sys.executable, "experiments/v3_1_alpha/mock_pipeline.py", "--n", "500", "--seeds", "20", "--out", str(out)]
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(out.read_text())
    assert "power" in payload
    assert "roc" in payload


@pytest.mark.integration
def test_scaling_fit_schema() -> None:
    # Read scaling_results.json (must exist from V1.0) and verify schema.
    payload = json.loads((REPO / "scaling_results.json").read_text())
    assert "results" in payload
    assert "sv_power_law" in payload
    assert "fim_power_law" in payload
    for r in payload["results"]:
        assert {"width", "params", "max_sv_ratio", "fim_tier1_tier3"} <= set(r.keys())
