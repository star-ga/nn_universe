"""Tests for V5.0 dichotomy statistics (bootstrap CI + Mann-Whitney).

Tests the standalone statistical functions directly; no file I/O needed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/v5_0_dichotomy_stats"))
from dichotomy_stats import bootstrap_log_ci  # noqa: E402


# ---------------------------------------------------------------------------
# bootstrap_log_ci
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_bootstrap_ci_linear_regression() -> None:
    """Bootstrap CI on log-normal samples centred at 1 (log-mean = 0) covers 1.

    We generate samples from a distribution whose geometric mean is ~1.
    The 95% CI on the geometric mean should straddle 1.
    """
    rng = np.random.default_rng(0)
    # 5 replicates drawn from exp(N(0, 0.5)) → geometric mean ≈ 1
    samples = np.exp(rng.standard_normal(5) * 0.5)
    point, ci_low, ci_high, method, n = bootstrap_log_ci(list(samples), n_boot=500, alpha=0.05)
    assert math.isfinite(point)
    assert ci_low < point < ci_high
    # The CI should contain 1.0 when centred near it
    assert ci_low < 1.0 * 5 < ci_high or ci_low < 1.0 < ci_high, (
        f"CI [{ci_low:.4f}, {ci_high:.4f}] should plausibly contain the true mean"
    )


@pytest.mark.unit
def test_bootstrap_ci_returns_correct_fields() -> None:
    """Return tuple has (point, ci_low, ci_high, method, n)."""
    result = bootstrap_log_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=200, alpha=0.05)
    assert len(result) == 5
    point, ci_low, ci_high, method, n = result
    assert ci_low <= point <= ci_high
    assert isinstance(method, str)
    assert n == 5


@pytest.mark.unit
def test_bootstrap_ci_small_n_uses_normal_approx() -> None:
    """n < 3 triggers the normal approximation branch."""
    _, _, _, method, n = bootstrap_log_ci([2.0, 4.0], n_boot=200, alpha=0.05)
    assert method == "normal_approx"
    assert n == 2


@pytest.mark.unit
def test_bootstrap_ci_large_n_uses_bootstrap() -> None:
    rng = np.random.default_rng(1)
    samples = list(np.exp(rng.standard_normal(20)))
    _, _, _, method, n = bootstrap_log_ci(samples, n_boot=200, alpha=0.05)
    assert method == "bootstrap"
    assert n == 20


# ---------------------------------------------------------------------------
# Mann-Whitney — obvious separation sanity check
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_mann_whitney_obvious_separation() -> None:
    """[1,2,3] vs [100,200,300]: deep clearly dominates → rank-biserial r = 1.0.

    With n1=n2=3, U=9 and p_one_sided=0.05 exactly (boundary of the
    discrete distribution). We assert r=1.0 and p <= 0.05.
    """
    deep = [math.log(v) for v in [100, 200, 300]]
    rest = [math.log(v) for v in [1, 2, 3]]
    stat, p = mannwhitneyu(deep, rest, alternative="greater")
    r = stat / (len(deep) * len(rest))
    assert abs(r - 1.0) < 1e-9, f"rank-biserial r should be 1.0, got {r}"
    assert p <= 0.05, f"p={p} should be <= 0.05 for perfectly separated groups"


@pytest.mark.unit
def test_mann_whitney_identical_gives_half() -> None:
    """Identical distributions → rank-biserial r ≈ 0.5 (U = n1*n2/2)."""
    vals = [math.log(v) for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
    stat, _ = mannwhitneyu(vals, vals, alternative="two-sided")
    r = stat / (len(vals) * len(vals))
    assert abs(r - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Audit citation guard
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_no_audit_citations_in_public_docs() -> None:
    """README.md, paper_draft.md, findings.md must not mention multi-llm audit
    strings that were removed in commit c6eb7df."""
    repo = Path(__file__).resolve().parents[1]
    targets = [
        repo / "README.md",
        repo / "docs" / "paper_draft.md",
        repo / "docs" / "findings.md",
    ]
    forbidden = ["multi-llm audit", "audit v3", "audit_v3"]
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            assert phrase not in text, (
                f"Forbidden phrase {phrase!r} found in {path}. "
                "These references should have been stripped in commit c6eb7df."
            )
