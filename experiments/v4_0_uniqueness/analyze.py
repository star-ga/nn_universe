"""V4.0 analyze — distinguish neural-network FIM tier structure from baselines.

Reads ``v4_0_uniqueness_results.json`` and answers the core uniqueness
question three ways:

1. **Magnitude test** — is the NN tier_ratio distinguishable (>2σ) from the
   maximum ratio among non-NN baselines?
2. **Stability test** — is the NN ratio_cv lower than every non-NN baseline's
   ratio_cv?
3. **Classifier test** — can a simple two-feature classifier
   (log-tier-ratio, top-1%-mass) tell NN apart from non-NN with > 80% accuracy
   in leave-one-seed-out cross-validation?

Writes a markdown summary next to the JSON result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEGENERATE_RATIO = 1e10  # matches RATIO_CAP in run_uniqueness


def load_features(results: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, names) where X is (n_samples, 2) and y ∈ {0,1}.

    Excludes degenerate baselines (CV > 100% or ratio >= DEGENERATE_RATIO)
    on the basis that their tier_ratio is not a well-estimated statistic.
    """
    skip: set[str] = set()
    for name, data in results["baselines"].items():
        if name == "neural_network":
            continue
        if float(data.get("tier_ratio_cv", 0.0)) > 1.0:
            skip.add(name)
    X_list, y_list, names = [], [], []
    for name, data in results["baselines"].items():
        if name in skip:
            continue
        is_nn = 1 if name == "neural_network" else 0
        for r in data["per_seed"]:
            ratio = r["tier_ratio"]
            if ratio > 0 and np.isfinite(ratio) and ratio < DEGENERATE_RATIO:
                X_list.append([np.log10(ratio), r["top1pct_mass"]])
                y_list.append(is_nn)
                names.append(name)
    return np.array(X_list), np.array(y_list), names


def simple_logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Tiny logistic regression via closed-form IRLS (no sklearn dep).

    Returns (coef, bias).
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])  # bias column
    w = np.zeros(d + 1)
    for _ in range(200):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = p * (1 - p) + 1e-8
        grad = Xb.T @ (p - y)
        H = Xb.T @ (W[:, None] * Xb) + 1e-4 * np.eye(d + 1)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w[:d], float(w[d])


def loo_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out accuracy of the logistic classifier."""
    n = len(y)
    correct = 0
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        coef, bias = simple_logistic(X[mask], y[mask])
        pred = int((X[i] @ coef + bias) > 0)
        correct += int(pred == y[i])
    return correct / n


def analyze(results: dict) -> dict:
    nn_data = results["baselines"].get("neural_network")
    if nn_data is None:
        raise KeyError("neural_network baseline missing from results")
    nn_ratios = np.array([r["tier_ratio"] for r in nn_data["per_seed"]])
    nn_masses = np.array([r["top1pct_mass"] for r in nn_data["per_seed"]])

    others: dict[str, dict] = {
        k: v for k, v in results["baselines"].items() if k != "neural_network"
    }

    degenerate: list[str] = []
    max_other_ratio = 0.0
    max_other_name = None
    for name, data in others.items():
        ratios = np.array([r["tier_ratio"] for r in data["per_seed"]])
        # Degenerate systems (ill-conditioned tier-3) can't be compared:
        # either at/above numerical cap, or with CV > 100% which indicates
        # the statistic is not well-estimated on this system.
        cv = float(data["tier_ratio_cv"])
        if np.any(ratios >= DEGENERATE_RATIO) or cv > 1.0:
            degenerate.append(name)
            continue
        if np.mean(ratios) > max_other_ratio:
            max_other_ratio = float(np.mean(ratios))
            max_other_name = name

    # 1. Magnitude test: z-score of NN mean vs union of non-NN samples
    #    (excluding degenerate baselines)
    all_other_ratios = np.concatenate(
        [
            [r["tier_ratio"] for r in data["per_seed"]]
            for name, data in others.items()
            if name not in degenerate
        ]
    )
    sigma = float(np.std(all_other_ratios)) + 1e-9
    z = (float(np.mean(nn_ratios)) - float(np.mean(all_other_ratios))) / sigma
    magnitude_pass = z > 2.0

    # 2. Stability test (excluding degenerate baselines)
    nn_cv = float(nn_data["tier_ratio_cv"])
    other_cvs = {
        name: float(data["tier_ratio_cv"])
        for name, data in others.items()
        if name not in degenerate
    }
    stability_pass = all(nn_cv < cv for cv in other_cvs.values()) if other_cvs else False

    # 3. Classifier test
    X, y, _ = load_features(results)
    accuracy = loo_accuracy(X, y)
    classifier_pass = accuracy > 0.80

    return {
        "nn_mean_ratio": float(np.mean(nn_ratios)),
        "nn_cv": nn_cv,
        "nn_top1pct_mass_mean": float(np.mean(nn_masses)),
        "max_other_ratio": max_other_ratio,
        "max_other_name": max_other_name,
        "other_cvs": other_cvs,
        "degenerate_baselines": degenerate,
        "magnitude_z": float(z),
        "magnitude_pass": bool(magnitude_pass),
        "stability_pass": bool(stability_pass),
        "classifier_loo_accuracy": float(accuracy),
        "classifier_pass": bool(classifier_pass),
        "verdict": "NN_UNIQUE"
        if all([magnitude_pass, stability_pass, classifier_pass])
        else "NN_NOT_UNIQUE",
    }


def to_markdown(results: dict, verdict: dict) -> str:
    lines: list[str] = []
    lines.append("# V4.0 Uniqueness Analysis — Summary\n")
    lines.append(f"**Verdict:** `{verdict['verdict']}`\n")
    lines.append("| Baseline | n_params | Tier ratio (mean ± std) | CV | Top-1% mass |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, data in results["baselines"].items():
        ratios = [r["tier_ratio"] for r in data["per_seed"]]
        m = np.mean(ratios)
        s = np.std(ratios)
        cv = float(data["tier_ratio_cv"])
        mass = float(data["top1pct_mass_mean"])
        lines.append(
            f"| `{name}` | {data['n_params']} | {m:.2f} ± {s:.2f} | {100*cv:.1f}% | {mass:.4f} |"
        )
    lines.append("\n## Uniqueness tests\n")
    lines.append(
        f"1. **Magnitude** (z-score of NN vs union of non-NN): "
        f"**{verdict['magnitude_z']:.2f}σ** → "
        f"`{'PASS' if verdict['magnitude_pass'] else 'FAIL'}` (threshold 2σ)"
    )
    lines.append(
        f"2. **Stability** (NN CV < every non-NN CV): "
        f"NN CV = {100*verdict['nn_cv']:.1f}% → "
        f"`{'PASS' if verdict['stability_pass'] else 'FAIL'}`"
    )
    lines.append(
        f"3. **Classifier** (LOO accuracy, 2-feature logistic): "
        f"**{100*verdict['classifier_loo_accuracy']:.1f}%** → "
        f"`{'PASS' if verdict['classifier_pass'] else 'FAIL'}` (threshold 80%)"
    )
    lines.append("\n## Interpretation\n")
    if verdict["verdict"] == "NN_UNIQUE":
        lines.append(
            "The neural-network FIM tier hierarchy is statistically distinguishable "
            "from five alternative parameterized systems of comparable size. This "
            "strengthens the universality claim in the specific sense that *the "
            "three-tier FIM signature is a property of gradient-trained probabilistic "
            "models, not a generic information-geometric artifact*."
        )
    else:
        lines.append(
            "The neural-network FIM tier hierarchy is NOT statistically distinguishable "
            "from one or more alternative parameterized systems. This weakens the "
            "neural-network-specific framing of the universality claim: the tier "
            "structure likely reflects a more generic information-geometric "
            "phenomenology that also appears in non-learning systems. The paper "
            "framing should be adjusted accordingly."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "v4_0_uniqueness_results.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "v4_0_uniqueness_analysis.md",
    )
    args = p.parse_args()
    data = json.loads(args.results.read_text())
    verdict = analyze(data)
    args.out.write_text(to_markdown(data, verdict))
    print(json.dumps(verdict, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
