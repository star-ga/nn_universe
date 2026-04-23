"""Generate summary plots for all experiment results.

Reads the JSON outputs and produces markdown tables + (optional) matplotlib
figures. Run after all experiments complete.

Usage:
    python3 experiments/visualize.py            # tables only
    python3 experiments/visualize.py --plots    # emits plots/*.png  (requires matplotlib)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict | None:
    p = REPO / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def v1_0_table() -> str:
    d = _load("toy_experiment_results.json") or {}
    if not d:
        return "_(V1.0 toy results not present)_"
    sv = d.get("sv_stats", [])
    rows = ["| Layer | Shape | Top 3 SV | SV Ratio | SV Std |", "|-------|-------|----------|----------|--------|"]
    for s in sv:
        rows.append(f"| {s['layer']} | {s['shape']} | {s['top3_sv']} | {s['sv_ratio']}x | {s['sv_std']} |")
    return "### V1.0 Per-Layer SV (seed 42)\n\n" + "\n".join(rows)


def v1_2_scaling_table() -> str:
    d = _load("scaling_results.json") or {}
    rows = ["| Width | Params | SV Ratio | FIM T1/T3 |", "|-------|--------|----------|-----------|"]
    for r in d.get("results", []):
        rows.append(f"| {r['width']} | {_fmt_int(r['params'])} | {r['max_sv_ratio']}x | {r['fim_tier1_tier3']}x |")
    pl = d.get("sv_power_law", {})
    fpl = d.get("fim_power_law", {})
    foot = (
        f"\n**SV power law:** $N^{{{pl.get('exponent', '?')}}}$, $R^2 = {pl.get('r_squared', '?')}$. "
        f"**FIM power law:** $N^{{{fpl.get('exponent', '?')}}}$, $R^2 = {fpl.get('r_squared', '?')}$."
    )
    return "### V1.2 Width Sweep\n\n" + "\n".join(rows) + foot


def v2_1_qec_table() -> str:
    d = _load("experiments/v2_1_qec/v2_1_sweep_results.json") or {}
    if not d.get("results"):
        return "_(V2.1 QEC sweep results not present)_"
    rows = ["| Width | Params | SV Ratio | FIM T1/T3 | Final BCE loss |", "|-------|--------|----------|-----------|----------------|"]
    for r in d["results"]:
        rows.append(f"| {r['width']} | {_fmt_int(r['n_params'])} | {r['max_sv_ratio']}x | {r['fim_tier1_tier3']}x | {r['final_loss']} |")
    pl = d.get("sv_power_law", {})
    fpl = d.get("fim_power_law", {})
    foot = (
        f"\n**QEC SV power law:** $N^{{{pl.get('exponent', '?')}}}$, $R^2 = {pl.get('r_squared', '?')}$. "
        f"**QEC FIM power law:** $N^{{{fpl.get('exponent', '?')}}}$, $R^2 = {fpl.get('r_squared', '?')}$."
    )
    return "### V2.1 QEC Decoder Width Sweep\n\n" + "\n".join(rows) + foot


def v2_0_lattice_table() -> str:
    d = _load("experiments/v2_0_lattice/lattice_analytic_results.json") or {}
    if not d.get("levels"):
        return "_(V2.0 lattice analytic results not present)_"
    rows = ["| Level | Spacing $a$ | u^T G u | |err| | rel_err |", "|-------|-------------|---------|------|---------|"]
    for r in d["levels"]:
        rows.append(f"| {r['level']} | {r['spacing']:.4f} | {r['bilinear_form']:.4e} | {r['abs_err']:.2e} | {r['rel_err']:.2e} |")
    rate = d.get("observed_convergence_rate", "?")
    foot = f"\n**Observed convergence rate:** $|err| \\sim a^{{{rate:.3f}}}$ (theoretical: $O(a^2)$)."
    return "### V2.0 Lattice Cauchy Convergence\n\n" + "\n".join(rows) + foot


def v3_1_power_table() -> str:
    d = _load("experiments/v3_1_alpha/mock_results.json") or {}
    if not d.get("power"):
        return "_(V3.1 mock results not present — run `python3 experiments/v3_1_alpha/mock_pipeline.py`)_"
    rows = ["| Threshold | α | FPR | Power |", "|-----------|---|-----|-------|"]
    for name, info in d["power"]["thresholds"].items():
        rows.append(f"| {name} | {info['alpha']:.2e} | {info['false_positive_rate']:.3f} | {info['power']:.3f} |")
    return "### V3.1 Mock Power Analysis\n\n" + "\n".join(rows)


def depth_sweep_table() -> str:
    d = _load("experiments/v1_2_scaling/depth_sweep_results.json") or {}
    if not d.get("results"):
        return "_(V1.2 depth sweep not yet run)_"
    rows = ["| Depth | Params | SV Ratio | FIM T1/T3 | Train time |", "|-------|--------|----------|-----------|------------|"]
    for r in d["results"]:
        fim = r["fim_tier1_tier3"]
        fim_display = f"{fim:.1e}" if fim >= 1e5 else f"{fim:.1f}x"
        rows.append(f"| {r['depth']} | {_fmt_int(r['params'])} | {r['max_sv_ratio']:.1f}x | {fim_display} | {r['train_time']}s |")
    note = (
        "\n> Note: FIM T1/T3 ratios for depth ≥ 8 are numerically unreliable because "
        "Tier-3 FIM values underflow float32 at these depths. The monotone upward trend "
        "is physical; the absolute magnitudes at depth ≥ 8 are lower bounds only."
    )
    return "### V1.2 Depth Sweep (fixed width 256)\n\n" + "\n".join(rows) + note


def seed_robustness_table() -> str:
    d = _load("experiments/v1_2_scaling/seed_robustness_results.json") or {}
    if not d.get("results"):
        return "_(V1.2 seed robustness not yet run)_"
    rows = ["| Seed | SV Ratio | FIM T1/T3 |", "|------|----------|-----------|"]
    for r in d["results"]:
        rows.append(f"| {r['seed']} | {r['max_sv_ratio']}x | {r['fim_tier1_tier3']}x |")
    foot = (
        f"\n**Mean SV:** {d['sv_mean']:.1f} ± {d['sv_std']:.1f} (CV {d['sv_cv']*100:.1f}%). "
        f"**Mean FIM T1/T3:** {d['fim_mean']:.1f} ± {d['fim_std']:.1f} (CV {d['fim_cv']*100:.1f}%)."
    )
    return "### V1.2 Seed Robustness (width=256)\n\n" + "\n".join(rows) + foot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--out", type=str, default=str(REPO / "docs/results_summary.md"))
    args = ap.parse_args()

    sections = [
        "# nn_universe — Experimental Results Summary\n",
        "_Auto-generated from JSON outputs in the repository. Do not edit manually._\n",
        v1_0_table(),
        v1_2_scaling_table(),
        seed_robustness_table(),
        depth_sweep_table(),
        v2_0_lattice_table(),
        v2_1_qec_table(),
        v3_1_power_table(),
    ]
    text = "\n\n".join(sections)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(f"Wrote {args.out}")

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plots.")
            return 0
        plots_dir = REPO / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Plot 1: SV scaling comparison (cosmology vs QEC)
        cos = _load("scaling_results.json")
        qec = _load("experiments/v2_1_qec/v2_1_sweep_results.json")
        if cos and qec and cos.get("results") and qec.get("results"):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.loglog([r["params"] for r in cos["results"]], [r["max_sv_ratio"] for r in cos["results"]], "o-", label=f"Cosmology (V1.2): N^{cos['sv_power_law']['exponent']}")
            ax.loglog([r["n_params"] for r in qec["results"]], [r["max_sv_ratio"] for r in qec["results"]], "s-", label=f"QEC decoder (V2.1): N^{qec['sv_power_law']['exponent']}")
            ax.set_xlabel("Parameter count N")
            ax.set_ylabel("Max SV ratio")
            ax.set_title("SV scaling: cosmology vs QEC decoder")
            ax.legend()
            ax.grid(True, which="both", alpha=0.3)
            fig.savefig(plots_dir / "sv_scaling_comparison.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {plots_dir}/sv_scaling_comparison.png")

        # Plot 2: V2.0 Cauchy convergence
        lat = _load("experiments/v2_0_lattice/lattice_analytic_results.json")
        if lat and lat.get("levels"):
            fig, ax = plt.subplots(figsize=(7, 5))
            xs = [l["spacing"] for l in lat["levels"]]
            ys = [l["abs_err"] for l in lat["levels"]]
            ax.loglog(xs, ys, "o-", label="|error|")
            ax.set_xlabel("Lattice spacing a")
            ax.set_ylabel("|u^T G u — continuum|")
            ax.set_title(f"V2.0 Cauchy convergence (rate: a^{lat.get('observed_convergence_rate', 0):.2f})")
            ax.grid(True, which="both", alpha=0.3)
            fig.savefig(plots_dir / "v2_0_cauchy_convergence.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {plots_dir}/v2_0_cauchy_convergence.png")

        # Plot 3: V3.1 ROC
        mock = _load("experiments/v3_1_alpha/mock_results.json")
        if mock and mock.get("roc"):
            fig, ax = plt.subplots(figsize=(6, 6))
            fpr = [pt["fpr"] for pt in mock["roc"]]
            tpr = [pt["tpr"] for pt in mock["roc"]]
            ax.plot(fpr, tpr, "o-")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"V3.1 mock pipeline ROC (N={mock['config']['n_sightlines']})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / "v3_1_roc.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {plots_dir}/v3_1_roc.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
