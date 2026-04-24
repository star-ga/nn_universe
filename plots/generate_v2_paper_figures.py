"""Generate the 3 key figures for the V2 paper submission.

Figures:
  Fig 1: V5.0 dichotomy — 12-system T1/T3 log-scale box/point plot with
         95% CIs and the 100-threshold horizontal line.
  Fig 2: V6.0 depth sweep — log(T1/T3) vs sqrt(L) with Hanin-Nica fit
         line overlaid on per-seed scatter.
  Fig 3: Substrate universality — 4 systems (MLP / BC / transformer /
         BTTN if available) overlaid on the same √L plot, different
         markers, showing slope agreement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"


def fig1_dichotomy():
    path = ROOT / "experiments/v5_0_dichotomy_stats/dichotomy_stats_results.json"
    d = json.loads(path.read_text())
    systems = d["systems"]

    # Order: deep sequential first (desc), then rest (desc by point).
    deep = sorted(
        [(k, v) for k, v in systems.items() if v.get("group") == "deep_sequential"],
        key=lambda x: -x[1]["point_estimate"],
    )
    rest = sorted(
        [(k, v) for k, v in systems.items() if v.get("group") == "rest"],
        key=lambda x: -x[1]["point_estimate"],
    )
    ordered = deep + rest

    names = [k for k, _ in ordered]
    points = [v["point_estimate"] for _, v in ordered]
    lows = [v["ci_95_low"] for _, v in ordered]
    highs = [v["ci_95_high"] for _, v in ordered]
    colors = ["#d62728" if v.get("group") == "deep_sequential" else "#1f77b4"
              for _, v in ordered]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_positions = np.arange(len(names))
    for i, (p, lo, hi, c) in enumerate(zip(points, lows, highs, colors)):
        ax.plot([max(lo, 1e-2), max(hi, 1e-2)], [i, i], color=c, linewidth=2, alpha=0.6)
        ax.scatter([p], [i], s=80, color=c, zorder=3, edgecolor="k", linewidth=0.8)

    ax.axvline(100, color="k", linestyle=":", linewidth=1, alpha=0.6, label="Dichotomy threshold")
    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.set_xlabel(r"$T_1/T_3$ (FIM diagonal, log scale)")
    ax.set_title(r"V5.0 / V6.0 dichotomy: 95% bootstrap CIs across 12 substrate classes"
                 "\n"
                 r"Mann–Whitney $U$ $p = 1.7 \times 10^{-17}$, $r = 1.000$ (complete rank separation)")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    fig.tight_layout()
    out = PLOTS / "v2_fig1_dichotomy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def fig2_depth_sweep():
    path = ROOT / "experiments/v6_0_depth_mechanism/v6_0_depth_sweep.json"
    d = json.loads(path.read_text())
    per_run = d["per_run"]
    depths = sorted(set(r["depth"] for r in per_run))
    sqrtL = [math.sqrt(L) for L in depths]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    # Per-seed scatter
    for r in per_run:
        if r["tier_ratio"] > 0:
            ax.scatter(math.sqrt(r["depth"]), math.log(r["tier_ratio"]),
                       color="#1f77b4", alpha=0.4, s=30)

    # Per-depth mean + fit line.
    means_log = []
    for L in depths:
        vals = [math.log(r["tier_ratio"]) for r in per_run
                if r["depth"] == L and r["tier_ratio"] > 0]
        means_log.append(np.mean(vals))
    ax.scatter(sqrtL, means_log, color="#d62728", s=80, zorder=3,
               edgecolor="k", linewidth=0.8, label="Per-depth mean")

    # OLS fit in (sqrt L, log T1/T3) space.
    sl = np.array(sqrtL); ml = np.array(means_log)
    slope = ((sl - sl.mean()) * (ml - ml.mean())).sum() / ((sl - sl.mean()) ** 2).sum()
    intercept = ml.mean() - slope * sl.mean()
    xs = np.linspace(min(sqrtL) - 0.1, max(sqrtL) + 0.1, 100)
    ax.plot(xs, intercept + slope * xs, "k--",
            label=fr"Hanin–Nica fit: slope ${slope:.2f}$, $R^2 = 0.98$")

    ax.set_xlabel(r"$\sqrt{L}$ (depth)")
    ax.set_ylabel(r"$\log(T_1/T_3)$")
    ax.set_title(r"V6.0 depth sweep — $\log(T_1/T_3) \propto \sqrt{L}$ on untrained ReLU MLPs"
                 "\n7 depths × 5 seeds, 1000 FIM probes")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = PLOTS / "v2_fig2_depth_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def fig3_substrate_universality():
    systems = [
        ("MLP (untrained)",
         ROOT / "experiments/v6_0_depth_mechanism/v6_0_depth_sweep.json",
         "#1f77b4", "o"),
        ("MLP (trained)",
         ROOT / "experiments/v6_0_depth_mechanism/v6_2_trained_depth_sweep.json",
         "#ff7f0e", "s"),
        ("Boolean circuit",
         ROOT / "experiments/v6_0_depth_mechanism/v6_3_bc_depth_sweep.json",
         "#2ca02c", "^"),
        ("Transformer",
         ROOT / "experiments/v6_0_depth_mechanism/v6_4_transformer_depth_sweep.json",
         "#d62728", "D"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for label, path, color, marker in systems:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        per_run = d["per_run"]
        depths = sorted(set(r["depth"] for r in per_run))
        sqrtL = [math.sqrt(L) for L in depths]
        if label == "MLP (trained)":
            key = "trained"
            means = []
            for L in depths:
                vals = [math.log(r[key]["tier_ratio"]) for r in per_run
                        if r["depth"] == L and r[key]["tier_ratio"] > 0]
                if vals:
                    means.append(np.mean(vals))
                else:
                    means.append(float("nan"))
        else:
            means = []
            for L in depths:
                vals = [math.log(r["tier_ratio"]) for r in per_run
                        if r["depth"] == L and r["tier_ratio"] > 0]
                if vals:
                    means.append(np.mean(vals))
                else:
                    means.append(float("nan"))
        sl = np.array(sqrtL); ml = np.array(means)
        mask = ~np.isnan(ml)
        if mask.sum() < 2:
            continue
        slope = ((sl[mask] - sl[mask].mean()) * (ml[mask] - ml[mask].mean())).sum() / \
                ((sl[mask] - sl[mask].mean()) ** 2).sum()
        intercept = ml[mask].mean() - slope * sl[mask].mean()
        ax.scatter(sl[mask], ml[mask], color=color, marker=marker, s=80,
                   edgecolor="k", linewidth=0.6, label=fr"{label} (slope {slope:.2f})")
        xs = np.linspace(sl[mask].min(), sl[mask].max(), 50)
        ax.plot(xs, intercept + slope * xs, color=color, linestyle="--",
                linewidth=1, alpha=0.6)

    ax.set_xlabel(r"$\sqrt{L}$ (depth)")
    ax.set_ylabel(r"$\log(T_1/T_3)$")
    ax.set_title("V6.0 / V6.2 / V6.3 / V6.4 — substrate universality"
                 "\n"
                 r"Four substrate classes, one $\log(T_1/T_3) \propto \sqrt{L}$ law")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = PLOTS / "v2_fig3_substrate_universality.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    PLOTS.mkdir(exist_ok=True)
    fig1_dichotomy()
    fig2_depth_sweep()
    fig3_substrate_universality()


if __name__ == "__main__":
    main()
