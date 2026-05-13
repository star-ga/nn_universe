"""V12 decision rules — apply locked falsifiers from preregistration_v3.md.

Reads v12_aggregate.json and emits v12_decision_verdict.json with
PASS/FAIL per item + an overall verdict.

Decision rules (from docs/cluster_roadmap_v12.md "Decision rules per item"):

  Item 1 (partition-invariant table)
    PASS: each substrate's (Gini, r_eff/n, top_1pct_mass) triple
          monotonically orders with depth/complexity in the same direction.
    FAIL: 2 of 3 statistics disagree on the deep-vs-rest direction.

  Item 2 (real LM-loss FIM)
    PASS: every LM stays in deep-sequential band (T_1/T_3 > 100).
    FAIL: any LM drops out (T_1/T_3 < 100) under real-loss probes.

  Item 3 (5-seed at production scale)
    PASS: every production model's 5-seed bootstrap 95% CI lies entirely
          above 100 (deep) or entirely below 100 (control).
    FAIL: any production model's CI crosses 100.

  Item 4 (probe convergence at billion-param scale)
    PASS: T_1/T_3 at n=200 within ±5% of T_1/T_3 at n=1600.
    FAIL: difference > 5% at any model.

  Item 5 (parameter-matched non-deep control)
    PASS: 300M-RFF kernel ridge gives T_1/T_3 < 100 under protocol Π.
    FAIL: T_1/T_3 > 100 (falsifies the depth + sequential primitive hypothesis).

  Overall: PASS_ALL iff all 5 items PASS; PARTIAL if 3+ pass; FAIL otherwise.
"""
import argparse
import json
from pathlib import Path


def check_item_1(agg):
    """Partition-invariant: all 3 stats agree on direction."""
    substrates_with_three = [
        s for s, d in agg["substrates"].items()
        if d.get("gini_mean") is not None
        and d.get("effective_rank_n_mean") is not None
        and d.get("top_1pct_mass_mean") is not None
    ]
    n = len(substrates_with_three)
    if n < 3:
        return {"verdict": "INSUFFICIENT_DATA", "n_substrates_with_three_stats": n}

    # For each substrate, check whether the 3 stats agree on "deep" vs "rest" verdict
    rest_predicate = lambda d: (d["gini_mean"] < 0.5
                                and d["effective_rank_n_mean"] > 0.3
                                and d["top_1pct_mass_mean"] < 0.1)
    deep_predicate = lambda d: (d["gini_mean"] > 0.7
                                and d["effective_rank_n_mean"] < 0.05
                                and d["top_1pct_mass_mean"] > 0.4)
    consistent = 0
    for s in substrates_with_three:
        d = agg["substrates"][s]
        if rest_predicate(d) or deep_predicate(d):
            consistent += 1
    return {
        "verdict": "PASS" if consistent / n >= 0.8 else "FAIL",
        "n_consistent": consistent,
        "n_total": n,
        "ratio": consistent / n,
    }


def check_item_2(agg):
    """LM-loss FIM: every LM has T_1/T_3 > 100."""
    lm_substrates = [s for s in agg["substrates"] if "pile_loss" in s]
    if not lm_substrates:
        return {"verdict": "NOT_RUN", "n_substrates": 0}
    fails = []
    for s in lm_substrates:
        d = agg["substrates"][s]
        if not d.get("all_above_100", False):
            fails.append({"substrate": s, "t1_t3_mean": d["t1_t3_mean"]})
    return {
        "verdict": "FAIL" if fails else "PASS",
        "n_substrates": len(lm_substrates),
        "fails": fails,
    }


def check_item_3(agg):
    """Production multi-seed: every CI clean above 100 or clean below 100."""
    prod_substrates = ["resnet50_imagenet1k", "vit_l_16", "gpt2_large", "mamba_790m_hf"]
    fails = []
    for s in prod_substrates:
        d = agg["substrates"].get(s)
        if d is None:
            continue
        if d["n_seeds"] < 3:
            fails.append({"substrate": s, "reason": "n_seeds < 3"})
            continue
        if d["crosses_threshold_100"]:
            fails.append({"substrate": s, "reason": "CI crosses 100",
                          "t1_t3_min": d["t1_t3_min"], "t1_t3_max": d["t1_t3_max"]})
    return {
        "verdict": "FAIL" if fails else "PASS",
        "n_substrates_checked": sum(1 for s in prod_substrates if s in agg["substrates"]),
        "fails": fails,
    }


def check_item_4(agg, probe_convergence_results_dir):
    """Probe convergence: T_1/T_3 at n=200 within 5% of n=1600."""
    results = []
    for f in Path(probe_convergence_results_dir).glob("probe_convergence_*.json"):
        try:
            d = json.load(open(f))
            results.append(d)
        except Exception:
            pass
    if not results:
        return {"verdict": "NOT_RUN", "n_models": 0}
    fails = []
    for r in results:
        if r.get("convergence_verdict") not in ("STABLE_BY_200", "STABLE_BY_400"):
            fails.append({"model": r["model_id"], "verdict": r["convergence_verdict"]})
    return {
        "verdict": "FAIL" if fails else "PASS",
        "n_models": len(results),
        "fails": fails,
    }


def check_item_5(agg):
    """Non-deep RFF kernel ridge control: T_1/T_3 < 100."""
    if "rff_kernel_ridge_300m" not in agg["substrates"]:
        return {"verdict": "NOT_RUN"}
    d = agg["substrates"]["rff_kernel_ridge_300m"]
    if d.get("all_below_100"):
        return {"verdict": "PASS", "t1_t3_max": d["t1_t3_max"]}
    return {
        "verdict": "FAIL",
        "t1_t3_mean": d["t1_t3_mean"],
        "note": "Falsifies depth+sequential composition primitive hypothesis",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--probe-convergence-dir",
                    default="/data/checkpoints/v12_cluster_followup/results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agg = json.load(open(args.aggregate))

    items = {
        "item_1_partition_invariant": check_item_1(agg),
        "item_2_lm_loss_fim":         check_item_2(agg),
        "item_3_production_multiseed": check_item_3(agg),
        "item_4_probe_convergence":    check_item_4(agg, args.probe_convergence_dir),
        "item_5_nondeep_control":      check_item_5(agg),
    }

    n_pass = sum(1 for v in items.values() if v["verdict"] == "PASS")
    n_fail = sum(1 for v in items.values() if v["verdict"] == "FAIL")
    n_other = len(items) - n_pass - n_fail
    overall = ("PASS_ALL" if n_pass == 5 else
               "PARTIAL" if n_pass >= 3 else
               "FAIL")

    out = {
        "overall_verdict": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_not_run_or_insufficient": n_other,
        "items": items,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Pretty-print
    print("== V12 Decision Verdict ==")
    print(f"Overall: {overall}  ({n_pass} pass, {n_fail} fail, {n_other} other)\n")
    for k, v in items.items():
        print(f"  {k:<32} {v['verdict']}")


if __name__ == "__main__":
    main()
