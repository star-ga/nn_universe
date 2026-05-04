#!/usr/bin/env bash
# reproduce_main_results.sh
# ==========================
# One-command reproduction of the four load-bearing results in the paper:
#   1. V5.0 dichotomy statistics (Mann–Whitney U, bootstrap 95% CI, threshold sensitivity)
#   2. V6.0 mechanism (Hanin–Nica depth sweep — H1: Var[log F] vs L; H2: log T1/T3 vs sqrt(L))
#   3. V4.5 partition-invariant verification (Gini, effective rank, top-1% FIM mass)
#   4. V6.0c pooling-error bound numerical verification
#
# Wall-clock on a single CPU is ~12 minutes; on a CUDA GPU ~3 minutes.
# All four scripts are deterministic (seeds fixed in source) and produce
# JSON outputs in the same directory as the script.
#
# Usage:
#     bash scripts/reproduce_main_results.sh           # full reproduction
#     bash scripts/reproduce_main_results.sh --quick   # skip V6.0 (~7 min savings)
#
# Environment:
#     Requires Python 3.10+, numpy, scipy, torch (see requirements.txt).
#     Run from the repository root.

set -euo pipefail

cd "$(dirname "$0")/.."

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=1
fi

echo "=================================================================="
echo "  nn_universe — main-result reproduction"
echo "=================================================================="
echo

run_step () {
    local label="$1"
    local script="$2"
    echo "[$label] $script"
    python3 "$script"
    echo "[$label] DONE"
    echo
}

# --- 1. V5.0 dichotomy statistics ---------------------------------------
echo "----- (1/4) V5.0 dichotomy statistics --------"
run_step "V5.0" experiments/v5_0_dichotomy_stats/dichotomy_stats.py
run_step "V5.1" experiments/v5_0_dichotomy_stats/threshold_sensitivity.py
run_step "V5.2" experiments/v5_0_dichotomy_stats/mw_bootstrap.py

# --- 2. V6.0 mechanism (Hanin–Nica depth sweep) -------------------------
if [[ $QUICK -eq 0 ]]; then
    echo "----- (2/4) V6.0 mechanism --------------------"
    run_step "V6.0" experiments/v6_0_depth_mechanism/depth_sweep.py
else
    echo "----- (2/4) V6.0 mechanism --- SKIPPED (--quick) ----"
    echo
fi

# --- 3. V4.5 partition-invariant verification ---------------------------
echo "----- (3/4) V4.5 partition-invariant verification ----"
run_step "V4.5" experiments/v4_3_statistics/partition_invariant_dichotomy.py

# --- 4. V6.0c pooling-error bound ---------------------------------------
echo "----- (4/4) V6.0c pooling-error bound ----------------"
run_step "V6.0c" experiments/v6_0_mechanism/pooling_error_bound.py

echo "=================================================================="
echo "  Reproduction complete."
echo
echo "  Compare output JSONs against the committed reference results:"
echo "    experiments/v5_0_dichotomy_stats/dichotomy_stats_results.json"
echo "    experiments/v5_0_dichotomy_stats/v5_1_threshold_sensitivity_results.json"
echo "    experiments/v5_0_dichotomy_stats/v5_2_mw_bootstrap_results.json"
echo "    experiments/v6_0_depth_mechanism/v6_0_depth_sweep.json"
echo "    experiments/v4_3_statistics/v4_3_partition_invariant_dichotomy.json"
echo "    experiments/v6_0_mechanism/v6_0c_pooling_error_bound_results.json"
echo
echo "  Headline numbers expected:"
echo "    V5.0  Mann–Whitney p = 1.7e-17,  rank-biserial r = 1.000"
echo "    V6.0  H1 (Var[log F] propto L)            R^2 = 0.906"
echo "          H2 (log T1/T3 propto sqrt(L))       R^2 = 0.983"
echo "    V4.5  Gini, eff-rank, top-1% mass — all monotone in depth"
echo "    V6.0c bound satisfied for L in {4, 8, 12}"
echo "=================================================================="
