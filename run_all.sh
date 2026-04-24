#!/usr/bin/env bash
# run_all.sh — regenerate every experimental result referenced in the paper.
#
# Scope:
#   - Runs every experiment that completes on a single RTX 3080 10GB
#     (or CPU fallback) in a few hours.
#   - Does NOT run the V3.0 cluster-scale H200 experiments (W=14000 /
#     W=22000 / W=45000 NN training) — those need external compute.
#
# Order is chosen so that each experiment's inputs are available when
# it runs. All scripts are idempotent: they overwrite their JSON output.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

log() { printf "\n\033[1;34m[run_all]\033[0m %s\n" "$*"; }

log "Python : $(python3 --version)"
log "Torch  : $(python3 -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available())')"
log "Threads: OMP=$OMP_NUM_THREADS"

# =============================================================
# V1.0 — 296k-param toy experiment (parent paper baseline)
# =============================================================
log "V1.0 — parent-paper toy experiment (296k params)"
python3 experiment_pytorch.py

# =============================================================
# V1.2 — scaling + seed robustness + depth sweep
# =============================================================
log "V1.2 — scaling ladder (6 widths)"
python3 scaling_experiment.py

log "V1.2 — ladder fill (additional widths)"
python3 experiments/v1_2_scaling/fill_ladder.py

log "V1.2 — seed robustness at W=256"
python3 experiments/v1_2_scaling/seed_robustness.py --width 256 --seeds 0 1 2 3 4 5

log "V1.2 — depth sweep"
python3 experiments/v1_2_scaling/depth_sweep.py

# =============================================================
# V2.0 — lattice-embedded Cauchy convergence
# =============================================================
log "V2.0 — lattice analytic Cauchy test"
python3 experiments/v2_0_lattice/lattice_analytic.py --d 2 --levels 4 --eval-density 121

# =============================================================
# V2.1 — QEC toric-code decoder
# =============================================================
log "V2.1 — QEC decoder width sweep"
python3 experiments/v2_1_qec/run_sweep.py --widths 32 64 128 256 512 1024 --steps 15000

# =============================================================
# V3.1 — mock α-drift pipeline (no real data)
# =============================================================
log "V3.1 — mock α-drift pipeline"
python3 experiments/v3_1_alpha/mock_pipeline.py --kappa-scale 1 \
  --out experiments/v3_1_alpha/mock_physical.json
python3 experiments/v3_1_alpha/mock_pipeline.py --kappa-scale 1e26 --noise-sigma 1e-5 \
  --out experiments/v3_1_alpha/mock_strong_signal.json

# =============================================================
# V4.0 — non-deep learning baselines (4 systems, 5 seeds)
# =============================================================
log "V4.0 — learning baselines (linear / kernel / logistic / GP)"
python3 experiments/v4_0_uniqueness/learning_baselines.py \
  --out experiments/v4_0_uniqueness/v4_learning_baselines_results.json

# =============================================================
# V4.1 — trained-vs-untrained NN
# =============================================================
log "V4.1 — trained-vs-untrained NN (5 widths × 5 seeds)"
python3 experiments/v4_0_uniqueness/run_trained_vs_untrained.py

# =============================================================
# V5.0 — U(1) pure-gauge lattice FIM
# =============================================================
log "V5.0 — U(1) lattice gauge (L=8, d=4, 3 seeds)"
python3 experiments/v5_0_lattice_qcd/lattice_u1.py \
  --seeds 0 1 2 --decorr 20 --n-samples 80

# =============================================================
# V6.x — mechanism experiments (depth / width / boolean / transformer)
# =============================================================
log "V6.0 — MLP depth sweep (Hanin-Nica empirical confirmation)"
python3 experiments/v6_0_depth_mechanism/depth_sweep.py

log "V6.1 — width sweep at fixed depth=8"
python3 experiments/v6_0_depth_mechanism/width_sweep.py

log "V6.2 — trained-NN depth sweep"
python3 experiments/v6_0_depth_mechanism/trained_depth_sweep.py

log "V6.3 — layered boolean-circuit depth sweep"
python3 experiments/v6_0_depth_mechanism/bc_depth_sweep.py

log "V6.4 — transformer depth sweep"
python3 experiments/v6_0_depth_mechanism/transformer_depth_sweep.py

# =============================================================
# V7.0 — SU(2) non-abelian lattice gauge
# =============================================================
log "V7.0 — SU(2) non-abelian lattice gauge (L=3, d=4, 3 seeds)"
python3 experiments/v7_0_lattice_su2/lattice_su2.py \
  --L 3 --d 4 --thermalise 50 --n-samples 20 --decorr 10 --seeds 0 1 2

# =============================================================
# V5.0-stats — bootstrap CIs + Mann-Whitney (depends on everything above)
# =============================================================
log "V5.0-stats — bootstrap CIs + Mann-Whitney on the 12-system dichotomy"
python3 experiments/v5_0_dichotomy_stats/dichotomy_stats.py

# =============================================================
# σ_min validation (task #34) — small widths where CPU SVD is fast
# =============================================================
log "σ_min validation (widths 256 / 1024 / 4096 / 8192)"
python3 experiments/v1_2_scaling/sigma_min_validation.py \
  --widths 256 1024 4096 8192

# =============================================================
# Regenerate aggregated summary + plots
# =============================================================
log "Aggregate results + plots"
python3 experiments/visualize.py --plots

# =============================================================
# Regression tests
# =============================================================
log "Running pytest (104 tests expected)"
python3 -m pytest tests/ -q

log "Pipeline complete."
log "Paper source: docs/paper_draft.md"
log "Rebuild PDF with:"
log "  pandoc docs/paper_draft.md -o docs/nn_universe_paper_V2.pdf --pdf-engine=lualatex -V mainfont='DejaVu Serif' -V geometry:margin=1in -V colorlinks=true -V fontsize=10pt"
