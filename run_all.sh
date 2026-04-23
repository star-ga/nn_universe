#!/usr/bin/env bash
# Execute the full V1.0–V3.1 experimental pipeline from scratch.
# Idempotent where possible; safe to re-run individual phases.
set -euo pipefail

echo "=== nn_universe full pipeline ==="
echo "Hardware: $(python3 -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

echo ""
echo "--- V1.0 toy experiment (reference) ---"
python3 experiment_pytorch.py

echo ""
echo "--- V1.0 EWC standalone ---"
python3 ewc_experiment.py

echo ""
echo "--- V1.0 / V1.2 scaling (6-width core) ---"
python3 scaling_experiment.py

echo ""
echo "--- V1.2 ladder fill (widths 32/128/512/2048) ---"
python3 experiments/v1_2_scaling/fill_ladder.py

echo ""
echo "--- V1.2 seed robustness ---"
python3 experiments/v1_2_scaling/seed_robustness.py --width 256 --seeds 0 1 2 3 4 5

echo ""
echo "--- V1.2 depth sweep ---"
python3 experiments/v1_2_scaling/depth_sweep.py

echo ""
echo "--- V2.0 lattice-analytic Cauchy convergence ---"
python3 experiments/v2_0_lattice/lattice_analytic.py --d 2 --levels 4 --eval-density 121

echo ""
echo "--- V2.1 QEC decoder — single large run (width 256) ---"
python3 experiments/v2_1_qec/train.py --L 5 --p 0.05 --width 256 --steps 50000

echo ""
echo "--- V2.1 QEC decoder — width sweep ---"
python3 experiments/v2_1_qec/run_sweep.py --widths 32 64 128 256 512 1024 --steps 15000

echo ""
echo "--- V3.1 mock α-drift pipeline (physical κ + strong signal) ---"
python3 experiments/v3_1_alpha/mock_pipeline.py --kappa-scale 1 --out experiments/v3_1_alpha/mock_physical.json
python3 experiments/v3_1_alpha/mock_pipeline.py --kappa-scale 1e26 --noise-sigma 1e-5 --out experiments/v3_1_alpha/mock_strong_signal.json

echo ""
echo "--- Regenerate results summary + plots ---"
python3 experiments/visualize.py --plots

echo ""
echo "--- Regression tests ---"
pytest tests/ -v --tb=short -W ignore::SyntaxWarning

echo ""
echo "Pipeline complete. See docs/results_summary.md and plots/ for outputs."
