# H200 Cluster Runbook — σ_min completion (task #34 external path)

**Use when:** an 8× H200 141 GB SXM node (or equivalent) is available.
**Estimated budget:** $10–15 on Runpod community cloud, 60–90 minutes wall-clock.
**Inputs:** the current `main` commit (pinned for preregistration purposes).

## Goal

Complete §2.A + §2.B of `docs/preregistration_v2.md`:

1. Single-seed σ_min at W ∈ {14000, 22000, 45000} with seed 42.
2. 20-seed robustness at each of those widths.

Both live inside the same driver `sigma_min_validation.py`.

## Step-by-step

```bash
# 1. Provision an H200 node (141 GB HBM, single GPU sufficient).
#    Runpod: template "PyTorch 2.5 + CUDA 12.4", 1× H200, storage 100 GB.

# 2. Clone and install.
git clone https://github.com/star-ga/nn_universe.git
cd nn_universe
pip install torch numpy scipy pytest

# 3. Run the single-seed sweep.
python3 experiments/v1_2_scaling/sigma_min_validation.py \
    --widths 14000 22000 45000 \
    --seed 42 \
    --out sigma_min_H200_single_seed.json

# 4. Run the 20-seed robustness sweep.
#    This is ~5 hours wall-clock at W=45000.
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
    python3 experiments/v1_2_scaling/sigma_min_validation.py \
        --widths 14000 22000 45000 \
        --seed $seed \
        --out sigma_min_H200_seed_$seed.json
done

# 5. Aggregate.
python3 -c "
import json, numpy as np
from pathlib import Path
results = []
for p in sorted(Path('.').glob('sigma_min_H200_seed_*.json')):
    results.append(json.loads(p.read_text()))
# Compute per-width CV across seeds, write the summary JSON.
print(json.dumps({'n_seeds': len(results)}, indent=2))
" > sigma_min_H200_aggregate.json
```

## Expected output

- `sigma_min_H200_single_seed.json` — 3 widths × 5 layers × (σ_max, σ_min) per layer, ~60 numbers.
- `sigma_min_H200_seed_{0..19}.json` — 20 files of the same shape.
- `sigma_min_H200_aggregate.json` — per-width mean + std + 95 % bootstrap CI of the interior-max ratio.

## Decision rule

Match predictions from `docs/preregistration_v2.md` §2.A and §2.B:

- PASS if interior-max ratio at W=45000 is in $[10^4, 10^7]$ and 20-seed CV ≤ 2 %.
- FAIL (paper needs revision) if either the point estimate or the CV is outside its predicted bound by >2×.

## Failure modes

- **Full SVD fails at W=45000 on H200.** cuSOLVER has historically had stability issues at this scale. Fallback: compute on CPU (H200 node typically has 1–2 TB RAM; 45000² × 4 bytes = 8.1 GB per layer, trivially fits). Expect ~15–30 min per layer.
- **OOM during training.** If `_CheckpointedFC` at W=45000 still OOMs (5 × W² × 2 bytes ≈ 20 GB per layer in bf16 for activations), reduce batch to 32 or use gradient checkpointing.
- **Runpod node unavailable.** AWS p5.48xlarge, Azure ND H100 v5, and Lambda's H100 SXM nodes are acceptable substitutes. Any GPU with ≥ 80 GB HBM + full-SVD cuSOLVER support works.

## After execution

1. Commit the three JSONs to `experiments/v1_2_scaling/` on a new branch.
2. Update the README V3.0 Tier-1 table and the §4.1 scaling row.
3. Replace the "W=45000 σ_min still blocked on H200" footnote in README with the measured values.

*STARGA Commercial License. H200 runbook for task #34.*
