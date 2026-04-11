# FIM-Onsager Toy Cosmology Experiment

Computational validation of three predictions from the neural-network cosmology framework.

> **"The Universe as a Self-Organizing Neural Network"**
> Nikolai Nedovodin, STARGA Inc., 2026.

## Run

```bash
# MIND (GPU — cuBLAS TF32)
mindc run --target cuda

# MIND (CPU)
mindc run --target cpu

# Python (reference)
python3 experiment_pytorch.py
```

## Experiment Results

Hardware: NVIDIA H200 SXM 141GB, CUDA 12.4, Driver 570.211.01. Seed 42.

| Prediction | Measured | Threshold | Status |
|-----------|----------|-----------|--------|
| Symmetry breaking | SV ratio 1921x | >100x | **PASS** |
| FIM 3-tier hierarchy | Tier1/Tier3 = 637x | >100x | **PASS** |
| EWC forgetting resistance | 21.5x reduction (λ=50000) | >10x | **PASS** |

### Symmetry Breaking Detail (per-layer SVD, seed 42)

| Layer | Shape | Top-3 SV | SV Ratio | SV Std |
|-------|-------|----------|----------|--------|
| net.0.weight | 256x64 | 1.828, 1.789, 1.765 | 2.9x | 0.326 |
| net.2.weight | 256x256 | 2.076, 1.253, 1.247 | 847.1x | 0.348 |
| net.4.weight | 256x256 | 1.536, 1.295, 1.275 | 570.3x | 0.346 |
| net.6.weight | 256x256 | 1.366, 1.314, 1.280 | **1921.5x** | 0.344 |
| net.8.weight | 256x256 | 1.313, 1.297, 1.273 | 560.6x | 0.344 |
| net.10.weight | 64x256 | 1.187, 1.162, 1.138 | 3.5x | 0.252 |

Interior layers (2-8) show extreme SV ratios (560-1921x), indicating spontaneous symmetry breaking. Input/output layers (0, 10) remain near-symmetric (2.9-3.5x) due to dimensional bottleneck.

### FIM 3-Tier Hierarchy Detail (diagonal FIM, 1000 gradient samples)

| Tier | Percentile | Count | Mean FIM | Description |
|------|-----------|-------|----------|-------------|
| Tier 1 | Top 1% | 2,963 | 7.73e-06 | "Physical constants" |
| Tier 2 | 1-50% | 145,165 | 4.0e-07 | "Coupling constants" |
| Tier 3 | Bottom 50% | 148,128 | 1.21e-08 | "Gauge DOF" |

Tier1/Tier3 ratio: **637x**. Top 10 FIM values: [2.2e-05, 2.2e-05, 2.2e-05, 2.2e-05, 2.2e-05, 2.1e-05, 2.1e-05, 2.1e-05, 2.1e-05, 2.0e-05].

### EWC Detail (per-sample Fisher diagonal, seed 42)

| λ | Task A Degradation | Forgetting Reduction |
|---|-------------------|---------------------|
| 0 | 80.9x | — |
| 10,000 | 25.3x | 3.2x |
| 50,000 | 3.8x | 21.5x |
| 100,000 | 1.8x | 45.6x |

### Scaling Experiment (width sweep, 5 orders of magnitude)

SV ratio power law: **SV ~ N^0.47, R² = 0.935**

| Width | Params | SV Ratio | FIM Tier1/Tier3 |
|-------|--------|----------|-----------------|
| 16 | 1,888 | 377x | 191x |
| 64 | 16,672 | 1,048x | 616x |
| 256 | 214,048 | 9,491x | 150x |
| 1,024 | 3,215,392 | 32,228x | 335x |
| 4,096 | 50,610,208 | 77,169x | 417x |
| 8,192 | 201,883,680 | 59,364x | 453x |

Symmetry breaking intensifies with scale. FIM hierarchy persists at every scale (150-616x).

## Files

| File | Description |
|------|-------------|
| `experiment_pytorch.py` | Full experiment: 5-layer NN, SVD, FIM, EWC |
| `ewc_experiment.py` | Standalone EWC forgetting experiment |
| `scaling_experiment.py` | Width sweep: 16 to 8192 (5 orders of magnitude) |
| `experiment_training.mind` | Self-prediction training loop in MIND |
| `toy_experiment_results.json` | Raw results (SV, FIM, forgetting) |
| `ewc_results.json` | EWC results across λ values |
| `scaling_results.json` | Width sweep results + power-law fits |

## Requirements

- [MIND compiler](https://github.com/star-ga/mind) v0.2.3+
- CUDA toolkit (for GPU target)
- Python 3.8+ with PyTorch (for reference implementation)

## License

Copyright 2026 STARGA, Inc. STARGA Commercial License.

## Citation

```bibtex
@article{nedovodin2026universe,
  title={The Universe as a Self-Organizing Neural Network},
  author={Nedovodin, Nikolai},
  year={2026},
  institution={STARGA, Inc.}
}
```
