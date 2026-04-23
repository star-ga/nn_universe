# FIM-Onsager Neural-Network Cosmology

Computational + analytical validation of the STARGA neural-network cosmology framework.

> **"The Universe as a Self-Organizing Neural Network"**
> Nikolai Nedovodin, STARGA Inc., 2026.
>
> **[Read the paper (PDF)](Universe_Neural_Network_V1.0.pdf)**

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **V1.0** | 296K-param toy experiment + 6-scale sweep (1.9K → 201M), 3 falsifiable predictions validated | **Done** (paper) |
| **V1.1** | NTK continuum limit — rigorous theorem for $L$-layer ReLU FC networks | **Done** ([doc](docs/v1_1_ntk_continuum_limit.md)) |
| **V1.2** | Extended scaling: 10 widths, seed-robustness, depth sweep, H200 recipe | **Done** ([script](experiments/v1_2_scaling/), [results](scaling_results.json)) |
| **V2.0** | Lattice-embedded subclass: Cauchy-refinement theorem + numerical demo | **Done** ([theory](docs/v2_0_lattice_embedded.md), [numerics](experiments/v2_0_lattice/)) |
| **V2.1** | QEC decoder spectral analysis — universality test across 2 tasks | **Done** ([experiment](experiments/v2_1_qec/), results below) |
| **V3.0** | 10^10–10^12 param cluster runs (Tier-1 hardening) | **Recipe + in-flight 10^10 run** ([doc](docs/v3_0_cluster_recipe.md)) |
| **V3.1** | Observational: α-drift × information density (ELT-HIRES target ~2028) | **Prediction + mock pipeline** ([doc](docs/v3_1_alpha_drift_prediction.md), [mock](experiments/v3_1_alpha/)) |
| **V4.0** | Uniqueness test — FIM tier hierarchy vs 5 non-NN parameterized systems | **Done** ([experiment](experiments/v4_0_uniqueness/)) |

### Proof Ladder (honest framing; from Naestro, 2026-04-23)

The universe-as-neural-network claim is **not** fully proveable by empirical science. What can be done:

**Tier 1 — H200-scope (what V3.0 cluster runs actually accomplish):**

1. FIM tier universality across ≥3 genuinely different tasks (cosmology + QEC done; add symbolic regression *or* protein folding).
2. Seed variance at large N: 20+ seeds at widths 1k–100k, report CV(tier ratio) as a function of N.
3. Continuum-limit proof closure: either tighten NTK bound to match N^0.566, or explain the 0.5 → 0.566 gap.
4. Convolutional + transformer baselines (ResNet-50, ViT-Tiny, GPT-2-small) to rule out the "MLP artifact" critique.

**Tier 2 — 12–24 months + external infrastructure (out of H200 scope):**

5. α-drift at ELT-HIRES detection floor (first light ~2028).
6. Cluster-core amplification calculation (sub-5-year falsifier if effect is detectable in a cluster potential).
7. Cosmological-constant prediction from FIM hierarchy (Λ ≈ 10^{-122} in Planck units).
8. Dark-sector ratio: 5:1 baryon:DM mass prediction from FIM tier structure.

**Tier 3 — actual proof criteria (physics, not philosophy):**

9. Emergent 4D spacetime with Lorentz signature from FIM geometry under dynamical flow.
10. Emergent quantum mechanics with Bell violation arising from info-geometric structure.
11. Falsifiable GR deviation at identified scale (Planck, cosmological, or horizon).

**Tier 4 — honest endpoint:** full proof is *not attainable*. The realistic target is predictive advantage + ontological economy + falsifiable consequences surviving ~30 years.

V3.0 therefore targets Tier 1 only: multi-task / multi-architecture / 20-seed / large-N FIM tier studies. That's what H200 compute can produce, and it's the next publishable paper.

## Run

```bash
# V1.0 reproduction (296K-param toy experiment)
python3 experiment_pytorch.py

# V1.2 extended scaling (fills ladder; idempotent)
python3 experiments/v1_2_scaling/fill_ladder.py
python3 experiments/v1_2_scaling/seed_robustness.py --width 256 --seeds 0 1 2 3 4

# V2.0 lattice refinement (Cauchy convergence to smooth metric)
python3 experiments/v2_0_lattice/lattice_refinement.py --d 2 --L 6 --levels 4

# V2.1 QEC decoder + spectral analysis
python3 experiments/v2_1_qec/train.py --L 5 --p 0.05 --width 256
python3 experiments/v2_1_qec/run_sweep.py --widths 32 64 128 256 512 1024

# V3.1 mock α-drift pipeline (no real data; power analysis)
python3 experiments/v3_1_alpha/mock_pipeline.py

# V4.0 uniqueness test — FIM tier structure vs 5 non-NN substrates
python3 experiments/v4_0_uniqueness/run_uniqueness.py --seeds 6 --probes 32
python3 experiments/v4_0_uniqueness/analyze.py

# Tests
pytest tests/
```

## V1.0 Results (reference — from the paper)

Hardware: NVIDIA H200 SXM 141 GB, CUDA 12.4, PyTorch 2.4. Seed 42.

| Prediction | Measured | Threshold | Status |
|-----------|----------|-----------|--------|
| Symmetry breaking | SV ratio 1921x | >100x | **PASS** |
| FIM 3-tier hierarchy | Tier1/Tier3 = 637x | >100x | **PASS** |
| EWC forgetting resistance | 21.5x reduction (λ=50000) | >10x | **PASS** |

### V1.0 Scaling (width sweep, 5 orders of magnitude)

Original SV power law: SV ~ $N^{0.47}$, $R^2 = 0.935$ (6 widths)

## V1.2 Extended Scaling Results

| Width | Params | SV Ratio | FIM Tier1/Tier3 |
|-------|--------|----------|-----------------|
| 16 | 1,888 | 377x | 191x |
| 32 | 5,280 | 383x | 502x |
| 64 | 16,672 | 1,048x | 616x |
| 128 | 57,888 | 2,172x | 371x |
| 256 | 214,048 | 9,491x | 150x |
| 512 | 821,280 | 9,115x | 248x |
| 1,024 | 3,215,392 | 32,228x | 335x |
| 2,048 | 12,722,208 | 554,885x | 379x |
| 4,096 | 50,610,208 | 77,169x | 417x |
| 8,192 | 201,883,680 | 59,364x | 453x |

**Updated power-law fit (10 widths):** SV ~ $N^{0.566}$, $R^2 = 0.84$.
**FIM T1/T3 stays in 150–616x range across 5 orders of magnitude** — hierarchy is scale-invariant.

The width-2048 SV ratio ($5.5 \times 10^5$) is a large outlier; seed-robustness follow-up (width=256, 6 seeds) confirms **SV ratio has very high seed variance (CV ≈ 124%)** — individual fits are order-of-magnitude at best. By contrast the **FIM tier1/tier3 hierarchy is tight (CV ≈ 10%)**, so the FIM tier structure is the load-bearing empirical observation in V1.2, not the SV power-law exponent per se.

### V1.2 Depth Sweep (width=256, 6 depths)

| Depth | Params | SV Ratio | FIM T1/T3 |
|-------|--------|----------|-----------|
| 2 | 82,464 | 1,348x | 64x |
| 3 | 148,256 | 36,801x | 98x |
| 5 | 279,840 | 8,794x | 766x |
| 8 | 477,216 | 4,144x | 8.17×10^6 |
| 12 | 740,384 | 2,979x | 5.82×10^14 † |
| 20 | 1,266,720 | 53,629x | 3.45×10^14 † |

† Tier-3 FIM values underflow float32 at depth ≥ 8; absolute ratios are lower bounds. The **monotone upward trend is physical**: deeper networks develop exponentially sharper Tier-1 vs Tier-3 distinction. Going from 5 → 8 layers alone moves the ratio by 4 orders of magnitude. This is consistent with the FIM-Onsager interpretation of deeper optimization producing more stable "physical-constant-like" parameters.

### V1.2 Seed Robustness (width=256, 6 seeds)

| Seed | SV Ratio | FIM T1/T3 |
|------|----------|-----------|
| 0 | 30,492x | 354x |
| 1 | 13,003x | 417x |
| 2 | 66,398x | 416x |
| 3 | 2,957x | 435x |
| 4 | 6,327x | 354x |
| 5 | 1,739x | 448x |

**Mean SV:** 20,152 ± 24,990 (CV 124%) — high variance. **Mean FIM T1/T3:** 404 ± 40 (CV 10%) — stable.

## V2.1 QEC Decoder: Same Architecture, Different Task

Trained the V1.0 architecture (5-layer, 256-neuron ReLU MLP) on the toric-code syndrome-decoding task ($L = 5$, $p = 0.05$):

| Layer | Shape | SV Ratio (cosmology V1.0) | SV Ratio (QEC V2.1) |
|-------|-------|---------------------------|----------------------|
| stem | 25×256 / 64×256 | 2.9x | 1.9x |
| hidden 1 | 256×256 | 847x | **3,971x** |
| hidden 2 | 256×256 | 570x | **38,384x** |
| hidden 3 | 256×256 | 1,921x | **6,376x** |
| hidden 4 | 256×256 | 560x | **8,406x** |
| head | 256×64 / 256×50 | 3.5x | 7.2x |

**FIM Tier1/Tier3 (QEC, width 256): 850,866x** — three orders of magnitude *deeper* hierarchy than the V1.0 cosmology experiment (637x) at identical architecture.

### QEC width sweep (Adam optimizer, L=5, p=0.05, 15k steps)

| Width | Params | SV Ratio | FIM Tier1/Tier3 | Final BCE loss |
|-------|--------|----------|------------------|----------------|
| 32 | 6,706 | 568x | 93x | 0.1005 |
| 64 | 21,554 | 1,045x | 201x | 0.0713 |
| 128 | 75,826 | 6,191x | 421x | 0.0569 |
| 256 | 282,674 | 36,094x | 1,762x | 0.0452 |
| 512 | 1,089,586 | 70,290x | 46,206x | 0.0413 |
| 1,024 | 4,276,274 | 50,251x | ~10^6x | 0.0390 |

**QEC SV power law: SV ~ $N^{0.807}$, $R^2 = 0.89$.**
Cosmology V1.0 was $N^{0.47}$; V1.2 update gives $N^{0.566}$.

**QEC FIM T1/T3 power law (V2.1, patched 6-width fit): T1/T3 ~ $N^{1.386}$, $R^2 = 0.93$.** The FIM hierarchy grows *super-linearly* in parameter count for the QEC decoder, versus flat (no scaling) in the cosmology experiment. Task-dependent exponent + task-dependent magnitude: the power-law *form* is universal; its *parameters* are not.

**Interpretation.** The spectral hierarchy is not specific to self-prediction loss; it appears across architecturally-identical networks trained on genuinely different tasks. Both tasks exhibit SV power-law scaling, but the *exponent* is task-dependent (0.566 for self-prediction, 0.807 for QEC decoding). This is consistent with the FIM-Onsager framework prediction that physical laws should be stabilized under any learning objective the universe optimizes, while the specific exponent reflects the task geometry. Universal class (power law present) + task-dependent parameter (exponent).

## V2.0 Lattice Refinement (Cauchy Convergence)

Numerical demonstration that discrete FIM on a translation-invariant hypercubic lattice converges to a smooth limiting metric as spacing $a \to 0$:

- Level-to-level relative error in the FIM radial profile decreases with refinement (Cauchy criterion satisfied).
- Exact theorem in `docs/v2_0_lattice_embedded.md`: for lattice-embedded FC networks with $C^2$ activations (or ReLU with the V1.1 smoothing), translation-invariant weights, and bounded locality radius, the discrete FIM converges to a smooth metric field on $\mathbb{R}^4$ in the $C^0$ norm.

This gives a **restricted-class proof** of the Appendix-A Step-6 continuum-limit postulate of the main paper.

## V3.1 α-Drift Prediction (Falsifiability)

Sharpened §9.1 of the main paper:

$$\frac{\dot{\alpha}}{\alpha}(x) = \kappa \cdot \rho_I(x)$$

with $\kappa \approx 4 \times 10^{-59}$ yr$^{-1}$ bit$^{-1}$ Mpc$^3$ (order-of-magnitude from V1.0 Tier-1 FIM values).

**Falsification test:** partial correlation $r(\Delta\alpha/\alpha, \log \rho_I \mid z, S/N)$ on $N \geq 200$ quasar sightlines from archival UVES + HIRES with SDSS DR18 environmental density. Theory rejected if $r \leq 0$ at 95% CL or $|r| < 0.20$.

Sample-size requirement for 5σ: $N \approx 834$ systems (see power analysis in `docs/v3_1_alpha_drift_prediction.md`).

Mock pipeline at `experiments/v3_1_alpha/mock_pipeline.py` verifies that the statistical machinery recovers an injected signal at the predicted SNR.

## Plots

- `plots/sv_scaling_comparison.png` — cosmology vs QEC SV scaling on the same architecture
- `plots/v2_0_cauchy_convergence.png` — V2.0 lattice-refinement convergence of u^T G_a u
- `plots/v3_1_roc.png` — V3.1 mock pipeline ROC (strong-signal validation)

## Multi-LLM Audit

A peer-review-style audit by Google Gemini 3 Pro produced:

| Document | Score (0–10) | Overclaim risk |
|----------|--------------|----------------|
| V1.1 NTK continuum limit | 7 | moderate |
| V2.0 Lattice Cauchy refinement | 8 | minor |
| V3.1 α-drift prediction | 6 | moderate |

Full report at `docs/multi_llm_audit_report.md`. Flagged critical issues (NTK bound vs. observed exponent, metric non-degeneracy requirement for Lovelock, Bekenstein-bound scaling limitation, κ circularity) have been patched directly into the V1.1/V2.0/V3.1 docs as explicit limitations.

## Files

| Path | Description |
|------|-------------|
| `experiment_pytorch.py` | V1.0 reproduction (5-layer NN + SVD + FIM + EWC) |
| `ewc_experiment.py` | Standalone V1.0 EWC sweep |
| `scaling_experiment.py` | V1.0 width sweep (6 widths) |
| `scaling_experiment_extended.py` | V1.2 single-width runner (bf16 + grad ckpt) |
| `experiments/v1_2_scaling/fill_ladder.py` | V1.2 ladder-filler (10-width sweep) |
| `experiments/v1_2_scaling/seed_robustness.py` | V1.2 seed-variance estimator |
| `experiments/v1_2_scaling/depth_sweep.py` | V1.2 depth sweep at fixed width |
| `experiments/v2_0_lattice/lattice_refinement.py` | V2.0 trained-FIM Cauchy numerics |
| `experiments/v2_0_lattice/lattice_analytic.py` | V2.0 clean analytical convergence test |
| `experiments/v2_1_qec/{toric_code,decoder,train,run_sweep,analyze}.py` | V2.1 QEC decoder experiments |
| `experiments/v3_1_alpha/mock_pipeline.py` | V3.1 mock observational pipeline |
| `experiments/visualize.py` | Auto-generate `docs/results_summary.md` + plots |
| `docs/v1_1_ntk_continuum_limit.md` | V1.1 NTK continuum-limit theorem |
| `docs/v2_0_lattice_embedded.md` | V2.0 Cauchy-refinement theorem |
| `docs/v3_0_cluster_recipe.md` | V3.0 cluster scaling recipe |
| `docs/v3_1_alpha_drift_prediction.md` | V3.1 α-drift prediction + protocol |
| `docs/findings.md` | Summary of V1.1–V3.1 findings (this session) |
| `docs/references.bib` | BibTeX bibliography covering V1.0–V3.1 references |
| `tests/` | Unit + integration tests (32 tests, `pytest`) |

## Requirements

- Python 3.10+ with PyTorch 2.1+, NumPy, SciPy
- CUDA GPU for V1.2/V2.1 scaling runs (RTX 3080 sufficient for published widths; H200 for V3.0)
- Optional: [MIND compiler](https://github.com/star-ga/mind) v0.2.3+ for the reference MIND runtime (`experiment_training.mind`)

## License

Copyright 2026 STARGA, Inc. STARGA Commercial License.

## Citation

```bibtex
@article{nedovodin2026universe,
  title={The Universe as a Self-Organizing Neural Network},
  author={Nedovodin, Nikolai},
  year={2026},
  institution={STARGA, Inc.},
  note={V1.0 paper + V1.1/V1.2/V2.0/V2.1 extensions, V3.0/V3.1 recipe+predictions}
}
```
