# TMLR submission cover letter — Fisher Information Tier Hierarchy

**Target venue:** Transactions on Machine Learning Research (TMLR), rolling submission.
**Nature of the work:** empirical + mechanism-backed universality claim about the Fisher-information diagonal of deep networks and deeply-composed non-learning substrates.
**Submission system:** OpenReview (TMLR group).
**Submission state:** anonymous (TMLR is double-blind during review).

---

## 1. Executive summary

The **three-tier FIM diagonal hierarchy** — a ratio $T_1/T_3$ between the mean Fisher-information diagonal value of the top-1 % and bottom-50 % of parameters — is empirically a signature of **deep layered sequential composition** as a computational primitive, mechanistically a consequence of the Hanin–Nica 2020 log-normal theorem for products of random Jacobians, and verifiably separates a 12-system substrate panel into two groups with **complete rank separation** ($p = 1.7 \times 10^{-17}$, rank-biserial $r = 1.000$, ROC AUC = 1.0).

The claim is *panel-bounded* and *trained-network conditioned* (Definition 1, §4.5); it is not asserted as a universality theorem outside the V2 substrate panel.

## 2. What's new in V11.5 (2026-05-01)

V11.5 is the production-scale capstone of the V11 line. Beyond the V11.2 baseline, it adds:

- **ImageNet-1K ResNet-50 90-epoch from-scratch trajectory** at the canonical NeurIPS-era recipe (74.6 % top-1, $T_1/T_3$ drops 4 367× monotonically). Pattern A (monotonic decrease) confirmed at canonical scale; the V9.5b Imagenette anomaly is now solidly a small-data-regime outlier rather than an architectural property.
- **Cross-scale convergence finding**: ResNet-50 trained on full ImageNet-1K for 90 epochs ends with $T_1/T_3 = 778$ at 74.6 % top-1 — *exactly matching* ResNet-18 trained 10 epochs on CIFAR-10 ($T_1/T_3 = 778$ at 81.4 %). Two networks 25× apart in parameter count, trained on datasets 100× apart in effective sample count, both land at the same trained-equilibrium tier ratio.
- **Pythia-1.4B and Pythia-2.8B**: first and second billion-parameter LM data points in the FIM-diagonal literature.
- **Imagenette ViT-B/16 from-scratch trajectory** (third distinct trajectory pattern).
- **Three partition-invariant statistics** (Gini, effective rank, top-1 % mass) confirm the dichotomy without any tier-partition tuning.
- **Pre-registered dynamical-isometry falsifier confirmed**: identity+ε init drops $T_1/T_3$ to 42 (below threshold), exactly as the Hanin–Nica mechanism predicts.

## 3. Why TMLR

- TMLR's evaluation criteria reward *correctness, novelty of contribution, and reproducibility* over splashiness. The paper's claim is statistically testable and the falsifier is registered; we expect TMLR's reviewer pool to be a strong match.
- TMLR allows **arXiv coexistence** at any time, and TMLR-published papers feed into the **NeurIPS J2C track** (Sept 26 deadline). Submitting here is the canonical path for a paper that should both be peer-reviewed and reach a NeurIPS audience.
- The author's first arXiv submission is endorsement-blocked; TMLR's no-endorsement requirement removes one upstream blocker.

## 4. Reproducibility (TMLR Code & Data Availability)

The repository at `https://github.com/<anonymous>/nn_universe` (anonymised during review; deanonymised at acceptance) contains:

- `experiments/v1_*` through `v11_*`: all measurement scripts.
- `experiments/v9_modern_arch/v9_5c_imagenet_resnet50_fromscratch_results.json`: full ImageNet trajectory.
- `tests/test_v8_v9_v10_results_smoke.py`: 12 smoke tests asserting headline $T_1/T_3$ values are within ±50 % of the published estimates (catches silent JSON corruption).
- `scripts/build_trimmed.sh`: deterministic submission PDF builder; gates on page count, anonymisation, and external-vendor-name leak grep.
- `docs/preregistration_v2.md`: pre-registered hypotheses (5 falsifiers, 3 confirmed + 2 narrowed), locked before measurement.
- `docs/h200_cluster_runbook.md`: H200 deployment recipe for production-scale rerun.
- `docs/governance.md`: 5-layer governance design for the rfn-mind serving stack (related companion repo).

`bash run_all.sh` regenerates every experiment that fits on a single RTX 3080 in ~3 hours. The H200-cluster-scale experiments (full ImageNet ViT-Large from scratch) are pre-registered with locked predictions but not yet measured; this is a clear scope boundary.

The published bundle also includes:
- `docs/claim_manifest.json`: SHA-256-pinned weights for all pretrained checkpoints.
- `experiments/v5_0_dichotomy_stats/`: full statistical rigging (Mann–Whitney + Bonferroni + B = 10 000 bootstrap + LOSO).
- `experiments/v6_0_mechanism/`: pooling-error bound numerical verification (Proposition 2).

## 5. Suggested action editors

Action editors with relevant expertise on the paper's mechanism + statistics axes:

- **Surya Ganguli** (Stanford) — neural network theory, NTK, dynamical systems.
- **Suriya Gunasekar** (MSR) — implicit regularisation and learning theory.
- **Behnam Neyshabur** (Google) — generalisation theory and architectural understanding.
- **Levent Sagun** (FAIR) — empirical analysis of deep network optimisation landscapes (FIM, Hessian).

If TMLR's matching system selects a different editor, the paper does not depend on any single editor's specific expertise — the empirical evidence and the closed-form mechanism are independently checkable.

## 6. Honest gaps (declared upfront)

- **No proof of substrate-class universality.** The substrate-class claim is *empirical and panel-bounded* — we tested 12 substrate classes and observed complete rank separation, but we make no claim about substrate classes outside the panel (Mixture-of-Experts, MLP-Mixer, selective-gating SSM variants are all currently on the conjectural side).
- **Hanin–Nica's theorem is an existing result.** Our contribution is its application to the FIM diagonal, the closed-form Theorem 1' that converts depth-linear log-variance to a $\sqrt{L}$ tier-ratio prediction with computable prefactor, and the empirical extension across 5 substrate classes at $R^2 \geq 0.94$. We do not claim a new theorem; we claim a new application + a new identity.
- **GPT-Tiny attention does not follow $\sqrt{L}$.** We report this as an honest narrowing of the mechanism's reach, not as evidence against the dichotomy magnitude (which survives in attention).
- **ImageNet ViT-Large from-scratch is pre-registered but not measured.** ~120 H100-GPU-hours; the spend was held until a reviewer specifically requests it.
- **Reverse implication is conjectural.** Our claim is "deep-sequential ⇒ $T_1/T_3 > 100$"; we do not claim the converse.

## 7. Author identity

Anonymous during TMLR review. The repository, paper bundle, and arXiv preprint will deanonymise at acceptance. The author has no conflict of interest with any of the suggested action editors above.

---

*TMLR submission cover letter, V11.5, 2026-05-01.*
