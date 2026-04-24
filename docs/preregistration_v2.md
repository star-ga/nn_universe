# Preregistration — V2 Paper Predictions for External Compute

**STARGA, Inc.**
**Date of registration: 2026-04-24**
**Commit at registration: see `git log` for the most recent `main` commit at this date.**
**Purpose:** lock in, prior to external (H200 cluster / ELT-HIRES / archival) execution, the exact quantitative predictions the paper is making, so the predictions cannot drift post-hoc.

---

## 1. Scope

This preregistration covers three classes of prediction whose execution requires compute or data beyond the local workstation:

- **A.** Cluster-scale σ_min completion (H200, ~1 hour).
- **B.** 20-seed σ_min robustness across the existing V3.0 ladder (H200, ~4 hours).
- **C.** α-drift real-data analysis on archival UVES + HIRES × SDSS DR18 environmental density (no new compute; depends on data access).

Each section below states the prediction, the exact test, the decision rule, and the experimental metadata (seeds / widths / partition / probe count) to prevent post-hoc tuning.

---

## 2. Predictions

### A — Cluster σ_min at W = 45 000

**Script to run:** `experiments/v1_2_scaling/sigma_min_validation.py --widths 14000 22000 45000`.
**Expected runtime on H200:** ~60 min at W=45000 (full SVD on 45000×45000 matrix, float32, CPU fallback if GPU SVD unstable).
**Seed:** 42 (matching the existing V3.0 ladder).

**Prediction.** Interior-layer $\sigma_{\max}/\sigma_{\min}$ ratio at W=45000 should lie in the $10^4$–$10^6$ band (point prediction $5 \times 10^5$), consistent with extrapolation from the V1.2 ladder fit $\sigma_\text{ratio} \sim N^{0.516}$ (V1.1 NTK upper bound is $N^{1/2}$).

**Decision rule.**
- PASS: bootstrap 95 % CI of the interior-max ratio is within $[10^4, 10^7]$.
- AMBIGUOUS: CI straddles $10^7$ (finite-width excursion).
- FAIL: CI entirely below $10^4$ or above $10^8$ — paper's extrapolation is wrong by >1 order of magnitude, needs explanation.

No data exists at W=45000 as of 2026-04-24; this prediction is binding.

### B — 20-seed σ_min robustness at W ∈ {14000, 22000, 45000}

**Script to run:** same script, with `--seeds 0..19`.
**Expected runtime on H200:** ~5 hours (3 widths × 20 seeds × 5 interior layers, ~1 min/layer at W=22000, ~15 min/layer at W=45000).

**Prediction.** Seed CV of the FIM $T_1/T_3$ ratio (not SV) should decrease monotonically with $N$:

| W | N | Predicted CV | Source of prediction |
|---|---|--------------|----------------------|
| 14 000 | 589 M | ≤ 2 % | V3.0 measured (1.51 %) |
| 22 000 | 1.45 B | ≤ 1.5 % | V1.2 + V3.0 ladder trend |
| 45 000 | 6.08 B | ≤ 1.2 % | 3-seed pilot gave 1.2 % |

**Decision rule.**
- PASS: all three CVs below their predicted upper bounds.
- FAIL: any CV exceeds 2 × its bound.

### C — α-drift on archival UVES + HIRES × SDSS DR18

**Script to run:** `experiments/v3_1_alpha/real_data_pipeline.py` (stub exists; see §4).
**Data:** archival UVES + HIRES absorption-spectrum samples of ≥ 200 quasar sightlines with pre-computed $\Delta\alpha/\alpha$ from Webb / King / Murphy / Carswell. SDSS DR18 environmental density $\rho_I$ per sightline at absorber redshift.

**Prediction (V3.1 §1.3, physical κ):** at the tiny physical coupling $\kappa \approx 4 \times 10^{-59}$ yr⁻¹ bit⁻¹ Mpc³, the signal is $\sim 25$ orders of magnitude below the UVES+HIRES 2025 instrumental floor. Therefore the V3.1 real-data test **cannot detect a positive signal** — it can only produce a null result consistent with either κ-vanishing or the signal being below noise.

**Decision rule.**
- If partial-correlation $r(\Delta\alpha/\alpha, \log\rho_I \mid z, S/N) > 0$ at >3σ: paper explicitly credits a positive detection to the framework **only if** a separately-computed instrument-systematics null (same pipeline on random sightline pairings) is at least 2σ lower. Otherwise it is noise.
- If $r \le 0$ at 95 % CL: null result, consistent with paper's stated expectation (physical κ is too small).
- Framework **survives** the test only under the explicit caveat that the test is currently underpowered by 25 orders of magnitude and real falsification requires either cluster-core amplification (V3.3 now ruled out to 4 orders of precision by Earth-clock limits) or ELT-HIRES (~2028 first light).

This section's decision rule is asymmetric by design — a positive claim from the current UVES+HIRES data would be suspicious, not a win.

---

## 3. Locked conventions

The following conventions are **locked for the submission** and cannot be retuned in response to cluster results:

- **Partition:** top 1 % / middle 49 % / bottom 50 %, applied to the FIM diagonal.
- **FIM estimator:** per-sample gradient-squared, $F_{ii} = \mathbb{E}_x[(\partial_{\theta_i}\mathcal L(x))^2]$, accumulated in float64.
- **Training:** SGD with momentum=0.9, lr=1e-3, batch=128, 2 000 steps, MSE self-prediction (V3.0 protocol).
- **Seed:** 42 is the canonical single-seed; robustness runs use seeds 0..19.
- **Activation:** ReLU (V6.5 confirms GELU/tanh/Swish give the same qualitative scaling, different prefactor).
- **Width=45000 architecture:** 5 × `Linear(45000 → 45000)` with ReLU, stem `Linear(32 → 45000)`, head `Linear(45000 → 32)`.

---

## 4. α-drift real-data pipeline

A stub `experiments/v3_1_alpha/real_data_pipeline.py` will be created at the time this document is registered. Its status is **placeholder** — actual execution requires access to the UVES+HIRES and SDSS DR18 archives, which is not currently in scope.

The pipeline's protocol (§2.C above) is fully specified in advance: column definitions, partial-correlation estimator, null-test construction, σ-threshold. Any changes between this registration and the actual execution are logged in-document.

---

## 5. Reproducibility guarantee

All of §2's predictions can be reproduced by:

1. Running `run_all.sh` on this commit (covers local-scale experiments).
2. Running `experiments/v1_2_scaling/sigma_min_validation.py` with the `--widths 14000 22000 45000` arguments on an H200-class machine (covers §2.A + §2.B).
3. Running `experiments/v3_1_alpha/real_data_pipeline.py` with the archival-data paths (§2.C, when available).

All numerical predictions in this document are bound to **the commit hash of the repo at the moment of registration**, not to any subsequent change. Subsequent commits may refine the paper's framing but cannot retroactively change §2's decision rules.

*STARGA Commercial License. Preregistration v2 (V2 paper).*
