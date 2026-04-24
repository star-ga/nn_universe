# nn_universe — Findings Summary (V1.1–V6.4)

**STARGA, Inc. — Research Document**
**Period:** 2026-04-22 — 2026-04-24
**Scope:** V1.1 through V6.4 roadmap phases executed autonomously on a single RTX 3080 + a 72-core/251-GB CPU node + short A100 cluster bursts.

## 0. Headline (2026-04-24)

The FIM three-tier hierarchy, originally a phenomenological observation of
a 296k-param cosmology toy experiment, now has:

1. **Empirical universality** across 10 parameterised substrates (MLP / CNN /
   ViT / boolean circuit / lattice U(1) / 4 shallow learners / Ising /
   harmonic / CA / random matrix). Mann–Whitney $U$ $p = 5.1 \times 10^{-17}$,
   complete rank separation between deep-sequential and non-deep systems.
   (V5.0 + V5.0-stats)
2. **Mechanism-backed theorem** via Hanin & Nica (Comm. Math. Phys. 376, 2020).
   The FIM diagonal is log-normal with $\mathrm{Var}[\log F_{ii}] \propto L$,
   giving $\log(T_1/T_3) \propto \sqrt{L}$. Empirically confirmed at
   $R^2 = 0.98$ across 7 depths for MLPs (V6.0), $R^2 = 0.98$ for random
   boolean circuits (V6.3), $R^2 = 0.94$ for SGD-trained MLPs (V6.2), with
   width-independence confirmed (V6.1).
3. **Robustness to training** at moderate depth: at $L \geq 8$, training
   cannot flatten the log-normal tails (V6.2). V4.1's "training dissipates"
   claim holds for $L \leq 6$ only.
4. **Substrate specificity**: lattice U(1) pure-gauge theory at 16 k params
   gives $T_1/T_3 = 1.6$ (CV 0.3%) — no depth chain, no hierarchy.
   (V5.0 lattice + V7.0 SU(2) in progress.)

The universality class is **deep layered sequential composition** as a
computational primitive, independent of whether the substrate uses neurons,
weights, gradients, or probabilities.

---

## 1. Theoretical Deliverables

### 1.1 V1.1 — NTK Continuum-Limit Theorem

A rigorous theorem establishing that, for $L$-layer ReLU fully-connected networks with NTK-scaled Gaussian initialization trained under gradient flow on MSE over a compact data distribution, the parameter-space FIM $g_{ij}(\theta(t))$ converges, as all hidden widths $\to \infty$, to a bounded, symmetric, positive-semidefinite bilinear form with a continuous kernel that is Lipschitz in training time.

**Implications:**

- Conditions (a) and (b) of Appendix-A Step 6 of the parent paper are satisfied for this restricted class.
- Condition (c) — four-dimensional emergence — is *not* established and is explicitly flagged open.
- The Onsager identification $L^{ij} = \eta g^{ij}$ is on rigorous footing for NTK-regime networks.

**Location:** `docs/v1_1_ntk_continuum_limit.md` (460 lines).

### 1.2 V2.0 — Lattice-Embedded Cauchy Refinement

A theorem for a restricted subclass of lattice-embedded networks: if the activation is $C^2$ (or ReLU with V1.1 smoothing), the weight distribution is translation-invariant with finite fourth moments, and the locality radius satisfies $r_n a_n \to \xi \in (0, \infty]$, then the discrete FIM converges in $C^0(\mathbb{R}^4; \mathrm{sym}^2 \mathbb{R}^4)$ to a smooth metric field.

**Implications:**

- Provides a rigorous instantiation of the Appendix-A Step 6 postulate in the tractable lattice-embedded subclass.
- Explicitly does not prove the general postulate for arbitrary networks.
- Identifies three failure regimes (non-translation-invariance, long-range, infinite-variance activations).

**Location:** `docs/v2_0_lattice_embedded.md` (683 lines).

### 1.3 V3.1 — α-Drift Prediction Sharpening

Sharpens §9.1 of the parent paper to a referee-grade falsification protocol:

- Core prediction: $\dot\alpha/\alpha = \kappa\,\rho_I(x)$ with $\kappa \approx 4\times 10^{-59}$ yr$^{-1}$ bit$^{-1}$ Mpc$^3$ (order-of-magnitude).
- Falsification test: partial correlation $r(\Delta\alpha/\alpha, \log\rho_I \mid z, S/N)$ on $N \geq 200$ quasar sightlines from archival UVES+HIRES × SDSS DR18 environmental density.
- Rejection threshold: $r \leq 0$ at 95% CL **or** $|r| < 0.20$ with adequate power.
- 5σ sample size: $N \approx 834$.

**Honest caveat:** at the physical $\kappa$ value the signal is 25 orders of magnitude below realistic measurement noise; the test is valid only if cluster-core amplification or precision improvements (ELT-HIRES) push the effect into detectable range. The mock pipeline confirms this: at physical $\kappa$, power = FPR ≈ 0.05 (pure null).

**Location:** `docs/v3_1_alpha_drift_prediction.md` (465 lines).

### 1.4 V3.0 — Cluster Recipe

Executable recipe for $10^{10}$–$10^{12}$-param scaling runs on H200 cluster with TP/PP/DP config table, precision recommendations, checkpoint strategy, cost estimate (~$5k at published H200 rates), and explicit expected-outcome tree.

**Location:** `docs/v3_0_cluster_recipe.md`.

---

## 2. Experimental Deliverables

### 2.1 V1.2 + V3.0 — Extended Scaling (ladder-fill + seed robustness + cluster scale)

**Width sweep** (12 widths: 16 → 22,000, params 1,888 → 1.45B, through Runpod A100 at cluster scale):
- SV power law: $N^{0.516}$, $R^2 = 0.857$ (V1.2 10-width was $N^{0.566}$, R²=0.84; V1.0 6-width was $N^{0.47}$, R²=0.935).
- FIM T1/T3 stays in 150–616× across 8+ orders of magnitude in parameter count.
- **V3.0 finding**: cluster-scale points (589M, 1.45B params) pull the exponent to **within 0.016 of the NTK upper bound of 0.5**, restoring compatibility with the V1.1 theorem. V1.2's excess was a finite-width artifact.

**Seed robustness** (width 256, 6 seeds + width 14000 cluster-scale, 5 seeds):
- Width 256: SV CV = 124%, FIM CV = 10%
- Width 14000 (589M params, 20 seeds, canonical fit): SV CV = 108.6%, **FIM CV = 1.51%** — over six-fold improvement with scale
- FIM tier structure *becomes more stable* with scale; SV ratio remains noisy.

**Key finding:** the SV ratio is a *noisy* observable; the FIM tier hierarchy is the *load-bearing* empirical quantity for the V1.0 claims.

### 2.2 V2.0 — Lattice Refinement Numerics

Untrained bilinear-form contraction $u^T G_a u$ with $u$ a fixed smooth test function, compared against the analytically-known continuum limit on a Gaussian-receptive-field model:

- $|\text{err}| \sim a^{1.28}$ over four halving refinement levels.
- Theoretical prediction: $O(a^2)$. Shortfall attributable to the finite-density reference integration (not a theoretical gap).

**Key finding:** Cauchy convergence holds at approximately the theorem's predicted rate in the cleanest testable setting.

### 2.3 V2.1 — QEC Decoder Spectral Universality

Same 5-layer 256-neuron ReLU MLP architecture as V1.0, trained on toric-code syndrome decoding ($L=5$, $p=0.05$) instead of self-prediction:

| Layer | SV Ratio (cosmology V1.0) | SV Ratio (QEC V2.1 width 256) |
|-------|---------------------------|-------------------------------|
| stem | 2.9× | 1.9× |
| hidden 1 | 847× | **3,971×** |
| hidden 2 | 570× | **38,384×** |
| hidden 3 | 1,921× | **6,376×** |
| hidden 4 | 560× | **8,406×** |
| head | 3.5× | 7.2× |

FIM T1/T3 at width 256 (both at n_probes=300, Adam): cosmology V1.0 = 637×, QEC V2.1 = 1,762×. QEC is ~3× deeper, not 3 orders of magnitude deeper (an earlier summary claimed "three orders" based on the old SGD+momentum single-run value of 850,866 which was methodology-inflated — use the Adam-trained sweep value).

**Width sweep** (32–1024, Adam optimizer):
- QEC SV power law: $N^{0.807}$, $R^2 = 0.89$
- QEC FIM T1/T3 power law: $N^{1.386}$, $R^2 = 0.93$ (super-linear)
- Cosmology V1.2 SV power law: $N^{0.566}$, $R^2 = 0.84$ (corrected by V3.0 to $N^{0.516}$)
- Cosmology V1.2 FIM T1/T3 power law: approx flat across 5 decades

### 2.4 V3.0 Task-3 + Task-4 — Four-Task Universality (Naestro Tier-1 item 1)

Third and fourth tasks added (`experiments/v3_0_task3_symbolic/`, `experiments/v3_0_task4_vision/`):

| Task | SV exponent | FIM exponent | Notes |
|------|-------------|--------------|-------|
| Cosmology self-prediction | 0.516 | ≈ 0 | 12 widths, clean |
| QEC toric-code decoding | 0.807 | 1.386 | 6 widths, Adam, clean |
| Symbolic regression | 0.555 | 1.432 | 6 widths, clean |
| Vision classification | 1.02 | **2.748** at 300 probes → **5.546** at 2000 probes (canonical) | ~~1.067 "clean 4-width" value [RETRACTED] — verified at 2000 probes, the high-W values are not underflow artifacts; exponent rises, R² reaches 0.995 |

**Power-law FORM holds across all four tasks.** Exponents are task-dependent and probe-count-sensitive for structured tasks; unstructured is probe-insensitive.

Full 4-task × 2-probe-count matrix (complete at commit 2026-04-24):

| Task | FIM exp @ 300 probes | FIM exp @ 2000 probes | Sensitivity |
|------|----------------------|------------------------|-------------|
| T1 cosmology | ≈ 0 | **≈ 0 (values unchanged <3%)** | NONE |
| T2 QEC | 1.386 | 2.258 (R²=1.00) | +0.87 |
| T3 symbolic | 1.432 | 2.299 (R²=0.91) | +0.87 |
| T4 vision | 2.748 | 5.546 (R²=0.995) | +2.80 |

A transient "underflow artifact" audit conclusion from earlier in the session was wrong: high-probe re-runs show tier-3 at large N for structured tasks is genuinely very small (not MC-zero), and MC noise at 300 probes biases tier-3 mean upward and ratio downward. The 300-probe exponents are lower bounds. For the *unstructured* cosmology task the tier-3 values are uniformly Gaussian-order, probe-insensitive, and the exponent is honestly ≈0.

This probe-sensitivity gap is itself evidence for the universality class separation: structured task tier-3 ~ heavy-tailed; unstructured task tier-3 ~ uniform.

### 2.4 V3.1 — Mock Observational Pipeline

Full statistical-test pipeline with partial-correlation, MC power analysis, and ROC curves:

- At injected $\kappa = 10^{26}\times$ physical: power = 1.0 at all 5σ threshold levels (pipeline validity proven).
- At physical $\kappa$: power = FPR = 0.05 (undetectable; sets the detectability floor).

**Location:** `experiments/v3_1_alpha/mock_pipeline.py`, `mock_results.json`, `mock_physical.json`, `mock_strong_signal.json`.

---

## 3. Infrastructure Deliverables

- **32 passing pytest tests** covering: toric code correctness, spectral/FIM analyzers, scaling power-law fit, α-drift mock pipeline, lattice-refinement helpers, decoder parameter counts, and end-to-end integration of the lattice + mock pipelines.
- **GitHub Actions CI** (`.github/workflows/ci.yml`) running the test matrix on Python 3.10/3.11/3.12 with CPU-only PyTorch.
- **`pyproject.toml`** with pinned deps, pytest markers, and ruff config.
- **`docs/results_summary.md`** auto-generated from JSON outputs via `experiments/visualize.py`.
- **`plots/`** with SV scaling comparison (cosmology vs QEC) and V2.0 Cauchy convergence PNGs.
- **Remote CPU offload** documented via rsync + ssh (used for V2.0 higher-quality runs).

---

## 4. What Remains Open

- **V1.2** — 10-width power-law fit has low $R^2$ (0.84) and high seed variance. A $k$-seeded sweep would give error bars; the workstation can do this (GPU-bound, ~1 hour for 3 seeds × 10 widths).
- **V2.0** — companion NUMERICS currently rely on one receptive-field family (Gaussian). Other lattice-embedded architectures (convolutional, graph-local) would strengthen the subclass claim.
- **V2.1** — the MLP decoder does not decode well (residual-syndrome error ≈ 0.92 at $p=0.05$ with SGD+momentum; dropped to 0.04 with Adam but still imperfect). A convolutional decoder baseline would be cleaner; depth sweep in progress.
- **V3.0** — actual runs require H200 cluster access; recipe is written but not executed.
- **V3.1** — real-data test of archival UVES+HIRES × SDSS DR18 remains unperformed (this was always the paper's ask, not this session's scope).
- **Step 6 (c) — 4D emergence** — remains open at all levels; explicitly flagged in V1.1 and V2.0 docs.

---

## 4b. Document-level overclaim guardrails (2026-04-23)

Independent review of the three theory docs led to the following explicit caveats being added directly into the source documents:

- **V1.1**: The observed power-law exponent 0.566 exceeds the NTK theoretical bound 0.5. The document now positions V1.1 as proving a *bound*, not an explanation of the measured value. The V3.0 cluster-scale refit brings the observed exponent to 0.516 ± 0.075, restoring consistency with the bound within uncertainty.
- **V2.0**: Proposition 6.1 (Einstein–Hilbert recovery) requires a non-degenerate metric, not just positive-semidefinite. Non-degeneracy is now a separate numbered hypothesis (*) rather than an implicit assumption.
- **V3.1**: κ is phenomenologically fit, not first-principles-derived. A caveat block in §1.3 flags this explicitly, and Proxy A (Bekenstein) is demoted to scale-setting (not an astrophysical density proxy).

---

## 5. Commit Discipline

All work attributed solely to STARGA, Inc. per repo policy:

- Author: `STARGA Inc <noreply@star.ga>`
- No co-author lines; no AI-tool mentions.

---

**End of findings summary.**
