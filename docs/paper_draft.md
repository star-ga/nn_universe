# FIM Tier Hierarchy as an Architecture- and Task-Universal Property of Trained Neural Networks

**STARGA, Inc.**
**Nikolai Nedovodin, 2026-04-23**
**Draft: NeurIPS / ICML workshop submission**

---

## Abstract

**Empirical dichotomy.** We test the three-tier Fisher Information Matrix (FIM) diagonal hierarchy — first observed in a 296k-parameter cosmology toy experiment — across **12 parameterised substrate classes** (separately counting MLP / CNN / ViT, plus four shallow learners, plus boolean circuits, plus U(1) and SU(2) lattice gauge fields, plus Ising / harmonic / cellular-automaton dynamical controls, plus a random-matrix ensemble) spanning 13 widths ($1.9 \times 10^3 \le N \le 6.1 \times 10^9$), 4 tasks (self-prediction, toric-code decoding, symbolic regression, supervised vision classification), and 4 architecture families (MLP, CNN, ViT, transformer). The $T_1/T_3$ tier ratio separates into two groups with complete rank separation: bootstrap 95 % CIs are entirely above $100$ for every deep layered sequential system (trained NN, untrained NN, random boolean circuits) and entirely below $100$ for every non-deep system (four shallow learners with CI upper bounds below $6$; two lattice gauge theories, three dynamical systems, one random-matrix ensemble, all with CI upper bounds below $84$). A one-sided Mann–Whitney $U$ test on per-seed $\log T_1/T_3$ values gives $p = 1.7 \times 10^{-17}$ with rank-biserial $r = 1.000$ ($n_{\text{deep}} = 46$, $n_{\text{rest}} = 50$).

**Mechanism.** The dichotomy is quantitatively predicted by Hanin & Nica (Comm. Math. Phys. 376, 287–322, 2020): the log of the squared gradient norm of a deep random network is asymptotically Gaussian with variance $\sigma^2 L$ linear in depth. Log-normal quantile analysis of the resulting $F_{ii}$ distribution yields $\log(T_1/T_3) \propto \sqrt{L}$. We confirm both predictions empirically: a 7-depth × 5-seed sweep on untrained ReLU MLPs gives $\mathrm{Var}[\log F_{ii}] \propto L$ at $R^2 = 0.906$ and $\log(T_1/T_3) \propto \sqrt{L}$ at $R^2 = 0.983$; a width sweep confirms the theorem's width-independence prediction. A 5-depth × 3-seed sweep on *strictly layered random boolean circuits* (no neurons, no real weights, no gradients) gives the same scaling: $\mathrm{Var}[\log F] \propto L$ at $R^2 = 0.961$ and $\log(T_1/T_3) \propto \sqrt{L}$ at $R^2 = 0.980$. The universality class is therefore **deep layered sequential composition** as a computational primitive, confirmed substrate-independently.

**Cosmological relevance.** The tier hierarchy is (i) already present at random Kaiming initialisation with $T_1/T_3 \sim 10^3$–$10^4$, and (ii) gradient-descent training *reduces* it by a factor of 4–24× rather than creating it. Combined with the mechanism, this refines the FIM–Onsager cosmology program's substrate specification from "learning neural network" to "deep layered recursive computation," which includes MERA-style tensor networks, circuit-complexity / bulk-volume constructions, and Wheeler's It-from-bit as admissible, and excludes spatially-parallel quantum-field substrates (lattice QCD / QED) as inadmissible — the latter empirically confirmed by our U(1) measurement at $T_1/T_3 = 1.6$ (CV 0.3%) over 16 384 link phases. Whether the universe's substrate falls inside the admissible class is an open physical question; our work establishes that the tier signature is a necessary condition with a mechanism-backed test.

**Keywords:** Fisher Information Matrix, NTK, neural-network cosmology, information geometry, universality, FIM–Onsager correspondence, universality class, power-law scaling.

---

## 1. Introduction

The Fisher Information Matrix (FIM) is the unique (Chentsov, 1982) invariant Riemannian metric on the statistical manifold of a parameterized probabilistic model. For a neural network with parameters $\theta$ and a Gaussian-likelihood interpretation of its output, the FIM diagonal distribution encodes which parameter directions are maximally informative about the data distribution — and, equivalently, which directions are frozen under gradient-flow dynamics.

The V1.0 FIM–Onsager cosmology experiment (Nedovodin, 2026) measured the FIM diagonal distribution of a 5-layer 256-neuron ReLU MLP (296k params) trained on a self-prediction task, and observed a sharp *three-tier* structure: the top 1% of FIM-diagonal values dominated the middle 49% by a factor of ~13, and dominated the bottom 50% by a factor of 637×. In the FIM–Onsager correspondence, these three tiers were identified with "physical constants" (Tier 1, rarely change), "coupling constants" (Tier 2, drift slowly), and "gauge degrees of freedom" (Tier 3, flow freely).

**Terminology note (V4.3):** throughout this paper we use "FIM diagonal hierarchy" for what earlier iterations called "FIM eigenvalue hierarchy". We measure only the diagonal of the empirical FIM, $F_{ii} = \mathbb{E}[(\partial_{\theta_i}\ell)^2]$. The full eigenvalue spectrum of the empirical FIM (for any finite sample size) is rank-deficient and dominated by numerical zero modes (`experiments/v4_2_fim_spectrum_validation/`); the diagonal is the stable, non-trivial observable we use throughout.

**Partition-convention note (V4.3):** the three-tier partition (top 1% / middle 49% / bottom 50%) is a naming convention inherited from V1.0, not a spectral feature. Reported $T_1/T_3$ magnitudes vary by up to 5 orders of magnitude across plausible partition choices (`experiments/v4_3_statistics/tier_partition_sensitivity.py`). All qualitative universality claims in this paper survive partition choice in direction; magnitudes should be read relative to the 1%/50% convention.

This paper tests the empirical robustness of that three-tier claim along five axes:

1. **Scale**: does the tier structure persist from $10^3$ to $10^9$ parameters?
2. **Seed**: is the tier ratio stable across random initialisations?
3. **Task**: does the hierarchy appear under unrelated learning objectives?
4. **Architecture**: is it MLP-specific, or does it appear in CNN and Transformer architectures?
5. **Computation type**: does the hierarchy require learning? require neurons? require probabilistic inference? require gradient descent? We test four non-deep parameterised learners (linear / kernel ridge / logistic / GP), one non-learning layered-sequential system (random-gate boolean circuits), one spatially-parallel QFT substrate (U(1) pure-gauge lattice), and three dynamical-system controls (Ising, harmonic chain, cellular automaton).

We find: yes (1), increasingly yes with $N$ (2), yes in form across 4 tasks (3), yes across MLP/CNN/ViT (4), and for (5): **the hierarchy appears if and only if the system performs deep layered sequential composition**. It appears in boolean circuits (layered, non-learning, non-probabilistic) at $T_1/T_3 \sim 10^7$. It is absent from four shallow parameterised learners that *do* learn and generalise ($T_1/T_3 \le 5$), absent from a U(1) lattice gauge field at 16k parameters ($T_1/T_3 = 1.6$, CV 0.3%), absent from an SU(2) non-abelian lattice gauge field at 972 parameters ($T_1/T_3 = 4.85$, CV 3.1%), and absent from three dynamical-system controls. A follow-up (V4.1) further shows the hierarchy is already present at random Kaiming init and that training *decreases* it — so neither learning nor optimisation creates the signature. A separate mechanism experiment (V6.0, §4.6) confirms that $\log(T_1/T_3) \propto \sqrt{L}$ with $R^2 = 0.98$ across 7 depths, matching the Hanin–Nica (2020) log-normal prediction for products of random Jacobians. Together these establish the FIM tier hierarchy as a universality class of **deep layered sequential composition** — a computational primitive that need not involve learning, neurons, real-valued weights, or probabilities.

## 2. Related work

- **Fisher information in deep learning.** Amari (1998) developed natural-gradient descent using the FIM; Kirkpatrick et al. (2017) used FIM-weighted regularisation to mitigate catastrophic forgetting. Karakida, Akaho & Amari (AISTATS 2019, arXiv:1806.01316) characterise the FIM spectrum of deep networks at large width as "long-tailed" with a small number of very large outliers. Their follow-up (Karakida et al., Neural Comp. 2021, arXiv:1910.05992) names the spectrum "pathological" for deep architectures. Pennington & Worah (NeurIPS 2018) give an exact free-probability characterisation for one hidden layer. Papyan (ICML 2019, arXiv:1901.08244) reports three-level hierarchical outlier structure in the deepnet Hessian spectrum (driven by class means / cross-class covariances); our structure is related but is in the FIM diagonal across all layers, not in the Hessian's outlier block.
- **Product-of-random-matrices log-normal.** Hanin & Nica (Comm. Math. Phys. 376, 287–322, 2020; arXiv:1812.05994) prove that for a depth-$L$ random-weight ReLU network the log of the gradient-norm squared is asymptotically Gaussian with mean $\mu L$ and variance $\sigma^2 L$. This is the mechanism behind our V5.0 dichotomy: log-normal $F_{ii}$ with depth-linear variance gives $\log(T_1/T_3) \propto \sqrt{L}$ by log-normal quantile analysis. We confirm this empirically in V6.0 (§4.6, $R^2 = 0.98$).
- **Neural Tangent Kernel.** Jacot, Gabriel & Hongler (2018) established the NTK continuum limit for wide networks; Yang (2019) extended to arbitrary tensor-program architectures. Yang & Hu (ICML 2021, arXiv:2011.14522) prove a dynamical-dichotomy theorem between NTK ("lazy") and feature-learning ($\mu P$) regimes. Our V4.1 result (training dissipates the hierarchy 4–24×) is consistent with the NTK $\to \mu P$ flattening, but the hierarchy's appearance at random init predates any training-regime distinction. We use the NTK upper bound $\alpha \leq 1/2$ on the SV exponent.
- **Scaling laws.** Kaplan et al. (2020) and Hoffmann et al. (2022) reported loss-vs-$N$ scaling laws in language models; our work is on *spectral* (not loss) scaling.
- **Lottery tickets.** Frankle & Carbin (ICLR 2019) identify sparse magnitude-based subnetworks that, retrained from the same init, match full-network accuracy. Our tier-1 parameters are FIM-selected (sensitivity), not magnitude-selected, and appear before any training; the two phenomena are orthogonal, but both indirectly assert that a small subset of parameters carries most of the architectural information.
- **Neural-network cosmology + holographic precursors.** Vanchurin (Entropy 22, 2020, arXiv:2008.01540) argued that general relativity and quantum mechanics emerge as the near-equilibrium dynamics of a learning neural network. Swingle (PRD 86, 065007, 2012) and Pastawski–Yoshida–Harlow–Preskill (JHEP 2015) give tensor-network / holographic-code precedents for layered recursive computational structure in spacetime. Susskind–Brown (PRD 97, 086015, 2018) argue circuit complexity grows with physical volume. V1.0 operationalised a testable subset (FIM–Onsager correspondence in the restricted FC class); this paper provides the empirical foundation and (V6.0) the random-matrix-theory mechanism.

## 3. Setup

### 3.1 Tasks

- **T1 – Cosmology self-prediction (V1.0):** MSE self-reconstruction of 32-d Gaussian inputs. Architecture-agnostic baseline.
- **T2 – QEC toric-code decoding (V2.1):** Binary cross-entropy on syndrome-to-correction mapping for a $L=5$ toric code at physical error rate $p = 0.05$.
- **T3 – Symbolic regression (V3.0 task-3):** MSE on recovery of degree-8 random-polynomial coefficients from 16 evaluation pairs.
- **T4 – Supervised vision classification (V3.0 task-4):** 10-class classification on 1024-dimensional Gaussian inputs with fixed random teacher; cross-entropy loss.

### 3.2 Architectures

- **MLP:** 5-layer 256-neuron ReLU, ~300k parameters (unless scaled).
- **SmallCNN:** 4-block conv encoder + mirror deconv decoder, 1.38M params at base_ch=32.
- **SmallViT:** ViT-Tiny variant, patch=4, embed_dim=192, depth=4, heads=3, 1.81M params.

### 3.3 Measurements

- Singular-value ratio $\sigma_\max / \sigma_\min$ per interior 2-D weight matrix.
- FIM diagonal by per-sample backward: $F_{ii} = \mathbb{E}[(\partial_{\theta_i} \ell)^2]$.
- Three-tier partition: top 1% ("Tier 1"), 1–50% ("Tier 2"), bottom 50% ("Tier 3"), with reported quantity $F_1 / F_3$.
- Coefficient of variation (CV) across seeds at fixed architecture and task.

### 3.4 Compute

All compute on consumer hardware (RTX 3080 for $N \leq 2 \times 10^8$) + Runpod A100 80GB community cloud for $N \in [6 \times 10^8, 1.45 \times 10^9]$. Total cloud compute: ~3.5 GPU-hours (~$5 USD).

## 4. Results

### 4.1 Scaling (V1.0 + V1.2 + V3.0 = 12 widths, $N \in [1.9 \times 10^3, 1.45 \times 10^9]$)

$$
\text{SV} \sim N^{0.516},\qquad R^2 = 0.86 \text{ (full sweep)}.
$$

Interior-fit (widths $\geq 64$, 10 points): $\alpha = 0.473 \pm 0.093$, consistent with the NTK upper bound $1/2$ (Yang, 2019). The full-sweep exponent is a mixture of feature-learning at small $n$ and lazy-training at large $n$; see the separate note *V1.1 gap closure* for a detailed analysis.

FIM T1/T3 remains in the 150–616× band across 8 orders of magnitude in $N$, without power-law scaling in the cosmology task.

### 4.2 Seed robustness (20-seed sweeps at $N = 3.2 \times 10^6$ and $5 \times 10^7$)

| Width | $N$ params | SV CV | FIM T1/T3 CV |
|-------|-----------|-------|---------------|
| 256 | 2.1 × 10⁵ | 124% | 10% |
| 1,024 | 3.2 × 10⁶ | 66% | **4.96%** |
| 4,096 | 5.0 × 10⁷ | 249% | **2.81%** |
| 14,000 | 5.9 × 10⁸ | 109% | **1.51%** |

**FIM tier CV drops monotonically with $N$.** The SV ratio CV is non-monotone and remains order-of-magnitude (60–250%).

### 4.3 Task universality

| Task | $\beta$ (FIM exp. @ 300 probes) | $\beta$ (FIM exp. @ 2000 probes) | Probe sensitivity |
|------|----------------------------------|------------------------------------|---------------------|
| T1 cosmology self-prediction | ≈ 0 | ≈ 0 (values change < 3%) | **NONE** |
| T2 QEC toric-code decoding | 1.386 (R²=0.93) | **2.258 (R²=1.00)** | yes, $+0.87$ |
| T3 symbolic regression | 1.432 (R²=0.94) | **2.299 (R²=0.91)** | yes, $+0.87$ |
| T4 vision classification | 2.748 (R²=0.90) | **5.546 (R²=0.995)** | yes, $+2.80$ |

Power-law form present in all four tasks. **A sharp separation emerges between unstructured and structured tasks:**

- **T1 cosmology self-prediction** (unstructured, Gaussian-noise reconstruction): FIM T1/T3 is flat in $N$ *and* probe-count-insensitive. Values at W=256/512/1024 change by ≤ 3% between 300 and 2000 probes (149.6 → 144.8; 247.8 → 241.5; 335.3 → 329.0). Exponent stays at ≈ 0.
- **T2, T3, T4** (structured): FIM T1/T3 grows super-linearly in $N$, and the exponent *increases* with probe count. At 300 probes the MC noise in the tier-3 mean biases the ratio downward; at 2000 probes the true exponent emerges.

Structured-task exponents at 2000 probes cluster in the 2.3 – 5.5 band, with T4 (supervised classification) showing the steepest scaling. The probe-count sensitivity is itself a *feature*, not a bug: it reveals that the tier-3 distribution is genuinely heavy-tailed for structured tasks (the smallest FIM values are very small and require fine MC estimation) but uniform for unstructured tasks (tier-3 values are typical Gaussian-noise order, easily estimated at 300 probes).

### 4.4 Architecture universality (at $N \approx 1.5 \times 10^6$)

| Arch | SV ratio | FIM T1/T3 |
|------|----------|-----------|
| MLP | 1,674× | 2,808× |
| CNN | 60× | 2,312× |
| ViT | 1,378× | **121,670×** |

CNN's convolutional filters have intrinsically low-rank weight matrices (SV ~60×, much lower than MLP/ViT), but the FIM hierarchy is present in all three and in the same order of magnitude or higher.

### 4.5 Non-learning controls (V4.0 uniqueness)

Six parameterized systems at matched parameter scale ($N \approx 3\text{k}$):

| System | FIM T1/T3 | CV across seeds |
|--------|-----------|-----------------|
| Deep NN (trained, MLP / CNN / ViT, matched $N$) | **10²–10⁴×** | 3–110% |
| Boolean circuit (random gates, layered sequential, non-learning) | **10⁷–10⁸×** | 146–172% |
| Deep NN (untrained, Kaiming init) | **10³–10⁴×** | — |
| Linear regression (shallow learner) | **1.10×** | low |
| Kernel ridge regression (shallow learner) | **1.42×** | low |
| Logistic regression (1-layer softmax learner) | **3.14×** | low |
| Gaussian process (non-parametric learner) | **1.97×** | low |
| U(1) pure-gauge lattice (L=8, 16 384 link phases) | **1.6×** | 0.3% |
| SU(2) pure-gauge lattice (L=3, 972 params, non-abelian) | **4.85×** | 3.1% |
| Ising chain | 2.6× | 10% |
| Harmonic chain | 5.0× | 28% |
| Cellular automaton (Rule 110) | 3.8× | 20% |
| Random matrix (GOE) | 104× | 5% |

**Sharp empirical dichotomy.** Systems that perform deep layered sequential computation (≥ 4 hidden layers, trained or untrained, neural networks *or* random boolean circuits) produce tier ratios bounded *below* by $10^2$: log-bootstrap 95 % CIs are $[244,\ 468]$ for pooled trained NNs, $[2\,723,\ 5\,115]$ for pooled untrained NNs, and $[3\,739,\ 4.1 \times 10^6]$ for random boolean circuits. Every other system we tested — four shallow parameterised learners (linear, kernel ridge, logistic, GP), *both* gauge groups of our lattice test (U(1) abelian and SU(2) non-abelian), three dynamical-system controls, and a random-matrix ensemble — has a 95 % CI entirely below $100$, with all four shallow learners' upper bounds below $6$. A one-sided Mann–Whitney $U$ test on the per-seed $\log T_1/T_3$ values yields $p = 1.7 \times 10^{-17}$ and rank-biserial $r = 1.000$: every deep-sequential observation ranks above every non-deep observation (complete separation, $n_{\text{deep}} = 46$, $n_{\text{rest}} = 50$). The boolean-circuit result is the decisive data point — no neurons, no real-valued weights, no gradients, no probabilistic structure, and no training, yet its FIM diagonal hierarchy matches or exceeds a trained ViT. The universality class is **deep layered sequential composition**, not neural networks, not learning, not optimisation. See Appendix A for the full bootstrap + Mann–Whitney methodology.

### 4.6 Mechanism — log-normal Jacobian product (V6.0)

The dichotomy of §4.5 is quantitatively explained by a published random-matrix-theory theorem:

> **Theorem (Hanin & Nica 2020, Comm. Math. Phys. 376, 287–322).** For a depth-$L$ fully-connected ReLU network with i.i.d. Gaussian weights and width $n$, as $L, n \to \infty$ the log of the squared gradient norm $\log \|\partial \mathcal{L}/\partial x\|^2$ converges in distribution to a Gaussian with mean $\mu L$ and variance $\sigma^2 L$, where $\mu, \sigma$ depend only on the nonlinearity and weight ensemble.

Applied to the FIM diagonal of a parameter $\theta_i$ in layer $\ell$, the downstream Jacobian chain has length $L - \ell$; so $\log F_{ii}$ is approximately Gaussian with variance $2\sigma^2 (L-\ell)$, i.e. log-normal $F_{ii}$ with depth-linear spread. Log-normal quantile analysis gives
\[
\log(T_1/T_3) \;\approx\; 3.47 \, \sigma \, \sqrt{2 L} \;\propto\; \sqrt{L}.
\]

We test this empirically (`experiments/v6_0_depth_mechanism/depth_sweep.py`) with 7 depths $L \in \{2,3,4,6,8,12,20\}$ × 5 seeds on untrained ReLU MLPs at width 64, dim 16, 1000 FIM probes. Observed (seed mean):

| $L$ | $N$ params | $T_1/T_3$ | $\mathrm{Var}[\log F]$ | Skew | Excess kurt. |
|----:|-----------:|----------:|-----------------------:|-----:|-------------:|
|  2 |  2 128 | $2.0 \times 10^{1}$ |  0.71 | $+0.34$ | $-0.79$ |
|  3 |  6 288 | $1.1 \times 10^{2}$ |  2.12 | $-1.35$ | $+6.86$ |
|  4 | 10 448 | $6.2 \times 10^{2}$ |  4.35 | $-1.51$ | $+5.58$ |
|  6 | 18 768 | $4.1 \times 10^{4}$ |  8.15 | $-1.10$ | $+4.47$ |
|  8 | 27 088 | $6.87 \times 10^{6}$ | 12.40 | $-0.23$ | $+2.62$ |
| 12 | 43 728 | $3.65 \times 10^{12}$ † | 30.05 | $+0.67$ | $+0.54$ |
| 20 | 77 008 | $9.53 \times 10^{15}$ | 107.0 | $+0.69$ | $-0.64$ |

† At $L = 12$ the per-seed range is $4.8 \times 10^9 – 1.6 \times 10^{13}$ (one outlier seed dominates); the 5-seed mean is the reported $3.65 \times 10^{12}$.

Three falsifiable predictions:

- **(H1) $\mathrm{Var}[\log F_{ii}] \propto L$.** OLS over 7 depths: slope $5.726$, $R^2 = 0.906$. **PASS.**
- **(H2) $\log(T_1/T_3) \propto \sqrt{L}$.** OLS: slope $11.51$, $R^2 = 0.983$. **PASS.**
- **(H3) Approximate log-normality at $L \geq 6$.** Excess kurtosis drops from $+4.47$ at $L{=}6$ to $+2.62$ at $L{=}8$ and toward Gaussian ($0$) as $L$ grows; $|\mathrm{skew}|$ is non-monotone but bounded. Asymptotic log-normality consistent with the large-$L$ theorem.

A width sweep (`experiments/v6_0_depth_mechanism/width_sweep.py`, widths $16, 32, 64, 128, 256$ at fixed $L{=}8$) confirms the **tier-ratio** width-independence predicted by the theorem: $\log(T_1/T_3)$ is flat with width (slope $-0.013$ per width-unit; H6 PASS). The raw $\mathrm{Var}[\log F]$ shows a finite-width correction (decreases from 18.0 at width 16 to 9.7 at width 256 as the Gaussian approximation to the log-normal sharpens and tier-3 tail moves out of the finite-sample underflow zone); the *slope* of Var[log F] vs $L$ is the width-independent quantity, not Var[log F] itself at fixed $L$. The load-bearing prediction is the $T_1/T_3$ width-independence, which holds to within $\pm 1\%$.

A trained-NN depth sweep (`experiments/v6_0_depth_mechanism/trained_depth_sweep.py`, same protocol as V6.0 but with 10 000 SGD training steps between init and measurement, 6 depths × 5 seeds) shows that the $\sqrt{L}$ scaling law *survives training*: trained-network $\log(T_1/T_3) \propto \sqrt{L}$ with $R^2 = 0.936$, slope $11.78$ (compared to the untrained 6-depth slope $10.98$ and the V6.0 7-depth untrained slope $11.51$). The slope itself is not reduced by training in this fit window — the functional form and prefactor persist. What training *does* reduce is the Var[$\log F$] accumulation coefficient, by roughly 5 % in the OLS slope (2.72 trained vs 2.86 untrained). The practical consequence is depth-dependent dissipation of the $T_1/T_3$ ratio: at $L \leq 6$ the factor untrained/trained runs 2.2–20.7× across seeds (mean 3–11× per depth), but at $L \geq 8$ it collapses to $\approx 1$×, and at $L = 12$ roughly half the seeds actually show *trained > untrained*. The Hanin–Nica tails become structurally locked at moderate depth: SGD noise is too small relative to a log-normal with $\sigma^2 L \gtrsim 10$ to re-randomise it. This is the quantitative upgrade of V4.1's "training dissipates 4–24×" claim, which was measured at $L = 5$ (inside the dissipation regime).

A layered boolean-circuit depth sweep (`experiments/v6_0_depth_mechanism/bc_depth_sweep.py`, strictly layered random-gate circuits with softmax mixtures of {AND, OR, XOR} and 16 gates/layer, 5 depths × 3 seeds, 200 probes) tests whether the mechanism extends to substrates with no real-valued weights, no gradients, and no probabilistic structure. Both hypotheses pass: $\mathrm{Var}[\log F_{ii}] \propto L$ at slope $2.01$, $R^2 = 0.961$; $\log(T_1/T_3) \propto \sqrt{L}$ at slope $4.09$, $R^2 = 0.980$. The mechanism is therefore *substrate-independent* within the deep-sequential class — depth + composition suffice, the substrate can be any layered composition of random local maps.

A transformer depth sweep (`experiments/v6_0_depth_mechanism/transformer_depth_sweep.py`, pre-norm Transformer blocks with multi-head self-attention + GELU MLP + residual stream, $d_{\text{model}} = 32$, 6 depths × 3 seeds, 500 FIM probes) tests whether the mechanism survives attention softmax and residual connections. The tier-ratio scaling $\log(T_1/T_3) \propto \sqrt{L}$ **passes** with $R^2 = 0.969$ (slope $0.44$), though the underlying $\mathrm{Var}[\log F] \propto L$ prediction is borderline ($R^2 = 0.79$, slope $0.10$) — the residual stream partially decouples the per-layer log-variance accumulation. Net transformer $T_1/T_3$ ranges 96–210 across $L = 1$ to $8$, all within the deep-sequential band. The mechanism's tier-ratio prediction is therefore universality-class-preserving even for modern attention architectures; only the per-layer variance accumulation constant is damped.

The V5.0 empirical dichotomy is therefore no longer phenomenology: deep layered sequential systems have log-normal $F_{ii}$ with depth-linear variance, producing exponential-in-$\sqrt{L}$ tier ratios. Spatially-parallel and shallow systems have no depth-composition chain, so the log-variance stays $O(1)$ and the tier ratio stays $O(1)$. See `docs/v6_0_mechanism_hanin_nica.md` and Appendix B for the full derivation.

## 5. Discussion

### 5.1 Summary of the empirical claim

The three-tier FIM diagonal hierarchy is:

- **Task-universal in form**: a power-law scaling $F_1/F_3 \sim N^\beta$ exists for structured tasks (T2, T3) with task-dependent $\beta$; flat in the unstructured self-prediction task (T1).
- **Architecture-universal**: present in MLP, CNN, and ViT at comparable parameter counts.
- **Seed-stable and scale-improving**: FIM T1/T3 CV drops from 10% at $N = 2\times 10^5$ to 1.51% at $N = 6 \times 10^8$.
- **Deep-layered-sequential-specific**: present in boolean circuits ($10^7$–$10^8$) despite having neither learning nor gradients; absent from four shallow parameterised learners (linear / kernel / logistic / GP) despite their being genuine learning systems; absent from a U(1) lattice gauge field despite having 16,384 parameters; absent from three dynamical-system controls; and modest ($\sim 100$) in random matrices. The signature tracks depth + compositionality, not optimisation.

### 5.2 Theoretical framing

**Upper-bound compatibility.** The FIM tier hierarchy is compatible with the NTK continuum-limit upper bound $\alpha \le 1/2$ on the SV ratio (Jacot et al., 2018), which our interior-fit value $0.473 \pm 0.093$ respects. The fuller V1.0 → V3.0 dataset and its interior-fit at multiple cutoffs is given in the supplementary gap-closure note.

**Log-normal mechanism (V6.0).** The Hanin–Nica 2020 theorem gives the core mechanism directly (§4.6): product of random Jacobians $\to$ log-normal gradient norm with depth-linear variance. This is the closest thing to a "theorem" that the paper provides for the dichotomy. Its predictions for *untrained* MLPs ($R^2 = 0.906$ on Var[log F] $\propto L$; $R^2 = 0.983$ on $\log T_1/T_3 \propto \sqrt{L}$) pass quantitatively. Training (V4.1) does *not* remove the log-normal shape; it reduces the per-layer variance coefficient $\sigma$ by factor 4–24, corresponding to a reduction in tier ratio but preservation of the $\sqrt{L}$ functional form.

**Large-$N$ convergence.** Beyond NTK, the monotone-with-$N$ stabilisation of FIM CV (10% → 1.51%) is suggestive of a thermodynamic-like limit in the parameter manifold: at finite $N$ the tier fractions $f_1, f_2, f_3$ fluctuate across seeds, but as $N \to \infty$ they appear to converge to well-defined values $(0.01, 0.49, 0.50)$ respectively. A formal large-$N$ theorem for the tier fractions themselves remains open.

### 5.3 Relevance to neural-network cosmology

The V1.0 FIM–Onsager correspondence (Nedovodin, 2026) hypothesised that the tier hierarchy maps onto the physical-law / coupling-constant / gauge-DOF distinction in cosmology. Our results establish that this hierarchy is (a) an intrinsic property of deep layered sequential computation (not specifically of learning or of neural networks), and (b) robust across task and architecture within that class (§4.3, §4.4). The natural refinement of the V1.0 specification is therefore **"the substrate performs deep layered recursive composition"** rather than "the substrate learns." This includes neural networks as one instance, but also admits boolean circuits and any Turing-machine-like layered computation as alternatives; it excludes spatially-parallel quantum fields (lattice QCD / QED), shallow parameterised learners, and ordinary dynamical systems. Within this specification the FIM tier hierarchy is *empirically satisfied*. Whether the universe's substrate falls inside this class remains an open cosmological question.

### 5.4 Limitations

- Parameter count explored: $10^3$ to $10^9$. Extrapolating tier invariance to cosmological scales ($10^{120+}$) remains conjectural.
- Spacetime emergence (4D + Lorentz signature) is not addressed empirically; remains an open theoretical question in the parent framework.
- The SV power-law exponent is *noisy* (CV 60–250%); interpretations must not over-rely on it. The FIM tier ratio is the robust observable.
- T3 (symbolic regression) final loss is 0.526 (trivial baseline 1.0); the task is imperfectly learned but sufficient to probe the FIM structure of a trained network.

## 6. Conclusion

**Empirical dichotomy** (§4.5). Across 10 parameterised substrates, the FIM 3-tier diagonal ratio $T_1/T_3$ separates two classes by 2–6 orders of magnitude with complete rank separation ($p = 5.1 \times 10^{-17}$, Mann–Whitney $U$). Deep layered sequential systems (MLP, CNN, ViT — trained or untrained — and random boolean circuits) have bootstrap 95 % CIs entirely above $100$. Four genuine shallow learners (linear / kernel / logistic / GP), a U(1) lattice gauge field, three dynamical-system controls, and a random-matrix ensemble all have CIs entirely below $100$. The boolean-circuit data point (no learning, no gradients, no probabilities, only layered composition) makes the dichotomy substrate-independent within its class.

**Mechanism** (§4.6). The dichotomy is quantitatively predicted by Hanin & Nica (2020, Comm. Math. Phys. 376), who prove that the log squared gradient norm of a deep random network is asymptotically Gaussian with variance linear in depth. Log-normal quantile analysis then gives $\log(T_1/T_3) \propto \sqrt{L}$; our 7-depth × 5-seed measurement confirms both the linear $\mathrm{Var}[\log F_{ii}] \propto L$ prediction ($R^2 = 0.906$) and the $\sqrt{L}$ tier-ratio prediction ($R^2 = 0.983$), and a width sweep confirms the width-independence predicted by the theorem. The FIM 3-tier hierarchy is therefore a structural consequence of the random-matrix-theory of products of Jacobians through deep layers — not a phenomenological observation requiring a new theorem.

**What remains open.** The FIM–Onsager cosmology program's substrate specification sharpens from "learning neural network" to "deep layered sequential composition." Whether the universe's substrate falls inside this class (MERA-style holographic tensor networks, circuit-complexity-bulk correspondence, Wheeler's It-from-bit, Vanchurin's neural cosmology — all consistent; lattice QFT / QCD / QED as a generic spatially-parallel gauge field — inconsistent) is an open physical question that our empirical work does not settle. What our work does settle is that satisfying the "tier-hierarchy signature" is a necessary condition for the V1.0 FIM–Onsager correspondence to apply, and that this necessary condition is empirically available on every testable deep-sequential substrate without requiring any training, learning, or gradient-based optimisation.

Theoretical closure of remaining items — a large-$N$ tier-fraction theorem, the 4D emergence argument of Nedovodin (2026), and a Lorentzian-signature derivation — are left open for future work.

---

## Code and data

All scripts, result JSONs, and the full computational log are public at
`https://github.com/star-ga/nn_universe`, reproducible from
`run_all.sh`.

## Appendix A — Bootstrap + Mann–Whitney procedure (§4.5)

Per-seed $T_1/T_3$ values were extracted from the JSON result files of
the 10 systems listed in §4.5, totalling $n = 93$ observations across
12 systems. For each system we computed the sample mean of $\log T_1/T_3$
across its seeds and resampled the per-seed values 2 000 times with
replacement ( `numpy.random.default_rng(42)` ), recording the resampled
mean each time. The 95 % confidence interval is the empirical
$[2.5\%, 97.5\%]$ percentiles of the resampled means, exponentiated back
to the $T_1/T_3$ scale. Log-bootstrap was used rather than linear
bootstrap because the underlying $T_1/T_3$ distribution is heavy-tailed;
the log-transform is the standard treatment (Efron & Hastie, 2016).

Group separation was tested by a one-sided Mann–Whitney $U$ statistic
on the per-seed $\log T_1/T_3$ values, comparing the 46 observations in
the deep-sequential group (trained NN × 4 widths × 5 seeds + untrained
NN × 4 widths × 5 seeds + boolean-circuit × 6 seeds) against the 50
observations in the rest group (GP × 5 + kernel × 5 + logistic × 5 +
linear × 5 + U(1) lattice × 3 + SU(2) lattice × 3 + Ising × 6 +
harmonic × 6 + CA × 6 + random-matrix × 6). Implementation:
`scipy.stats.mannwhitneyu(deep, rest, alternative="greater")`. Rank-biserial correlation $r$ is the effect-size proxy $r = 2U / (n_1 n_2) - 1$.

Full script: `experiments/v5_0_dichotomy_stats/dichotomy_stats.py`.
Raw outputs: `experiments/v5_0_dichotomy_stats/dichotomy_stats_results.json`.

## Appendix B — Log-normal FIM → tier ratio derivation (§4.6)

Hanin & Nica (2020) prove that, for an $L$-layer ReLU network with i.i.d.
Gaussian weights and width $n$, as $L, n \to \infty$ the log squared
gradient norm $\log \|\partial \mathcal{L}/\partial x\|^2$ converges in
distribution to a Gaussian with mean $\mu L$ and variance $\sigma^2 L$.
The constants $\mu, \sigma$ depend only on the activation and weight
ensemble and are explicitly computable from the moments of
$(\phi'(W^T x))^2$.

For a parameter $\theta_i$ in layer $\ell$, the per-sample loss gradient
factors as
$\partial \mathcal{L} / \partial \theta_i = g_\ell(x) \cdot J_{\ell \to L}(x)$,
where $g_\ell$ is the local "direct" gradient at layer $\ell$ and
$J_{\ell \to L} = \prod_{k=\ell+1}^{L} W_k D_k$ is the downstream Jacobian
chain. Applying Hanin–Nica to the chain of length $L - \ell$ gives
$\log \|J_{\ell \to L}\|^2 \sim \mathcal{N}(\mu(L-\ell), \sigma^2(L-\ell))$;
the FIM diagonal is then the sample average of the squared gradient, and
a further expectation over samples gives
\[
\log F_{ii} \;\sim\; \mathcal{N}\bigl(\mu_0 + 2\mu(L-\ell), \;\; 2\sigma^2(L-\ell)\bigr).
\]

A log-normal random variable with parameters $(m, v)$ has $\alpha$-quantile
$q_\alpha = \exp(m + \sqrt{v} \Phi^{-1}(\alpha))$. The top-1% mean
$\mathbb{E}[F \mid F > q_{0.99}]$ lies at mean standard-normal score
$z_{\mathrm{top}} \approx 2.67$ (the mean of the standard-normal distribution
conditional on being above its 99th percentile), and the bottom-50% mean
lies at $z_{\mathrm{bot}} \approx -0.80$ (mean below the median). Thus
\[
\frac{T_1}{T_3} \;\approx\; \exp\!\left(\sqrt{v} (z_{\mathrm{top}} - z_{\mathrm{bot}})\right)
= \exp\!\left(3.47\,\sqrt{v}\right).
\]

Substituting the per-layer-averaged Hanin–Nica variance $v = 2\sigma^2 L$
(averaging $\ell$ over $\{0, \ldots, L-1\}$ gives $v \propto L$ with a
proportionality fixed by the activation):
\[
\boxed{\; \log(T_1/T_3) \;\approx\; c \,\sigma\, \sqrt{L}, \quad c \approx 3.47\sqrt{2} \approx 4.9. \;}
\]

V6.0's measured slope of $11.5$ at $\sigma \approx 1.69$ gives
$c \approx 11.5 / 1.69 = 6.8$, within 39 % of the derivation's value $c
\approx 4.9$. The deficit is attributable to (i) finite width (the
theorem is asymptotic in $n$), (ii) the tier partition being a simple
top/bottom cut rather than an exact log-normal quantile expectation, and
(iii) higher-order $O(\sigma^4)$ corrections at moderate depth.

## References

[1] N. Nedovodin. "The Universe as a Self-Organizing Neural Network." STARGA Inc., April 2026.

[2] S. Amari. "Natural Gradient Works Efficiently in Learning." *Neural Computation* 10, 251 (1998).

[3] A. Jacot, F. Gabriel, C. Hongler. "Neural Tangent Kernel: Convergence and Generalization in Neural Networks." *NeurIPS* 2018.

[4] G. Yang. "Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes." *NeurIPS* 2019.

[5] J. Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks." *PNAS* 114, 3521 (2017).

[6] V. Vanchurin. "The World as a Neural Network." *Entropy* 22, 1210 (2020).

[7] J. Kaplan et al. "Scaling laws for neural language models." *arXiv:2001.08361* (2020).

[8] J. Hoffmann et al. "Training Compute-Optimal Large Language Models." *arXiv:2203.15556* (2022).

[9] N. N. Chentsov. *Statistical Decision Rules and Optimal Inference.* AMS (1982).

[10] D. Lovelock. "The Einstein Tensor and Its Generalizations." *J. Math. Phys.* 12, 498 (1971).

[11] B. Hanin, M. Nica. "Products of Many Large Random Matrices and Gradients in Deep Neural Networks." *Comm. Math. Phys.* 376, 287–322 (2020). [arXiv:1812.05994](https://arxiv.org/abs/1812.05994).

[12] R. Karakida, S. Akaho, S. Amari. "Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach." *AISTATS* 2019. [arXiv:1806.01316](https://arxiv.org/abs/1806.01316).

[13] R. Karakida et al. "Pathological Spectra of the Fisher Information Metric and Its Variants in Deep Neural Networks." *Neural Comp.* 33(8) (2021). [arXiv:1910.05992](https://arxiv.org/abs/1910.05992).

[14] V. Papyan. "Measurements of Three-Level Hierarchical Structure in the Outliers in the Spectrum of Deepnet Hessians." *ICML* 2019. [arXiv:1901.08244](https://arxiv.org/abs/1901.08244).

[15] G. Yang, E. Hu. "Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks." *ICML* 2021. [arXiv:2011.14522](https://arxiv.org/abs/2011.14522).

[16] L. Chizat, F. Bach. "On Lazy Training in Differentiable Programming." *NeurIPS* 2019. [arXiv:1812.07956](https://arxiv.org/abs/1812.07956).

[17] J. Frankle, M. Carbin. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." *ICLR* 2019. [arXiv:1803.03635](https://arxiv.org/abs/1803.03635).

[18] B. Swingle. "Entanglement Renormalization and Holography." *Phys. Rev. D* 86, 065007 (2012). [arXiv:0905.1317](https://arxiv.org/abs/0905.1317).

[19] A. R. Brown, L. Susskind. "The Second Law of Quantum Complexity." *Phys. Rev. D* 97, 086015 (2018). [arXiv:1701.01107](https://arxiv.org/abs/1701.01107).

[20] F. Pastawski, B. Yoshida, D. Harlow, J. Preskill. "Holographic Quantum Error-Correcting Codes: Toy Models for the Bulk/Boundary Correspondence." *JHEP* 06 (2015). [arXiv:1503.06237](https://arxiv.org/abs/1503.06237).

[21] B. Efron, T. Hastie. *Computer Age Statistical Inference.* Cambridge (2016).

---

*STARGA Commercial License.*
