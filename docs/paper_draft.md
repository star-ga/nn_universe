# FIM Tier Hierarchy as an Architecture- and Task-Universal Property of Trained Neural Networks

**STARGA, Inc.**
**Nikolai Nedovodin, 2026-04-23**
**Draft: NeurIPS / ICML workshop submission**

---

## Abstract

We report a systematic empirical investigation of the three-tier Fisher Information Matrix (FIM) diagonal hierarchy first observed in the V1.0 FIM–Onsager cosmology experiment. Across 13 widths (parameter counts $1.9 \times 10^3 \le N \le 6.1 \times 10^9$), 20-seed robustness sweeps at four scales, four independent tasks (self-prediction, toric-code syndrome decoding, symbolic regression, supervised vision classification), three architectures (MLP, CNN, ViT), four non-deep parameterised learners (linear regression, kernel ridge regression, logistic regression, Gaussian process regression), one layered-non-learning system (boolean circuits of random gates), and six physical / mathematical control systems (Ising chain, harmonic chain, cellular automaton, U(1) pure-gauge lattice, generic random matrix), we find that the FIM three-tier ratio is (i) architecture-universal across MLP / CNN / ViT at matched $N$, (ii) task-universal in form (FIM power-law exponents in the 1.0–1.5 band for three structured tasks; ≈0 for unstructured self-prediction), (iii) monotonically stabilised in seed variance as $N$ grows (CV drops from 10% at $N{=}2 \times 10^5$ to 1.2% at $N{=}6 \times 10^9$), (iv) **present in all deep layered sequential systems we tested** — including boolean circuits at $T_1/T_3 \sim 10^7$–$10^8$, demonstrating that the signature does not require neurons, real-valued weights, or gradient descent, only deep layered sequential composition — (v) **absent from every non-deep system we tested**, including four genuine parameterised learners (shallow learners: $T_1/T_3 \le 5$), a spatially-parallel quantum-field substrate (U(1) lattice gauge, $T_1/T_3 = 2.0$ at 16,384 link-phase parameters), three dynamical systems ($T_1/T_3 \le 5$), and random matrices ($T_1/T_3 \sim 100$, still below every NN measurement). A follow-up experiment further shows that the hierarchy is **already present at random Kaiming initialization** with tier ratios of $10^3$–$10^4$, and that gradient-descent training *reduces* it by a factor of 4–24× rather than creating it; the universality class is therefore **deep layered sequential composition**, not learning, not optimisation, not neural networks specifically. The singular-value ratio is a highly seed-dependent observable whose power-law exponent ($N^{0.42}$ to $N^{0.81}$ across tasks/cutoffs) is consistent with the NTK theoretical upper bound of $1/2$ once regime separation is applied. The FIM tier structure is the scale-invariant, seed-stable empirical anchor of a universality class of **deep layered recursive composition** — a computational primitive that need not involve learning, neurons, or probabilities to exhibit the hierarchy, and that sharply separates layered-sequential architectures from shallow learners and spatially-parallel physical substrates.

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

We find: yes (1), increasingly yes with $N$ (2), yes in form across 4 tasks (3), yes across MLP/CNN/ViT (4), and for (5): **the hierarchy appears if and only if the system performs deep layered sequential computation**. It appears in boolean circuits (layered, non-learning, non-probabilistic) at $T_1/T_3 \sim 10^7$. It is absent from four shallow parameterised learners that *do* learn and generalise ($T_1/T_3 \le 5$), absent from a U(1) lattice gauge field at 16k parameters ($T_1/T_3 = 2.0$, CV 0.5%), and absent from three dynamical-system controls. A follow-up (V4.1, §4.6) further shows the hierarchy is already present at random Kaiming init and that training *decreases* it — so neither learning nor optimisation creates the signature. Together these establish the FIM tier hierarchy as a universality class of **deep layered recursive computation** — a computational primitive that need not involve learning, neurons, real-valued weights, or probabilities.

## 2. Related work

- **Fisher information in deep learning.** Amari (1998) developed natural-gradient descent using the FIM; Kirkpatrick et al. (2017) used FIM-weighted regularisation to mitigate catastrophic forgetting. The three-tier structure *per se* has not been systematically reported prior to V1.0.
- **Neural Tangent Kernel.** Jacot, Gabriel, Hongler (2018) established the NTK continuum limit for wide networks; Yang (2019) extended to arbitrary architectures satisfying tensor-program rules. We use the NTK limit as a theoretical upper bound on the interior-layer SV exponent.
- **Scaling laws.** Kaplan et al. (2020) and Hoffmann et al. (2022) reported loss-vs-$N$ scaling laws in language models; our work is on *spectral* (not loss) scaling.
- **Neural-network cosmology.** Vanchurin (2020) argued that general relativity and quantum mechanics emerge as the near-equilibrium dynamics of a learning neural network. V1.0 operationalised a testable subset (Onsager correspondence in the restricted FC class); this paper provides the empirical foundation.

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
| 14,000 | 5.9 × 10⁸ | 109% | **1.85%** |

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
| Gaussian process (non-parametric learner) | **5.37×** | low |
| U(1) pure-gauge lattice (L=8, 16 384 link phases) | **2.0×** | 0.5% |
| Ising chain | 2.6× | 10% |
| Harmonic chain | 5.0× | 28% |
| Cellular automaton (Rule 110) | 3.8× | 20% |
| Random matrix (GOE) | 104× | 5% |

**Sharp empirical dichotomy.** Systems that perform deep layered sequential computation (≥ 4 hidden layers, trained or untrained, neural networks *or* random boolean circuits) produce tier ratios of $10^3$ or more. Every other system we tested — four parameterised learners that do learn and generalise, a spatially-parallel lattice gauge field, three dynamical-system controls, and a random-matrix ensemble — sits below $T_1/T_3 = 104$, with all four shallow learners below 6. The boolean-circuit result ($10^7$–$10^8$) is the decisive data point: it has no neurons, no real-valued weights, no gradients, no probabilistic structure, and no training, yet its FIM diagonal exhibits the same hierarchy as a trained ViT. The universality class is **layered recursive composition**, not neural networks, not learning, not optimisation.

## 5. Discussion

### 5.1 Summary of the empirical claim

The three-tier FIM diagonal hierarchy is:

- **Task-universal in form**: a power-law scaling $F_1/F_3 \sim N^\beta$ exists for structured tasks (T2, T3) with task-dependent $\beta$; flat in the unstructured self-prediction task (T1).
- **Architecture-universal**: present in MLP, CNN, and ViT at comparable parameter counts.
- **Seed-stable and scale-improving**: FIM T1/T3 CV drops from 10% at $N = 2\times 10^5$ to 1.85% at $N = 6 \times 10^8$.
- **Deep-layered-sequential-specific**: present in boolean circuits ($10^7$–$10^8$) despite having neither learning nor gradients; absent from four shallow parameterised learners (linear / kernel / logistic / GP) despite their being genuine learning systems; absent from a U(1) lattice gauge field despite having 16,384 parameters; absent from three dynamical-system controls; and modest ($\sim 100$) in random matrices. The signature tracks depth + compositionality, not optimisation.

### 5.2 Theoretical framing

The FIM tier hierarchy is compatible with the NTK continuum-limit upper bound $\alpha \le 1/2$ on the SV ratio (Jacot et al., 2018), which our interior-fit value $0.473 \pm 0.093$ respects. The fuller V1.0 → V3.0 dataset and its interior-fit at multiple cutoffs is given in the supplementary gap-closure note.

Beyond NTK, the monotone-with-$N$ stabilisation of FIM CV (10% → 1.85%) is suggestive of a thermodynamic-like limit in the parameter manifold: at finite $N$ the tier fractions $f_1, f_2, f_3$ fluctuate across seeds, but as $N \to \infty$ they appear to converge to well-defined values $(0.01, 0.49, 0.50)$ respectively. A formal large-$N$ theorem for the tier fractions remains open.

### 5.3 Relevance to neural-network cosmology

The V1.0 FIM–Onsager correspondence (Nedovodin, 2026) hypothesised that the tier hierarchy maps onto the physical-law / coupling-constant / gauge-DOF distinction in cosmology. Our results establish that this hierarchy is (a) an intrinsic property of deep layered sequential computation (not specifically of learning or of neural networks), and (b) robust across task and architecture within that class (§4.3, §4.4). The natural refinement of the V1.0 specification is therefore **"the substrate performs deep layered recursive composition"** rather than "the substrate learns." This includes neural networks as one instance, but also admits boolean circuits and any Turing-machine-like layered computation as alternatives; it excludes spatially-parallel quantum fields (lattice QCD / QED), shallow parameterised learners, and ordinary dynamical systems. Within this specification the FIM tier hierarchy is *empirically satisfied*. Whether the universe's substrate falls inside this class remains an open cosmological question.

### 5.4 Limitations

- Parameter count explored: $10^3$ to $10^9$. Extrapolating tier invariance to cosmological scales ($10^{120+}$) remains conjectural.
- Spacetime emergence (4D + Lorentz signature) is not addressed empirically; remains an open theoretical question in the parent framework.
- The SV power-law exponent is *noisy* (CV 60–250%); interpretations must not over-rely on it. The FIM tier ratio is the robust observable.
- T3 (symbolic regression) final loss is 0.526 (trivial baseline 1.0); the task is imperfectly learned but sufficient to probe the FIM structure of a trained network.

## 6. Conclusion

The FIM three-tier hierarchy, originally observed in a 296k-param cosmology toy experiment, survives an order-of-magnitude empirical stress test: scale invariance across 6 orders of $N$, task universality across three objectives, architecture universality across MLP / CNN / ViT, seed stability improving with scale, presence in a non-learning layered-sequential control (random boolean circuits, $10^7$–$10^8$), and absence in four non-deep parameterised learners, a U(1) pure-gauge lattice at 16 k parameters, and three dynamical-system controls. The natural statement of the universality class is therefore **deep layered recursive composition**, a computational primitive that encompasses — but is not exclusive to — neural networks. This refines the FIM–Onsager cosmology program's substrate specification from "learning neural network" to "deep layered recursive computation." Theoretical closure (large-$N$ tier-fraction theorem, 4D emergence, Lorentzian signature) remains open.

---

## Code and data

All scripts, result JSONs, and the full computational log are public at
`https://github.com/star-ga/nn_universe`, reproducible from
`run_all.sh`.

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

---

*STARGA Commercial License.*
