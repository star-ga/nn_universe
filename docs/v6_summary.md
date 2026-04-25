# FIM Tier Hierarchy: Mechanism Summary

**STARGA, Inc. — Workshop Handout**
**Nikolai Nedovodin, 2026-04-24**

---

## 1. The Observation

We measure the Fisher Information Matrix (FIM) diagonal tier ratio $T_1/T_3$ — the mean of the top-1% of parameter sensitivities divided by the mean of the bottom-50% — across 12 parameterised substrate classes spanning trained / untrained neural networks, random boolean circuits, balanced binary tensor networks, four shallow parameterised learners, U(1) and SU(2) lattice gauge fields, three dynamical-system controls, and a random-matrix ensemble. The ratio separates into two groups by 2–6 orders of magnitude, with complete rank separation across 96 per-seed observations ($p = 1.7 \times 10^{-17}$, Mann–Whitney $U$, rank-biserial $r = 1.000$, $n_{\text{deep}} = 46$, $n_{\text{rest}} = 50$).

This builds directly on prior work showing that deep networks have heavy-tailed FIM spectra (Karakida, Akaho & Amari, AISTATS 2019; Karakida et al., Neural Comp. 2021) and three-level Hessian outlier structure (Papyan, ICML 2019). The V6.0 contribution is (i) extending the measurement to non-neural substrates, (ii) identifying the boundary condition (depth ≥ 4 sequential composition), and (iii) deriving the scaling law from Hanin & Nica (2020).

---

## 2. The Dichotomy Table

**Definitions.** $T_1$ = mean FIM diagonal of the top-1% of parameters; $T_3$ = mean FIM diagonal of the bottom-50%. The three-tier partition (top 1% / middle 49% / bottom 50%) is a naming convention; all qualitative claims survive partition choice in direction. CIs are log-bootstrap, 2 000 resamples, percentile method (Efron & Hastie 2016). Threshold = 100, conservative relative to the observed gap.

*(Source: `experiments/v5_0_dichotomy_stats/dichotomy_stats_results.json` — re-run 2026-04-24 with corrected lattice U(1) gradient + SU(2) added.)*

| System | Type | $n$ seeds | Point est. $T_1/T_3$ | 95% CI | CI > 100? |
|---|---|---:|---:|---|:---:|
| Boolean circuit (N = 384) | deep-sequential | 6 | 88 440 | [3 781, 4 146 835] | YES |
| NN untrained (pooled 4 widths) | deep-sequential | 20 | 3 757 | [2 736, 5 134] | YES |
| NN trained (pooled 4 widths) | deep-sequential | 20 | 337 | [246, 472] | YES |
| Random matrix (GOE, N = 3 003) | rest | 6 | 80.7 | [77.8, 83.7] | no |
| Cellular automaton (Rule 110) | rest | 6 | 3.77 | [3.33, 4.39] | no |
| Harmonic oscillator chain | rest | 6 | 3.57 | [2.87, 4.50] | no |
| Logistic regression | rest | 5 | 3.13 | [2.89, 3.35] | no |
| Ising chain (1D) | rest | 6 | 2.54 | [2.35, 2.74] | no |
| SU(2) lattice gauge (L = 3, non-abelian) | rest | 3 | 4.85 | [4.67, 4.96] | no |
| U(1) lattice gauge (L = 8) | rest | 3 | 1.62 | [1.61, 1.62] | no |
| Gaussian process | rest | 5 | 1.97 | [1.94, 1.99] | no |
| Kernel ridge regression | rest | 5 | 1.42 | [1.41, 1.43] | no |
| Linear regression | rest | 5 | 1.10 | [1.09, 1.12] | no |

The gap between the deepest "rest" system (random matrix, CI upper bound 83.7) and the shallowest deep-sequential system (trained NN, CI lower bound 246) is a factor of approximately 3 in the worst case, and more than $10^4$ in expectation. No CI overlaps across the boundary.

---

## 3. The Theorem and the Derivation

**The load-bearing prior result.** Hanin & Nica (Comm. Math. Phys. 376, 287–322, 2020; [arXiv:1812.05994](https://arxiv.org/abs/1812.05994)) prove:

> For a depth-$L$ fully-connected ReLU network with i.i.d. Gaussian weights and width $n$, as $L, n \to \infty$ the log squared gradient norm $\log \|\partial \mathcal{L}/\partial x\|^2$ converges in distribution to a Gaussian with mean $\mu L$ and variance $\sigma^2 L$, where $\mu, \sigma$ depend only on the nonlinearity and weight ensemble (not on width).

The proof uses free-probability techniques applied to products of independent random matrices composed with a pointwise nonlinearity; it requires no learning assumption.

**From the theorem to the tier ratio.** For a parameter $\theta_i$ in layer $\ell$, the per-sample gradient factors as local-gradient $\times$ downstream Jacobian chain $J_{\ell \to L} = \prod_{k=\ell+1}^{L} W_k D_k$. Applying Hanin–Nica to that chain of length $L - \ell$:

$$\log F_{ii} \;\sim\; \mathcal{N}\!\bigl(\mu_0 + 2\mu(L-\ell),\;\; 2\sigma^2(L-\ell)\bigr).$$

A log-normal $F_{ii} \sim e^{\mathcal{N}(m, v)}$ has $\alpha$-quantile $q_\alpha = e^{m + \sqrt{v}\,\Phi^{-1}(\alpha)}$. The tier ratio $T_1/T_3$ is the ratio of conditional means in the top-1% and bottom-50% tails. Substituting the tail z-scores ($z_{\text{top}} \approx 2.67$, $z_{\text{bot}} \approx -0.80$) and averaging $\ell$ over layers (giving $v \propto L$):

$$\boxed{\;\log(T_1/T_3) \;\approx\; 3.47\,\sigma\,\sqrt{2L} \;\propto\; \sqrt{L}.}$$

This three-line derivation predicts: (a) $\mathrm{Var}[\log F_{ii}]$ is linear in $L$, and (b) $\log(T_1/T_3)$ is linear in $\sqrt{L}$. Both are falsifiable against data without free parameters (only $\sigma$, which is fixed by the activation).

---

## 4. Empirical Confirmation — Six Experiments, One Theorem

All four experiments test the same prediction on structurally different substrates. All four pass at $R^2 \geq 0.94$.

| Experiment | Substrate | Key prediction tested | $R^2$ | Result |
|---|---|---|---:|:---:|
| V6.0 untrained MLP | ReLU MLP, 7 depths $\times$ 5 seeds, width 64 | $\log(T_1/T_3) \propto \sqrt{L}$ | 0.983 | PASS |
| V6.2 trained MLP | Same architecture, 10 000 SGD steps, 6 depths $\times$ 5 seeds | $\log(T_1/T_3) \propto \sqrt{L}$ survives training | 0.936 | PASS |
| V6.3 boolean circuits | Layered random boolean gates, 5 depths $\times$ 3 seeds | $\log(T_1/T_3) \propto \sqrt{L}$, no neurons, no gradients | 0.980 | PASS |
| V6.4 transformers | Pre-norm Transformer blocks, 6 depths $\times$ 3 seeds | $\log(T_1/T_3) \propto \sqrt{L}$ with attention + residuals | 0.969 | PASS |
| V8.0 binary tensor networks | Balanced binary tree, 8 depths $\times$ 5 seeds, tanh tensor at every node | $\log(T_1/T_3) \propto \sqrt{L}$ on a non-neural MERA-like substrate | 0.992 | PASS |
| V9 ResNet residual stacks | BatchNorm + residual, depths 4/8/16/32, width 128 | $\log(T_1/T_3) \propto \sqrt{L}$ on modern residual architecture | 0.999 | PASS (slope 16.74, T1/T3 reaches $1.3\times10^{38}$ at L=32) |

*(Sources: `experiments/v6_0_depth_mechanism/v6_0_depth_sweep.json`, `v6_2_trained_depth_sweep.json`, `v6_3_bc_depth_sweep.json`, `v6_4_transformer_depth_sweep.json`)*

**V6.0 detail.** Width-64 ReLU MLPs, 7 depths $\in \{2, 3, 4, 6, 8, 12, 20\}$, 5 seeds each, 1 000 FIM probes. Depth-2 networks yield $T_1/T_3 \approx 20$; depth-20 networks yield $T_1/T_3 \approx 10^{16}$. The jump of 15 orders of magnitude occurs because the exponent is $\propto \sqrt{L}$, not $\propto L$, so each doubling of depth multiplies the log-ratio by $\sqrt{2}$. The linear-in-$L$ variance prediction also passes: OLS slope $5.726$, $R^2 = 0.906$.

**V6.2 detail.** Training does not remove the $\sqrt{L}$ law; it reduces the per-layer log-variance coefficient by approximately 5% on average ($R^2 = 0.936$, slope $\approx 11.0$ versus $11.51$ untrained). The dissipation is depth-dependent: at $L \leq 6$ the trained/untrained ratio runs 3–21×; at $L \geq 8$ the dissipation factor collapses to approximately $1\times$ and at $L = 12$ half the seeds show trained $>$ untrained. The log-normal tails become structurally locked once $\sigma^2 L \gtrsim 10$: SGD noise cannot re-randomise a distribution spread over $e^{\sqrt{10}} \approx 23\times$ dynamic range.

**V6.3 detail.** Boolean circuits have no real-valued weights, no neurons, and no gradients in the neural-network sense — the FIM is computed by differentiating a real-valued relaxation of the gate truth-tables with respect to the gate-selection parameters. Despite this, the $\sqrt{L}$ law holds at $R^2 = 0.980$. At depth 8, mean $T_1/T_3 \approx 1900$; at depth 32, mean $T_1/T_3 \approx 10^9$. The universality class is the composition structure, not the neural-network formalism.

**V6.4 detail.** Residual connections in transformers route gradient around the attention/MLP composition, reducing the effective per-layer $\sigma^2$. The direct prediction $\mathrm{Var}[\log F] \propto L$ is only borderline ($R^2 = 0.79$). Yet the tier-ratio scaling $\log(T_1/T_3) \propto \sqrt{L}$ passes clearly at $R^2 = 0.969$: $T_1/T_3$ runs 96 to 210 across $L = 1$ to $8$, all within the deep-sequential band. The scaling law is more robust than the individual variance prediction because it integrates over the layer-averaged spread rather than tracking per-layer increments.

---

## 5. Why the Non-Deep Systems Sit in the O(1–100) Band

The Hanin–Nica mechanism requires a **long sequential chain of composed random maps**. Systems without such a chain do not accumulate log-variance:

**Shallow learners** (linear regression, kernel ridge, logistic regression, Gaussian process): the gradient is a single linear functional of the input — effectively $L = 1$. There is no multiplicative chain to produce a log-normal spread. All four have $T_1/T_3 \leq 6$ with tight CIs. They learn and generalise; they do not develop the hierarchy.

**U(1) lattice gauge** ($L = 8$, $d = 4$, $N = 16\,384$ link phases, $T_1/T_3 = 1.62 \pm 0.005$, CV $= 0.3\%$): the action is a sum over local plaquettes and each link's FIM contribution involves only the six plaquettes containing it. The computation is spatially parallel, not sequentially deep. No chain rule accumulates across the lattice. This is the decisive control for the cosmological argument: a full quantum-field-theoretic substrate with thousands of parameters produces no hierarchy whatsoever. *(Source: `experiments/v5_0_lattice_qcd/v5_0_lattice_u1_results.json`)*

**Ising chain, harmonic chain, cellular automaton** (Rule 110): these 1D dynamical systems have local interactions that do not compose into deep multiplicative chains in the FIM sense. All have $T_1/T_3 \in [2.5, 5.0]$.

**Random matrices (GOE)**: $T_1/T_3 \approx 81$ (CI $[77.8, 83.7]$), the closest non-deep system to the threshold. This arises from the Wigner semicircle eigenvalue distribution — a different mechanism, a one-shot statistical effect rather than a depth-accumulated one — and remains a factor of 3 below even the most conservative deep-sequential CI lower bound.

---

## 6. Cosmological Implication

The FIM tier hierarchy is a **necessary condition** for the substrate to be deep layered sequential: every deep-sequential substrate we tested exceeds $T_1/T_3 \geq 10^2$ with complete statistical separation, and every non-sequential substrate falls below it. Combined with V4.1's finding that training reduces the hierarchy rather than creating it (so the signature is present at random initialisation), this places an empirically testable constraint on any cosmological "universe as computation" programme: if the universe's substrate is sequentially deep (MERA / holographic tensor networks / circuit-complexity constructions), the tier hierarchy follows as a structural consequence; if the substrate is spatially parallel (standard lattice QFT), it does not — as confirmed directly by the U(1) measurement. Whether the universe's substrate falls inside the admissible class is an open physical question; our work establishes the test. The longer argument connecting the tier hierarchy to the FIM–Onsager correspondence and the Boltzmann-weight analogy is in the paper's §5.3 (`docs/paper_draft.md`).

---

## References

1. B. Hanin, M. Nica. *Products of Many Large Random Matrices and Gradients in Deep Neural Networks.* Comm. Math. Phys. 376, 287–322 (2020). [arXiv:1812.05994](https://arxiv.org/abs/1812.05994).
2. R. Karakida, S. Akaho, S. Amari. *Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach.* AISTATS 2019. [arXiv:1806.01316](https://arxiv.org/abs/1806.01316).
3. R. Karakida et al. *Pathological Spectra of the Fisher Information Metric and Its Variants in Deep Neural Networks.* Neural Computation 33(8) (2021). [arXiv:1910.05992](https://arxiv.org/abs/1910.05992).
4. V. Papyan. *Measurements of Three-Level Hierarchical Structure in the Outliers in the Spectrum of Deepnet Hessians.* ICML 2019. [arXiv:1901.08244](https://arxiv.org/abs/1901.08244).
5. B. Swingle. *Entanglement Renormalization and Holography.* Phys. Rev. D 86, 065007 (2012). [arXiv:0905.1317](https://arxiv.org/abs/0905.1317).
6. V. Vanchurin. *The World as a Neural Network.* Entropy 22, 1210 (2020). [arXiv:2008.01540](https://arxiv.org/abs/2008.01540).
7. B. Efron, T. Hastie. *Computer Age Statistical Inference.* Cambridge University Press (2016).

---

*Data lineage: all numerical claims traceable to JSON files in `experiments/v6_0_depth_mechanism/` and `experiments/v5_0_lattice_qcd/` and statistical summary in `experiments/v5_0_dichotomy_stats/dichotomy_stats_summary.md`.*
