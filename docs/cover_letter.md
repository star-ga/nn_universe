# Cover letter — FIM tier hierarchy as a signature of deep layered sequential computation

**Target venue:** NeurIPS 2026 main track. Fallback: NeurIPS 2026 workshop on foundations of deep learning, or ICLR 2027.
**Nature of the work:** empirical + mechanism-backed universality claim about the Fisher-information diagonal of deep networks and deeply-composed non-learning substrates.
**Contact:** Nikolai Nedovodin, STARGA Inc., nikolai@star.ga
**Git commit:** see `main` HEAD of https://github.com/star-ga/nn_universe at the time of submission.

---

## 1. What the paper claims

The **three-tier FIM diagonal hierarchy** — a ratio $T_1/T_3$ between the mean Fisher-information diagonal value of the top-1 % and bottom-50 % of parameters — is *empirically* a signature of **deep layered sequential computation** as a computational primitive, *mechanistically* a consequence of the Hanin–Nica 2020 log-normal theorem for products of random Jacobians, and *verifiably* separates a 12-system substrate panel into two groups with **complete rank separation** ($p = 1.7 \times 10^{-17}$, rank-biserial $r = 1.000$).

## 2. Why it matters

Most universality results in deep learning are tied to the *training regime* (NTK, $\mu P$, scaling laws) or to *architecture-specific* properties (ResNet invariants, attention scaling). The tier-hierarchy signature is different: it survives when training is removed (Kaiming init already gives $T_1/T_3 = 10^3$–$10^4$), when learning is removed (random boolean circuits give $10^7$–$10^8$), and when the substrate family is changed (tested across MLP, CNN, ViT, transformer, and binary tensor-networks). At the same time it is *absent* from every non-deep system we tested — four shallow parameterised learners that do learn and generalise, two lattice gauge field theories (U(1) abelian and SU(2) non-abelian), three dynamical-system controls, and a random-matrix ensemble.

The mechanism is a known theorem (Hanin & Nica, Comm. Math. Phys. 376, 2020, arXiv:1812.05994) re-applied to the FIM diagonal: log-normal $F_{ii}$ with variance growing linearly in depth gives $\log(T_1/T_3) \propto \sqrt{L}$. We confirm this scaling empirically across four independent substrate classes (MLP untrained $R^2 = 0.98$, MLP trained $R^2 = 0.94$, random boolean circuits $R^2 = 0.98$, transformers $R^2 = 0.97$), pinning what had been a phenomenological observation to a published random-matrix-theory result.

## 3. What is genuinely new

The closest prior art we are aware of:

- **Karakida, Akaho, Amari (AISTATS 2019 + Neural Comp. 2021)** — characterise the FIM spectrum of deep networks as "long-tailed" / "pathological" but do not define a tier partition, do not test non-NN substrates, and do not identify a dichotomy with a necessary substrate-class condition.
- **Papyan (ICML 2019)** — reports three-level outlier structure in the *Hessian* (class means + cross-class covariances), not in the FIM diagonal across all layers.
- **Hanin & Nica (2020)** — the product-of-random-matrices log-normal theorem we apply. Their paper is about the output-gradient norm; we extend the application to the FIM diagonal and to non-neural substrates.

The novel contributions are:

1. The **12-substrate panel** — including boolean circuits (no gradients), tensor networks (no real-valued weights), and two lattice gauge theories — gives the first **substrate-independent** characterisation of the tier hierarchy's universality class.
2. The **Mann–Whitney complete-rank-separation** statistical treatment ($p = 1.7 \times 10^{-17}$) makes the dichotomy formally testable rather than qualitative.
3. The **cosmological refinement**: if the universe's substrate is an empirically characterisable deep layered sequential computation (Wheeler's It-from-bit, MERA / HaPPY holographic codes, Vanchurin's neural cosmology, or any Turing-machine-like layered model), the FIM tier hierarchy is a *necessary consequence* of the substrate, not a new physical postulate. Conversely, if the substrate is a spatially-parallel quantum field (lattice QCD / QED), the hierarchy does *not* arise — our U(1) and SU(2) measurements empirically confirm this side of the dichotomy.

## 4. Reproducibility

Everything in the paper can be regenerated from commit `HEAD` of `https://github.com/star-ga/nn_universe`:

- `bash run_all.sh` runs every experiment that fits on a single RTX 3080 in a few hours (V1.0 through V8.0, σ_min validation, 104-test pytest suite). Total time: ~3 hours.
- Only the H200-cluster-scale experiments (W = 14000 / 22000 / 45000 NN training) are out of scope; these have a pre-registered runbook (`docs/h200_cluster_runbook.md`) and locked predictions (`docs/preregistration_v2.md` §2.A + §2.B).
- The α-drift prediction is documented with a *null expectation* at current archival-data noise levels (V3.1 §1.3 + Appendix B of preregistration_v2.md).

## 5. Ethics + open science

No human subjects, no sensitive data. All measurements are on parameterised mathematical / computational systems generated with public seeds. The repo is permissive-licensed (STARGA Commercial License, permits non-commercial academic reuse with citation).

## 6. What would change our mind

The paper's central empirical claim is falsifiable in three concrete ways, pre-registered in `docs/preregistration_v2.md`:

1. **Cluster σ_min at W=45000** outside the $10^4$–$10^7$ band at H200 cluster scale.
2. **Any substrate in the deep-layered-sequential class** giving a bootstrap CI that crosses below 100, or any spatially-parallel / shallow system giving a CI crossing above 100.
3. **Independent replication** of the Hanin–Nica log-normal mechanism failing to give $R^2 > 0.8$ on the $\log(T_1/T_3) \propto \sqrt{L}$ prediction in any substrate class.

Each of these would require significant revision or retraction.

## 7. Suggested reviewers

Anyone with expertise in one of:

- Fisher information geometry of deep networks (Amari-school).
- Random-matrix theory of products of random matrices (Hanin / Nica / O'Donnell).
- Tensor networks / holographic codes (Swingle / Vidal / Pastawski).
- Neural tangent kernel / feature learning regimes (Jacot / Yang / Chizat).

---

*Cover letter v1, 2026-04-24. STARGA Commercial License.*
