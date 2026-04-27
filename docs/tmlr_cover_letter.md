# TMLR Submission Cover Letter

**Manuscript:** *Fisher Information Tier Hierarchy: A Panel-Bounded Empirical Regularity of Deep Layered Sequential Computation*
**Submission target:** TMLR (Transactions on Machine Learning Research) — rolling submission, J2C track eligibility for NeurIPS 2026
**Corresponding author:** Nikolai Nedovodin, STARGA, Inc. (`ceo@star.ga`)
**Code repository:** `https://github.com/star-ga/nn_universe`
**Submission date:** to be filled at upload

---

## 1. What the paper claims

We identify a *panel-bounded empirical regularity* of deep layered sequential computation: across a 12-substrate panel spanning $1.9 \times 10^3$ to $6.1 \times 10^9$ parameters, the FIM diagonal tier ratio $T_1/T_3$ separates two groups with **complete rank separation** — bootstrap 95 % CIs are entirely above 100 for every deep-sequential system, entirely below 100 for every non-deep system. The headline statistical result is a one-sided Mann–Whitney $U$ test giving $p = 1.7 \times 10^{-17}$ with rank-biserial $r = 1.000$, complemented by Bonferroni-corrected per-system tests, $B = 10\,000$ bootstrap of the Mann–Whitney p-value (max resample $p = 1.71 \times 10^{-17}$), and a leave-one-substrate-class-out check ($p < 5.8 \times 10^{-13}$ on every removal).

The mechanism is established through a closed-form Theorem 1 (a log-normal-quantile-to-tier-ratio identity) combined with the Hanin & Nica (2020) random-Jacobian log-normal limit. We also prove an exact finite-$v$ form (Theorem 1') and a pooling-error bound (Proposition 2, numerically verified at $L \geq 4$). The mechanism predicts $\log(T_1/T_3) \propto \sqrt{L}$ with prefactor $c \approx 4.90$; we confirm this scaling at $R^2 \geq 0.94$ across five substrate classes (untrained MLP, trained MLP, boolean circuits, vanilla transformers, binary tensor networks).

A pre-registered dynamical-isometry falsifier — proposed in §3.5 of the paper *before* measurement — is confirmed: an identity-perturbation initialisation drops $T_1/T_3$ to $\approx 42$, below the 100 threshold (V10b experiment). This is the cleanest mechanistic check available without an analytic limit and was the specific falsifier the paper itself registered as decisive.

Production-scale verification on canonical pretrained checkpoints (CNN: ResNet-50 V1+V2 at 76.13 % and 80.86 % ImageNet top-1; ViT: ViT-L/16 SWAG-LINEAR at 79.66 %; autoregressive LM: GPT-2-medium on WebText) gives multi-seed bootstrap CIs with std $\leq 0.18$ log units across 5 probe seeds. All five production-scale points sit firmly in the deep-sequential band.

## 2. Why TMLR is the right venue

TMLR's evaluation criterion — "the claims made in the submission are supported by accurate, convincing, and clear evidence; the submission is of interest to the wider community" — fits this paper better than a venue with a stricter novelty / impact bar. Specifically:

1. **The paper's contribution is incremental but clean.** We do not claim a paradigm shift; we identify a *panel-bounded empirical regularity* with a closed-form mechanism and a registered falsifier confirmed. TMLR's claims-and-evidence criterion calibrates correctly against this.
2. **Iterative review benefits the work.** A prior single-shot review (5.9/10 weak reject, archived locally) raised twelve specific priority items. The current V11 manuscript addresses all twelve. TMLR's iterative-review process surfaces and fixes such issues directly, which a one-shot conference reject would not.
3. **J2C track eligibility for NeurIPS 2026.** We intend to apply for J2C certification on acceptance. The paper is structured for J2C compatibility (claim hierarchy box, broader-impact disclosure, reproducibility manifest with SHA-256, multi-seed CIs).

## 3. Reproducibility commitments

- Public anonymous code link is given in the manuscript supplement; all experiments fit on a single RTX 3080 in $\leq 4$ GPU-hours total ($\leq 10$ kgCO₂eq end-to-end estimate).
- A claim-to-artifact manifest (`docs/claim_manifest.json`) maps every numerical claim in the paper to a script, a result JSON, and a SHA-256 of the output. Reviewers can re-run any one-line command and verify the produced JSON byte-equivalent.
- All small-scale results use $\geq 3$ seeds; production-scale runs use 5 probe seeds (ResNet-50 V1+V2 and ViT-L/16; GPT-2-medium reported as single-seed point estimate due to RTX-3080 VRAM ceiling on the FIM-accumulator, with the V6.0b probe-convergence study standing in for variance characterisation).
- All pretrained checkpoints are canonical *published* weights (torchvision / HuggingFace); we do not access raw ImageNet or WebText training data.

## 4. Honest scope and limitations (claim hierarchy, §1.2 of paper)

We separate **proved** (Theorem 1, 1', Proposition 2, Hanin–Nica), **empirically supported within Hanin–Nica scope** ($\sqrt{L}$ scaling on 5 substrate classes), **empirically supported, panel-bounded** (the Definition + Proposition 1 dichotomy), and **conjectural** (the reverse-implication direction, extension beyond the 12-substrate panel). The 12 columns of the §1.2 claim-hierarchy box make this explicit; the title was softened from "Universality Signature" to "Panel-Bounded Empirical Regularity" specifically to match scope.

Architecture-narrowing results (attention with tied/untied embeddings, time-unrolled RNN/LSTM, Mamba SSM with selective gating) are reported honestly — the mechanism's $\sqrt{L}$ scaling does not extend to those substrates. The dichotomy magnitude does, but on a graded basis.

## 5. Suggested action editor expertise

- Information geometry / Fisher information in deep learning (Amari, Karakida, Pennington schools)
- Random-matrix theory of products of random matrices (Hanin / Nica / Yang)
- Tensor networks and the renormalisation-group view of deep nets (Saxe / Vidal / Mehta)
- Empirical-spectral analysis of trained networks (Martin–Mahoney; Papyan)

## 6. Conflicts of interest

None to declare. The author has no employment / consulting / family relationships with any TMLR action editor or reviewer pool member known to the author.

## 7. Funding

Self-funded research at STARGA, Inc. No external grant funding.

---

*Cover letter v1 for TMLR submission, 2026-04-27. STARGA, Inc.*
