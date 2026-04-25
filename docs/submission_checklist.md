# Submission Checklist — NeurIPS 2026

Pre-flight review before submitting to NeurIPS 2026 main track (or workshop fallback / arXiv first-release).

## Paper artifacts

- [x] `docs/fim_tier_hierarchy_neurips2026.md` — final paper source, full empirical + mechanism + production-scale coverage.
- [ ] `docs/fim_tier_hierarchy_neurips2026.pdf` — generate via NeurIPS 2026 LaTeX template (regenerate before submission).
- [x] `docs/v6_0_mechanism_hanin_nica.md` — mechanism-level detailed writeup.
- [x] `docs/cover_letter.md` — submission cover letter (workshops/journal only; not used by NeurIPS main).
- [x] `docs/preregistration_v2.md` — locked predictions for cluster + archival work.
- [x] `docs/h200_cluster_runbook.md` — instructions for §2.A/B of the preregistration.
- [x] `docs/neurips_reproducibility_checklist.md` — filled-out NeurIPS 2026 reproducibility checklist.
- [x] `docs/v1_1_ntk_gap_closure.md` — finite-width NTK derivation referenced in §4.1.
- [x] `docs/references.bib` — BibTeX for all citations.
- [x] `docs/claim_manifest.json` — machine-readable claim → script → JSON map.
- [x] `plots/v2_fig1_dichotomy.png` — 12-substrate box plot with 95 % CIs.
- [x] `plots/v2_fig2_depth_sweep.png` — V6.0 √L fit on untrained MLPs.
- [x] `plots/v2_fig3_substrate_universality.png` — 4-substrate overlay.

## Code artifacts

- [x] `run_all.sh` — full regenerable pipeline, idempotent.
- [x] `experiments/v4_0_uniqueness/learning_baselines.py` — 4 shallow learners.
- [x] `experiments/v4_0_uniqueness/run_trained_vs_untrained.py` — V4.1.
- [x] `experiments/v5_0_lattice_qcd/lattice_u1.py` — U(1) pure-gauge.
- [x] `experiments/v5_0_lattice_qcd/beta_sweep.py` — V7.1 β-dependence.
- [x] `experiments/v6_0_depth_mechanism/depth_sweep.py` — V6.0 MLP depth.
- [x] `experiments/v6_0_depth_mechanism/width_sweep.py` — V6.1 width.
- [x] `experiments/v6_0_depth_mechanism/trained_depth_sweep.py` — V6.2.
- [x] `experiments/v6_0_depth_mechanism/bc_depth_sweep.py` — V6.3 boolean circuits.
- [x] `experiments/v6_0_depth_mechanism/transformer_depth_sweep.py` — V6.4.
- [x] `experiments/v6_0_depth_mechanism/activation_sweep.py` — V6.5 activations.
- [x] `experiments/v6_0_mechanism/probe_convergence.py` — V6.0b probe + dtype stability.
- [x] `experiments/v6_0_mechanism/pooling_error_bound.py` — V6.0c numerical Proposition 2.
- [x] `experiments/v7_0_lattice_su2/lattice_su2.py` — SU(2) non-abelian.
- [x] `experiments/v8_0_tensor_network/binary_tree_tensor_network.py` — V8.0 BTTN.
- [x] `experiments/v5_0_dichotomy_stats/dichotomy_stats.py` — bootstrap + Mann–Whitney U.
- [x] `experiments/v5_0_dichotomy_stats/threshold_sensitivity.py` — V5.1 ROC + LOSO.
- [x] `experiments/v5_0_dichotomy_stats/mw_bootstrap.py` — V5.2 p-value bootstrap.
- [x] `experiments/v9_modern_arch/resnet_gpt2_depth.py` — V9 ResNet + GPT-Tiny tied.
- [x] `experiments/v9_modern_arch/gpt_untied_depth.py` — V9.1 untied falsifier.
- [x] `experiments/v9_modern_arch/cifar_resnet18_fim.py` — V9.2 CIFAR-10 ResNet-18.
- [x] `experiments/v9_modern_arch/cifar100_resnet18_fim.py` — V9.2b CIFAR-100 ResNet-18.
- [x] `experiments/v9_modern_arch/rnn_depth.py` — V9.3 RNN/LSTM.
- [x] `experiments/v9_modern_arch/mamba_depth.py` — V9.4 Mamba SSM (pre-registered).
- [x] `experiments/v9_modern_arch/imagenet_resnet50_fim.py` — V9.5 ImageNet ResNet-50.
- [x] `experiments/v9_modern_arch/gpt2_medium_fim.py` — V9.6 GPT-2-medium.

## Quality gates

- [x] **Tests pass.** `pytest tests/` → all green.
- [x] **Git clean.** `git status` → clean.
- [x] **Numerical consistency.** All paper claims cross-checked against JSON outputs.
- [x] **Terminology unified.** "Deep layered sequential composition" throughout.
- [x] **Cross-references valid.** Every `[foo](bar)` link resolves to an existing file.
- [x] **No "TODO" / "in progress" markers** in public docs.
- [x] **Author attribution.** STARGA Inc. only — no other authors, no AI mentions.
- [ ] **Paper PDF renders** without LaTeX errors using NeurIPS 2026 template.
- [ ] **Anonymisation pass** — strip STARGA branding from author block per NeurIPS double-blind rules (use the `anonymous` template option).

## Known out-of-scope items

- Production-scale finetune-from-scratch ImageNet ResNet-50 90-epoch run (we use the canonical pretrained weights; sufficient for FIM-diagonal claim).
- Full real-data α-drift analysis (UVES + HIRES) — out of scope for this paper.
- 4D spacetime emergence with Lorentz signature — out of scope, noted as open.
- Formal extension of Hanin–Nica from FC networks to attention-based architectures — paper flags this as theoretical open problem.

## Submission targets (in priority order)

1. **NeurIPS 2026 main track** — primary target. Empirical universality + closed-form mechanism + 12-class substrate panel + production-scale verification.
2. **NeurIPS 2026 Workshop on Foundations of Deep Learning** — fallback if main rejects.
3. **ICLR 2027** — if NeurIPS 2026 cycle slips.
4. **arXiv (cs.LG + math.PR)** — first public release ahead of NeurIPS reviews.

## Pre-submission sanity questions

1. Is the strongest empirical claim in the abstract? Yes — complete rank separation $p = 1.7 \times 10^{-17}$, $r = 1.000$.
2. Is the mechanism tied to a published theorem? Yes — Hanin–Nica 2020 (Comm. Math. Phys. 376, 287–322, arXiv:1812.05994), plus our Theorem 1' exact form and Proposition 2 pooling-error bound.
3. Are the falsification criteria pre-registered? Yes — `docs/preregistration_v2.md` and §3.5 Falsifiability Ladder.
4. Are real-data benchmarks present? Yes — V9.2 CIFAR-10, V9.2b CIFAR-100, V9.5 ImageNet ResNet-50 (V1 and V2 weights), V9.6 GPT-2-medium.
5. Is reproducibility complete? Yes — `docs/neurips_reproducibility_checklist.md`, claim manifest with SHA-256, one-line per-figure repro recipe.

*Checklist v2, NeurIPS 2026 main-track edition. STARGA Commercial License.*
