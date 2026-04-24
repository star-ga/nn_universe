# Submission Checklist — V2 Paper

Pre-flight review before submitting to a NeurIPS / ICML workshop, ICLR TinyPapers,
or an arXiv first-release.

## Paper artifacts

- [x] `docs/paper_draft.md` — V2 source, 351 lines, full empirical + mechanism coverage.
- [x] `docs/nn_universe_paper_V2.pdf` — generated via `pandoc … --pdf-engine=lualatex`.
- [x] `docs/v6_summary.md` — 3-page workshop handout.
- [x] `docs/v6_0_mechanism_hanin_nica.md` — mechanism-level detailed writeup.
- [x] `docs/findings.md` §0 headline — cross-checked.
- [x] `docs/cover_letter.md` — this submission's cover letter.
- [x] `docs/preregistration_v2.md` — locked predictions for cluster + archival work.
- [x] `docs/h200_cluster_runbook.md` — instructions for §2.A/B of the preregistration.
- [x] `plots/v2_fig1_dichotomy.png` — 12-substrate box plot with 95% CIs.
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
- [x] `experiments/v7_0_lattice_su2/lattice_su2.py` — SU(2) non-abelian.
- [x] `experiments/v8_0_tensor_network/binary_tree_tensor_network.py` — V8.0 BTTN.
- [x] `experiments/v5_0_dichotomy_stats/dichotomy_stats.py` — bootstrap + MW U.
- [x] `experiments/v1_2_scaling/sigma_min_validation.py` — σ_min validator.
- [x] `experiments/v3_1_alpha/real_data_pipeline.py` — α-drift stub with null test.

## Quality gates

- [x] **Tests pass.** `pytest tests/` → 104+ tests, 0 failures.
- [x] **Git clean.** `git status` → clean.
- [x] **Git pushed.** `git log origin/main..HEAD` → empty.
- [x] **Numerical consistency.** All docs cross-checked against JSONs
      (see audit commit `8b9e3f0`).
- [x] **Terminology unified.** "Deep layered sequential composition"
      throughout; pre-V4.1 "FIM eigenvalue" renamed to "FIM diagonal".
- [x] **Cross-references valid.** Every `[foo](bar)` link resolves to an
      existing file.
- [x] **No "TODO"/"in progress"/"in preparation" markers** in public docs.
- [x] **Audit citations stripped** — internal multi-LLM audits are not
      cited in public-facing materials.
- [x] **Paper PDF renders** without LaTeX errors.

## Known out-of-scope items

- [ ] Full W=45000 σ_min measurement (requires H200; see runbook).
- [ ] 20-seed W ≥ 14000 σ_min robustness (same).
- [ ] Real-data α-drift analysis (requires UVES+HIRES archival data access).
- [ ] 4D spacetime emergence with Lorentz signature — out of scope, noted as open.
- [ ] Formal extension of Hanin–Nica from FC networks to attention-based
      architectures — paper flags this as theoretical open problem.

## Submission targets (in priority order)

1. **NeurIPS 2026 Foundations of Deep Learning workshop** — best fit (empirical
   + theorem-adjacent + cross-substrate universality).
2. **ICLR 2026 TinyPapers** — good fit for the 3-page workshop handout.
3. **arXiv (cs.LG + physics.comp-ph)** — first public release before workshop
   reviews come back.
4. **ML Safety Workshop / Neural Theory Workshop** (various) — if workshop
   submissions are capped at one.

## Pre-submission sanity questions

1. Is the strongest empirical claim in the abstract? Yes — complete rank
   separation with $p = 1.7 \times 10^{-17}$ is the headline.
2. Is the mechanism clearly tied to a published theorem? Yes — Hanin–Nica 2020
   (Comm. Math. Phys. 376, 287–322, arXiv:1812.05994).
3. Is the cosmological framing honestly scoped? Yes — refined to a necessary-
   condition claim (§5.3), explicit open-question framing on whether the
   universe's substrate falls inside the deep-layered-sequential class.
4. Are the falsification criteria pre-registered? Yes — `docs/preregistration_v2.md`.
5. Are all headline numbers cross-checked against JSON? Yes — audit `8b9e3f0`.

*Checklist v1, 2026-04-24. STARGA Commercial License.*
