# NeurIPS Reproducibility Checklist

Per the NeurIPS 2026 reproducibility checklist requirements (`https://nips.cc/public/guides/CodeSubmissionPolicy`).

## 1. Claims

- [x] **The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.** Yes; the empirical dichotomy and Hanin-Nica mechanism claims are scoped exactly as the experiments support, with explicit cosmological-substrate framing as a *necessary condition*, not a proof.
- [x] **Limitations of the work are discussed.** Yes; §5.4 lists four explicit limitations (parameter-count extrapolation, 4D emergence open, SV exponent noisy, T3 partial-fit).

## 2. Theory

- [x] **For each theoretical result, the full set of assumptions is provided.** Yes; Hanin-Nica 2020 assumptions (i.i.d. Gaussian weights, ReLU or sub-Gaussian nonlinearity, large $L$ and $n$) are stated explicitly in §4.6 and Appendix B.
- [x] **For each theoretical result, the complete proof is provided.** The mechanism derivation (log-normal $F_{ii}$ → $\log(T_1/T_3) \propto \sqrt{L}$) is in Appendix B, ten lines, full algebra. The deeper Hanin-Nica theorem itself is cited (Comm. Math. Phys. 376, 287–322, 2020) rather than re-derived.

## 3. Experiments

- [x] **Information sufficient to reproduce all experimental results.** Yes — every script in `experiments/`, with command-line args defaulting to the paper-reported configurations.
- [x] **All datasets used.** Synthetic only — Gaussian self-prediction, toric-code syndrome decoding, symbolic regression, supervised vision classification on Gaussian-labelled $32{\times}32{\times}3$ data; all generators in-script with fixed seeds.
- [x] **All models trained.** 5-layer 256-neuron ReLU MLP (V1.0/V1.2/V3.0 ladder), `_CheckpointedFC` for V3.0 cluster widths, `BinaryTreeTensorNetwork`, `LayeredBooleanCircuit`, custom `Transformer` (V6.4), `_lattice_u1` and `_lattice_su2` (Wilson action MC).
- [x] **All training details (hyperparameters, optimiser, batch size, etc.) are specified.** §3.4 + every script's CLI defaults; canonical: SGD momentum=0.9, lr=1e-3, batch=128, 2000 steps for V1.0 / V3.0 unless noted.
- [x] **Number of seeds, error-bar definition.** 3-20 seeds depending on width; 95 % bootstrap CIs (log-transformed) computed in `experiments/v5_0_dichotomy_stats/dichotomy_stats.py`.
- [x] **Compute resources (CPU/GPU, memory, time) sufficient to reproduce.** Single RTX 3080 10 GB + 64 GB RAM CPU node; total `run_all.sh` budget ~3 hours. Cluster-scale items (W ≥ 14000 NN training) require H200 (see `docs/h200_cluster_runbook.md`).
- [x] **Hyperparameter ranges searched.** No search; experiments are at fixed canonical hyperparameters across the ladder. Hyperparameter robustness via depth/width/seed sweeps (V6.0–V6.5).
- [x] **Multiple-comparison correction.** §4.5 reports a single one-sided Mann–Whitney U test on a single decision (deep-sequential vs rest); no Bonferroni needed for the headline test. Per-system bootstrap CIs at α = 0.05; Bonferroni-corrected α = 0.05/12 = 0.00417 over the 12 substrate classes still gives all CIs entirely on the correct side of the threshold (computation in `experiments/v5_0_dichotomy_stats/dichotomy_stats.py`).

## 4. Code

- [x] **Code is available.** Yes — public repo at `https://github.com/star-ga/nn_universe`.
- [x] **Code is documented.** README + per-section docstrings + 9 standalone docs (paper_draft, v6_summary, v6_0_mechanism_hanin_nica, findings, v1_1_ntk_continuum_limit, v2_0_lattice_embedded, v3_*, v4_*, v5_*).
- [x] **Code includes tests.** `tests/` with 104 pytest tests passing; runtime ~60 s.
- [x] **Code is licensed.** STARGA Commercial License (permissive academic reuse with citation; see repo `LICENSE`).

## 5. Statistical methodology

- [x] **Bootstrap protocol specified.** §4.5 + Appendix A: 2 000 log-bootstrap resamples, percentile method, seed 42.
- [x] **Hypothesis test specified.** §4.5 + Appendix A: one-sided Mann–Whitney U; rank-biserial r as effect size.
- [x] **Sample sizes reported per system.** §4.5 dichotomy table column "n seeds" + Appendix A breakdown.
- [x] **Confidence levels reported.** All CIs at 95 % unless noted otherwise.
- [x] **Multiple comparisons handled.** See §3 above.

## 6. Reproducibility

- [x] **Single command reproduces all in-scope experiments.** `bash run_all.sh` from the repo root.
- [x] **Cluster-scale experiments documented.** `docs/h200_cluster_runbook.md` gives exact commands for the H200-only items (W=45000 σ_min + 20-seed robustness).
- [x] **Pre-registration of cluster predictions.** `docs/preregistration_v2.md` locks decision rules for §2.A/B (cluster) and §2.C (archival α-drift) before execution.

## 7. Broader impact

- [x] **Broader-impact statement included.** §5.3 + §6: the cosmological-substrate refinement is an *empirically necessary-condition* claim, not a metaphysical one; the paper explicitly discusses what it does and does not show. No human-subject data, no privacy implications.

## 8. Ethics

- [x] **No human subjects.**
- [x] **No personally identifiable data.**
- [x] **No content-moderation considerations.**
- [x] **No dual-use / misuse risks identified.**

---

*Checklist v1, 2026-04-24, prepared in advance of NeurIPS 2026 main-track submission.*
