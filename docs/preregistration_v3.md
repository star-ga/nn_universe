# Preregistration V3 — V12 cluster follow-up

**Status:** locked at commit time. Decision rules below must not be edited after V12 runs start. Falsifiers and predictions are fixed *a priori* per item.

This document extends `docs/preregistration_v2.md` (which covers V3.0 cosmology-side σ_min and α-drift work) with the 5 V12 cluster follow-up items planned to strengthen the empirical evidence base for `paper_main_v11_8`.

## Why these items, and why locked now

A pre-submission rigor review of paper_main_v11_8 identified 5 concrete reviewer-style gaps that require new compute (not text edits). Items 6 (camera-ready text move) and 7 (formal extension theorem) need no new compute. Items 1–5 below all need new runs; locking the predictions + falsifiers BEFORE running protects against post-hoc analysis-choice optimisation (HARKing).

The full plan with hardware budgets is in `docs/cluster_roadmap_v12.md`. Decision logic that consumes the run output is in `experiments/v12_partition_invariant/decision_rules.py`.

## Item 1 — Partition-invariant table across the full 13-substrate panel

**Audit gap.** §4.5 partition-invariant verification (Gini / effective rank / top-1 % mass) is currently only computed for untrained MLP depth sweeps + uniform baseline. The dichotomy *prediction* is partition-free, but the panel-wide *evidence* relies on the 1 % / 49 % / 50 % T_1/T_3 partition for 11 of 13 substrates.

**Predicted outcome.**
- Every deep-sequential substrate has (Gini > 0.7, effective rank / n < 0.05, top-1 % mass > 0.4).
- Every non-deep substrate has (Gini < 0.5, effective rank / n > 0.3, top-1 % mass < 0.1).
- The three partition-free statistics agree on the deep-vs-rest classification for ≥ 80 % of the 13 substrates.

**Falsifier.** 2 of 3 statistics disagree on the deep-vs-rest direction for any substrate; or < 80 % of substrates show 3-way agreement.

**Script.** `experiments/v12_partition_invariant/run.py` — saves raw FIM-diagonal arrays per (substrate, seed) and computes the three statistics.

## Item 2 — FIM-diagonal under real LM cross-entropy loss

**Audit gap.** All V11.x LM measurements (Pythia 1.4B / 2.8B / 6.9B, OLMoE-1B-7B-0924, Mamba-790M-HF) use Gaussian-probe self-prediction, not next-token cross-entropy on real text. GPT-5.5 critique: "LM evaluations appear not to use language-modeling loss".

**Predicted outcome.** Every LM stays in the deep-sequential band (T_1/T_3 > 100) under real-text Pile probes. Magnitude likely 0.3–1.0 dex *lower* than Gaussian-probe (real text has stronger structure than random token IDs).

**Falsifier.** Any LM drops below T_1/T_3 = 100 under Pile-validation probes.

**Script.** `experiments/v12_lm_loss_fim/run.py` — uses Pile validation (HF `monology/pile-uncopyrighted`) for probes; falls back to local cache or Gaussian-text *with* an explicit `probe_distribution=gaussian_text_fallback` JSON flag.

## Item 3 — Multi-seed at production scale

**Audit gap.** V11.8 has 5-seed coverage for Pythia-6.9B + OLMoE-1B-7B only. ResNet-50, ViT-L/16, GPT-2-large, Mamba-790M-HF are reported single-seed.

**Predicted outcome.** Every production model's 5-seed (probe-randomisation) bootstrap 95 % CI lies entirely above 100 (for deep-sequential models) or entirely below 100 (for non-deep controls). CV ≤ 5 % at billion-param scale, consistent with the seed-CV-shrinks-with-N trend documented in §5.1.

**Falsifier.** Any production model's 5-seed CI crosses T_1/T_3 = 100.

**Note.** This is *probe-seed* multi-seed (no retraining), not training-seed multi-seed. Training-seed variance at ImageNet-1K from-scratch is a separate >$1000 cluster ask outside V12 scope.

**Script.** `experiments/v12_production_multiseed/run.py`.

## Item 4 — Probe-convergence sweep at billion-param scale

**Audit gap.** V6.0b probe-convergence sweep was on ≤ 300k-param MLPs. Reviewers ask whether the 200-probe convention holds at billion-param scale.

**Predicted outcome.** T_1/T_3 stabilises by n = 200 within ±5 % of T_1/T_3 at n = 1600 on Pythia-2.8B. (Matching the small-scale V6.0b finding.)

**Falsifier.** |T_1/T_3(200) − T_1/T_3(1600)| / |T_1/T_3(1600)| > 5 %.

**Script.** `experiments/v12_probe_convergence_large.py`.

## Item 5 — Parameter-matched non-deep production-scale control

**Audit gap.** No production-scale *non-deep* comparator exists. Reviewers can argue the dichotomy is just an effect of model size.

**Predicted outcome.** 300M-param random-feature ridge regression (RFF kernel ridge) has T_1/T_3 < 6 (matching the four small-scale shallow learners in §4.5).

**Falsifier.** RFF-300M gives T_1/T_3 > 100. This would falsify the depth + sequential composition primitive hypothesis — the dichotomy would instead be a model-size effect, and the entire mechanism story narrows substantially.

**Script.** `experiments/v12_nondeep_control/run.py`.

## Decision logic

`experiments/v12_partition_invariant/decision_rules.py` consumes `v12_aggregate.json` produced by `experiments/v12_partition_invariant/aggregate.py` and emits `v12_decision_verdict.json` with PASS/FAIL per item and an overall verdict:

- `PASS_ALL` ⇔ all 5 items PASS
- `PARTIAL` ⇔ 3+ pass
- `FAIL` ⇔ < 3 pass

## Locked at commit time

- Commit hash: (filled at commit time)
- SHA-256 of this file: (filled at commit time)
- Scripts SHA-256s:
  - `experiments/v12_partition_invariant/run.py`
  - `experiments/v12_lm_loss_fim/run.py`
  - `experiments/v12_production_multiseed/run.py`
  - `experiments/v12_nondeep_control/run.py`
  - `experiments/v12_probe_convergence_large.py`
  - `experiments/v12_partition_invariant/aggregate.py`
  - `experiments/v12_partition_invariant/decision_rules.py`

Any change to the predictions or falsifiers above after the V12 runs start invalidates the preregistration. Such a change requires a V12.1 preregistration committed before the affected re-run.

## Out of scope for V12

- Item 6 (camera-ready text move of Theorem 1' into main): no compute, no preregistration needed; lands at camera-ready (12-page allowance).
- Item 7 (formal Hanin–Nica extension theorem for attention / SSM / residual): theoretical work, separate paper.
- Real α-drift on UVES + HIRES + SDSS DR18: covered in V2 preregistration.
- 4D + Lorentz-signature spacetime emergence: covered in V2 preregistration as out-of-scope-of-this-paper.
