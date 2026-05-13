# Cluster Follow-up Roadmap — V12

**Purpose.** Close the remaining pre-submission rigor review gaps that block 10/10 in the multi-reviewer audit panel. Each item below maps to a specific reviewer-style critique surfaced by GPT-5.5, model-B, model-F-large, model-E-v4-pro, and vendor-I model-I against `paper_main_v11_8`. None of these items invalidate the current submission — §5.3 Limitations + the pre-registration disclosure cover them as honest scope statements — but they would move audit mean from ~6/10 (submission-time) to 8.5–9.5/10 for the camera-ready / journal-extension version.

## Status (2026-05-13)

- Main submission PDF: 9 pages, 3 audit passes, 5 latest-model audit responses (model-D.3 8, self-Opus 8, model-F 7, GPT-5.5 5, model-E-v4-pro 5, vendor-I 5). Currently honest-weak-accept territory.
- All in-paper wordsmithing gaps closed (panel-bounded language, dual statistical tests, pre-registered threshold disclosure, boolean-circuit sensitivity-analog caveat, NeurIPS-2026 header, contiguous bibliography 1–37).
- Remaining gaps require *new compute*, not text edits. That work is what this doc plans.

## Items + concrete budgets

| # | Audit gap | Experiment | Where it lives | Compute |
|---|---|---|---|---|
| 1 | Partition-invariant Gini / effective rank / top-1 % mass only computed for untrained MLP depth sweeps + baselines, not the full 13-substrate panel | Re-run §4.5 panel, *retain raw FIM diagonal arrays* (not just summary stats), compute the 3 partition-free statistics on each, ship 13×3 table | `experiments/v12_partition_invariant/` (new) | ~24 GPU-hours on 4070 / ~2 H100-hours |
| 2 | LM family uses Gaussian-probe self-prediction, not real LM cross-entropy loss | FIM-diagonal of Pythia-1.4B/2.8B/6.9B + OLMoE-1B-7B + Mamba-790M-HF under next-token cross-entropy on 200 samples drawn from The Pile (validation split). Compare to Gaussian-probe FIM on same checkpoints. | `experiments/v12_lm_loss_fim/` (new) | ~48 GPU-hours on 4070 (with INT4 for 6.9B + OLMoE) / ~6 H100-hours full FP16 |
| 3 | Single-seed at production scale (ResNet-50, ViT-L/16, GPT-2-large, Mamba-790M) | 5-seed re-evaluation (probe-seed re-randomisation, not retraining). Report mean ± std + 95 % CI per checkpoint. | `experiments/v12_production_multiseed/` (new) | ~12 GPU-hours on 4070 |
| 4 | Probe convergence at billion-param scale unshown | Probe-count sweep n ∈ {50, 100, 200, 400, 800, 1600} on Pythia-2.8B (and 6.9B if feasible), plot T_1/T_3 stability vs n | `experiments/v6_0_mechanism/probe_convergence.py` (extend) | ~6 GPU-hours |
| 5 | No production-scale non-deep comparator | **Parameter-matched control**: 300M-param random-feature ridge regression (RFF kernel ridge), measured under the same protocol Π, vs ViT-L/16 (304M params). NOT a billion-param GP — that's memory-prohibitive on any consumer GPU. | `experiments/v12_nondeep_control/` (new) | ~6 GPU-hours |
| 6 | Theorem 1' (closed-form quantile→tier ratio) only in supplementary | Move full derivation into camera-ready main text (12-page allowance) | n/a | 0 |
| 7 | Formal Hanin–Nica extension to attention / SSM / residual | Theoretical work. Open problem. Separate paper. | n/a | Theorist-month, not compute |

### Runpod options surveyed (May 2026 pricing)

| Class | VRAM | Community $/hr | Notes |
|---|---|---|---|
| B200 | 180 GB | $4.99 | Newest Blackwell |
| **H200 SXM** | **141 GB** | **$3.59** | Sweet spot for V12 |
| H100 SXM | 80 GB | $2.99 | Cheapest viable path |
| H100 PCIe | 80 GB | $2.79 | Slower interconnect |
| MI300X | 192 GB | varies | AMD, fewer ML tools |

Multi-GPU: Instant Clusters scale to **64 GPUs**; single-pod max 8× SXM with NVLink. V12 is *embarrassingly parallel* across substrates/seeds, so multi-node tensor parallelism gives zero speedup — VRAM-per-GPU is the binding constraint.

### Total compute budget — four scenarios

| Scenario | Hardware | Wall-clock | Cloud spend | Methodology risk |
|---|---|---|---|---|
| **A — Cheapest viable** | 4070×2 (free) + 1×H100 80 GB for Pythia-6.9B + OLMoE FP16 (3 h) | 7 days | **~$15** | INT4 quant noise on some items, mixed-GPU orchestration |
| **B — Single H200 (recommended)** | 1× H200 141 GB | ~8 hours | **~$32** | None — full FP16, single pod, clean |
| **C — 8× H200 speedrun** | 8× H200 SXM Instant Cluster | ~3 hours | **~$96** | None, but 3× cost for negligible methodological gain |
| **D — Biggest reasonable** | 64× B200 Instant Cluster | ~30 min | **~$160** | None, but ~5× cost vs B for zero scientific benefit (embarrassingly parallel) |

**Recommendation: Scenario B (1× H200, 8 hours, ~$32 total).** Clean methodology, full FP16, defensible against camera-ready audit reviewers. Scenario A is the zero-spend fallback with documented quant-noise caveats.

## Embarrassingly-parallel orchestration

The 2 RTX 4070s sit on separate Ethernet boxes — no NCCL-over-TCP. Use **substrate-level work-stealing**: a manifest of (substrate, seed) tuples, each box pulls work atomically via SSH-locked queue file, writes result JSON to NAS.

See `scripts/run_v12_cluster.sh` for the orchestration driver.

## Decision rules per item

- **Item 1 (partition-invariant table).** Falsifier: any substrate fails to monotonically order with depth under Gini AND $r_{\text{eff}}/n$ AND $m_1$. If 1 of 3 statistics disagrees: weakens but does not invalidate the dichotomy. If 2 or 3 disagree: dichotomy is partition-sensitive, paper claim narrows.
- **Item 2 (real-data LM-loss).** Falsifier: any LM moves out of the deep-sequential band (T_1/T_3 < 100) under LM cross-entropy probes. Predicted: stays in band, magnitude probably 0.3–1 dex lower than Gaussian-probe.
- **Item 3 (5-seed at scale).** Falsifier: any production model's 5-seed CI crosses T_1/T_3 = 100. Predicted: CIs are tight, none crosses.
- **Item 4 (probe convergence).** Falsifier: T_1/T_3 doesn't stabilise by n=400 within 5 % of n=1600 value. Predicted: stabilises by n=200 (matching small-scale V6.0b finding).
- **Item 5 (parameter-matched non-deep control).** Falsifier: 300M-RFF kernel ridge gives T_1/T_3 > 100. Predicted: stays below 6 (matching the small-scale shallow learners in §4.5).

## Pre-registration

Append these 5 items to `docs/preregistration_v2.md` §6 (new) with locked decision rules above. Commit before any V12 run starts. SHA-256 hash of this doc + the experiments/ scripts goes in commit message.

## Timeline if green-lit

- T+0: pre-register + commit V12 scripts (1 day)
- T+1 to T+5: run V12 on 4070×2 (5 days)
- T+6: rent 3 hours of H100 for Pythia-6.9B-FP16 path (1 day)
- T+7 to T+10: write V12 supplementary, regenerate Camera-ready PDF (4 days)

Total: 10 working days from green-light to camera-ready with all V12 items integrated.

## What this does NOT close

- Item 7 (formal Hanin–Nica extension theorem) is a separate theoretical paper. Best path: invite a Hanin or Pennington collaborator post-acceptance.
- Cosmological framings (Vanchurin, etc.) remain out of scope — explicitly. They live in the parent framework, not this paper.
- The 4D + Lorentz-signature spacetime emergence question is unsolved and stays an open problem.

## References to existing infrastructure

- `docs/preregistration_v2.md` — V3.0 cosmology preregistration (separate from V12 FIM work)
- `docs/h200_cluster_runbook.md` — σ_min validation runbook (separate from V12)
- `scripts/run_v12_cluster.sh` — V12 orchestration (this roadmap)
- `experiments/v6_0_mechanism/probe_convergence.py` — extended for item 4
- `experiments/v5_0_dichotomy_stats/` — baseline for item 1 partition-invariant extension
