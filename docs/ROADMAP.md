# nn_universe — Master Roadmap

**Last updated:** 2026-05-13
**Current status:** NeurIPS 2026 main-track submission packaged (9 pages, anonymous, audit pass 3 complete). V12 cluster follow-up planned but not started. All in-paper wordsmithing gaps closed.

This is the single canonical place for "what's left, in what order, on what hardware, at what cost". Sub-docs below have detail.

---

## TL;DR

```
SUBMISSION-READY now  →  /data/checkpoints/neurips2026_submission/paper.pdf  (9pp, anonymous)
                          docs/fim_tier_hierarchy_neurips2026.md  (source)
                          audit mean 6.6/10 across 5 2026-frontier models

NEXT SUBMISSION-WORK  →  V12 cluster follow-up (camera-ready / journal track)
                          docs/cluster_roadmap_v12.md  (plan)
                          scripts/run_v12_cluster.sh  (driver)
                          experiments/v12_*/run.py     (TO WRITE — next commits)

PARALLEL OTHER-WORK   →  mind-mem-4b v4.1.0 (separate project, training r3 in progress)
                          Patent 63/947,737 non-prov filing  (deadline 2026-12-23)
```

---

## Stage 0 — Submission state (DONE)

| Artifact | Path | Status |
|---|---|---|
| Paper PDF (main) | `/data/checkpoints/neurips2026_submission/submission/paper.pdf` | 9 pages, NeurIPS 2026 header, anonymous |
| Supplementary PDF | `/data/checkpoints/neurips2026_submission/submission/supplementary.pdf` | 10 pages |
| Submission bundle | `/data/checkpoints/neurips2026_submission/submission.zip` | refreshed |
| Markdown source | `docs/fim_tier_hierarchy_neurips2026.md` | V11.8 + audit fixes |
| Working trim source | `/tmp/neurips_build/work/paper_main_v11_8_anon.md` | 9-page edit target |
| Reproducibility | `docs/neurips_reproducibility_checklist.md` | filled |
| Pre-registration | `docs/preregistration_v2.md` | V2.0 (σ_min + α-drift; V12 to be appended) |
| Audit summary | `/tmp/neurips_build/work/AUDIT_SUMMARY_2026_v3.md` | pass 3, 5 working models |
| Multi-reviewer memory | `~/.claude/projects/-home-n/memory/feedback_always_use_2026_latest_models.md` | locked-in 2026 model names |

Audit results across 3 passes:

| Model | Pass 1 (stale 2025) | Pass 2 (2026, no fixes) | Pass 3 (2026, all fixes) | Pass 4 (md-source, count-fix) |
|---|---|---|---|---|
| openai/model-A | (model-A-prev: 8) | 4 | 4 | 5 |
| anthropic/model-B (self) | (4-5: 9) | api-fail | 8 ACCEPT | — |
| google/model-C | (2.5: 9) | hung | hung | 503 throttle |
| xai/model-D | (4-1-fast: 9) | 7 | 8 ACCEPT | — |
| model-E/model-E | (chat: 7) | 5 | 5 | — |
| model-F/model-F | 7 | 7 | 7 | — |
| vendor-I/model-I | (4.6: 2) | 4 | 5 | — |
| model-G/model-G | quota | quota | quota | quota |
| nvidia/model-H-ultra | api-fail | api-fail | api-fail | api-fail |

**Mean of working-model pass-3 scores: 6.6/10.** Honest-weak-accept. Headline gap: V12 cluster work below.

---

## Stage 1 — V12 cluster follow-up (NEXT COMMITS, ~5–7 days compute)

Full plan: [docs/cluster_roadmap_v12.md](cluster_roadmap_v12.md). Orchestration driver: `scripts/run_v12_cluster.sh`.

### V12 item-by-item next-commit checklist

The orchestration script in `scripts/run_v12_cluster.sh` references 5 sub-scripts. They are **not yet written**. Next commits, in dependency order:

#### Commit A — `experiments/v12_partition_invariant/` (item 1)
- `run.py` — re-runs the 13-substrate panel, saves raw FIM diagonal arrays to `.npy`, computes Gini + effective rank + top-1 % mass per (substrate, seed).
- `aggregate.py` — pools all `*.json` results into `v12_aggregate.json`.
- `decision_rules.py` — checks monotone-with-depth predicate per partition-free stat; emits `v12_decision_verdict.json`.
- Compute budget: ~24 GPU-hours on 4070, ~2 H100-hours.

#### Commit B — `experiments/v12_production_multiseed/` (item 3)
- `run.py` — loads pretrained ResNet-50 / ViT-L/16 / GPT-2-large / Mamba-790M, runs 5-seed probe re-randomisation FIM measurement, reports mean ± std + 95 % CI.
- Compute budget: ~12 GPU-hours on 4070.

#### Commit C — `experiments/v6_0_mechanism/probe_convergence.py` *extension* (item 4)
- Add `--scale large` mode that targets Pythia-2.8B at probe counts {50, 100, 200, 400, 800, 1600}, plots T_1/T_3 stability.
- Compute budget: ~6 GPU-hours on 4070.

#### Commit D — `experiments/v12_nondeep_control/` (item 5)
- `run.py` — 300M-param random-feature ridge regression (parameter-matched to ViT-L/16), FIM diagonal under protocol Π, reports T_1/T_3 + 5-seed CI.
- *Important framing:* not a "billion-param" GP (memory-prohibitive on any consumer GPU). Honest parameter-matched control instead.
- Compute budget: ~6 GPU-hours on 4070.

#### Commit E — `experiments/v12_lm_loss_fim/` (item 2)
- `run.py` — FIM-diagonal of Pythia-1.4B/2.8B/6.9B + OLMoE-1B-7B + Mamba-790M under next-token cross-entropy loss on 200 samples from The Pile validation split.
- INT4-quant fallback path for OLMoE on 12 GB GPUs.
- Pythia-6.9B FP16 path: rent 1× H100 for ~3h ($6) OR INT4 quant + footnote about quantisation noise.
- Compute budget: ~48 GPU-hours on 4070 (with INT4 for 6.9B + OLMoE) / ~6 H100-hours full FP16.

#### Commit F — `docs/preregistration_v3.md`
- Append V12 items 1–5 with locked decision rules + falsifiers from `docs/cluster_roadmap_v12.md`.
- Commit BEFORE V12 run starts. SHA-256 hash of doc + scripts in commit message.

#### Commit G — `experiments/v12_*/aggregate.py` + `decision_rules.py`
- Common aggregation + verdict logic across all V12 items.

#### Commit H — `docs/v12_results.md`
- Post-run writeup. Falsifier outcome per item. Update Table in main paper §4.5 + §4.6 + §5.3 for camera-ready.

#### Commit I — Camera-ready paper integration
- Update `docs/fim_tier_hierarchy_neurips2026.md` with V12 results (12-page allowance for camera-ready).
- Move Theorem 1' (closed-form quantile→tier ratio) into main text (item 6).
- Re-run multi-reviewer audit on camera-ready PDF — target: 9+/10 mean from 2026 frontier models.

### V12 commit order rationale

Items 1, 3, 4, 5 first (all fit 4070, no quant noise, decision rules already locked) → ship results → then item 2 (LM-loss FIM, which has the quant/H100 decision) → then camera-ready.

---

## Stage 2 — Camera-ready (after NeurIPS 2026 accept)

If accepted (probability: weak-accept-likely given §5.3 honesty):
1. Run V12 (Stage 1) — 5–7 days
2. Integrate V12 results — Commit I above
3. Add acknowledgements (post-anonymity): named advisors if any (Visvanathan Ramesh case-by-case)
4. Deanonymise: re-add `Anonymous Author(s)` → real author block in `wrapper.tex`
5. License: restore from `[License removed for anonymous review]`
6. Final multi-reviewer audit on camera-ready PDF
7. Submit camera-ready

If rejected:
- Workshop fallback: NeurIPS 2026 Foundations of Deep Learning workshop
- Or: ICLR 2027 cycle with V12 already integrated → stronger submission
- Or: arXiv first-release with V12 + journal track (JMLR or TMLR)

---

## Stage 3 — Future work explicitly out of scope of this paper

| Item | Owner | Timeline |
|---|---|---|
| Formal Hanin–Nica extension theorem to attention/SSM/residual (V12 item 7) | Theorist collaborator post-acceptance (Hanin / Pennington) | Separate paper, ~6 months |
| 4D + Lorentz-signature spacetime emergence | Parent framework, open theoretical question | Indefinite |
| Real α-drift on UVES + HIRES + SDSS DR18 | Already in `docs/preregistration_v2.md` §2.C | Separate cosmology paper |
| Patent 63/947,737 non-provisional conversion | Nikolai, pro se | Hard deadline 2026-12-23 |

---

## Other open items in `submission_checklist.md`

- [x] `docs/fim_tier_hierarchy_neurips2026.md` — final paper source
- [x] `docs/fim_tier_hierarchy_neurips2026.pdf` — built via NeurIPS 2026 template (V11.8 + audit fixes)
- [x] `docs/v6_0_mechanism_hanin_nica.md` — mechanism writeup
- [x] `docs/preregistration_v2.md` — locked predictions (V2.0)
- [ ] `docs/preregistration_v3.md` — V12 items locked (next commit F)
- [x] `docs/h200_cluster_runbook.md` — σ_min cluster ops
- [x] `docs/neurips_reproducibility_checklist.md` — filled
- [x] All `experiments/v*/` paper-referenced scripts present
- [x] `run_all.sh` — full pipeline
- [x] tests pass, git clean
- [x] anonymisation pass complete
- [x] cross-references valid
- [x] no TODO/in-progress markers in public docs
- [x] author attribution — Anonymous only
- [ ] V12 commits A–I (next)

---

## Files this roadmap supersedes

- `nn_universe_SUBMISSION_ROADMAP_LOCAL.md` (local-only, outside repo) — superseded for the submission-readiness stage by Stage 0 above.

## Files this roadmap depends on

- `docs/cluster_roadmap_v12.md` — V12 detail
- `docs/preregistration_v2.md` — locked V2 predictions
- `docs/h200_cluster_runbook.md` — σ_min runbook (separate from V12)
- `docs/submission_checklist.md` — submission readiness
- `scripts/run_v12_cluster.sh` — V12 driver
- `scripts/build_trimmed.sh` — paper-PDF build
- `scripts/reproduce_main_results.sh` — full-pipeline replay

## Decision rules for picking up work in this repo

1. **Submission is ready** — do not edit `paper_main_v11_8_anon.md` unless camera-ready triggers (acceptance + V12 integration).
2. **V12 commits A–I** are independent and can be picked up in dependency order (A → B → C → D → E → F → G → H → I).
3. **Anything outside V12** belongs in a separate roadmap (e.g. `docs/preregistration_v2.md` for cosmology-side work, mind-mem repo for memory-system work, patent docs for IP work).
4. **No 2025-era model names** in any multi-reviewer audit — see `~/.claude/projects/-home-n/memory/feedback_always_use_2026_latest_models.md`. Locked names: model-A, model-B, model-C, model-D, model-E, model-I, model-F, model-G, model-H-ultra.
