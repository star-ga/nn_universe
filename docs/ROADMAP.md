# nn_universe — Master Roadmap

**Last updated:** 2026-05-13
**Current status:** NeurIPS 2026 main-track submission packaged (9 pages, anonymous, pre-submission rigor review complete). V12 cluster follow-up planned, scripts landed, awaiting compute. All in-paper wordsmithing gaps closed.

This is the single canonical place for "what's left, in what order, on what hardware, at what cost". Sub-docs below have detail.

---

## TL;DR

```
SUBMISSION-READY now  →  /data/checkpoints/neurips2026_submission/paper.pdf  (9pp, anonymous)
                          docs/fim_tier_hierarchy_neurips2026.md  (source)

NEXT SUBMISSION-WORK  →  V12 cluster follow-up (camera-ready / journal track)
                          docs/cluster_roadmap_v12.md  (plan)
                          docs/preregistration_v3.md  (locked predictions + falsifiers)
                          scripts/run_v12_cluster.sh  (driver)
                          experiments/v12_*/run.py     (all 5 scripts landed)
```

---

## Stage 0 — Submission state (DONE)

| Artifact | Path | Status |
|---|---|---|
| Paper PDF (main) | `/data/checkpoints/neurips2026_submission/submission/paper.pdf` | 9 pages, NeurIPS 2026 header, anonymous |
| Supplementary PDF | `/data/checkpoints/neurips2026_submission/submission/supplementary.pdf` | 10 pages |
| Submission bundle | `/data/checkpoints/neurips2026_submission/submission.zip` | refreshed |
| Markdown source | `docs/fim_tier_hierarchy_neurips2026.md` | V11.8 + rigor-review fixes |
| Reproducibility | `docs/neurips_reproducibility_checklist.md` | filled |
| Pre-registration v2 | `docs/preregistration_v2.md` | σ_min + α-drift |
| Pre-registration v3 | `docs/preregistration_v3.md` | V12 items 1–5 locked |
| Anonymity | `bcdde96` commit | passes |

Pre-submission rigor review complete. Honest-weak-accept territory; gaps that require new compute (not text edits) are enumerated in Stage 1 and pre-registered.

---

## Stage 1 — V12 cluster follow-up (SCRIPTS LANDED, awaiting compute)

Full plan: [docs/cluster_roadmap_v12.md](cluster_roadmap_v12.md). Orchestration driver: `scripts/run_v12_cluster.sh`. Locked preregistration: [docs/preregistration_v3.md](preregistration_v3.md).

### V12 commit ledger

| Commit | Artifact | Status |
|---|---|---|
| A | `experiments/v12_partition_invariant/run.py` — 13-substrate FIM with raw arrays + Gini/r_eff/top-1% (item 1) | ✅ landed |
| B | `experiments/v12_production_multiseed/run.py` — 5-seed probe-re-randomisation at ResNet-50 / ViT-L / GPT-2-large / Mamba-790M (item 3) | ✅ landed |
| C | `experiments/v12_probe_convergence_large.py` — probe-convergence sweep on Pythia-2.8B at n ∈ {50,100,200,400,800,1600} (item 4) | ✅ landed |
| D | `experiments/v12_nondeep_control/run.py` — 300M-param RFF kernel ridge control (item 5) | ✅ landed |
| E | `experiments/v12_lm_loss_fim/run.py` — real LM-loss FIM on Pythia / OLMoE / Mamba via Pile validation (item 2) | ✅ landed |
| F | `docs/preregistration_v3.md` — V12 items 1–5 with locked decision rules + falsifiers | ✅ landed |
| G | `experiments/v12_partition_invariant/aggregate.py` + `decision_rules.py` — verdict pipeline | ✅ landed |
| H | `docs/v12_results.md` — post-run writeup | ⏳ blocked on compute |
| I | Camera-ready paper integration (move Theorem 1' to main, integrate V12 results) | ⏳ blocked on compute |

All V12 commits **A–G are scripts only — no compute fired yet**. They are runnable on either local 4070×2 or rented Runpod H200.

### To start V12 compute

When ready (user authorisation required):
```bash
# Path A — rent 1× H200 ($32, 8 hours total)
scripts/run_v12_cluster.sh h100         # prints the runpod recipe

# Path B — local 4070×2 (free, 5-7 days)
ssh-add ~/.ssh/<host_b_key>             # ensure SSH to second 4070 box works
scripts/run_v12_cluster.sh dual host_b  # embarrassingly-parallel orchestrator

# Path C — single GPU here (subset only)
scripts/run_v12_cluster.sh local
```

The orchestration script is idempotent. SHA-256-hash the preregistration_v3.md + scripts directory and add to the commit message BEFORE the first compute fires (item-by-item locks per `docs/preregistration_v3.md` §"Locked at commit time").

---

## Stage 2 — Camera-ready (after NeurIPS 2026 accept)

If accepted (probability: weak-accept-likely given §5.3 honesty):
1. Run V12 (Stage 1) — 5–7 days
2. Integrate V12 results — Commit I above
3. Add acknowledgements (post-anonymity): named advisors if any (Visvanathan Ramesh case-by-case)
4. Deanonymise: re-add `Anonymous Author(s)` → real author block in `wrapper.tex`
5. License: restore from `[License removed for anonymous review]`
6. Final pre-submission rigor review on camera-ready PDF
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
4. Pre-submission rigor review notes are stored locally only; this public repo carries the locked predictions + falsifiers (`docs/preregistration_v3.md`), not the review correspondence.
