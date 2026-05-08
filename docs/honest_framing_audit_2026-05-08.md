# Honest-framing audit — 2026-05-08

**Scope:** All `*.md` documents in `nn_universe/` repo root + `docs/` —
README.md, AGENTS.md, fim_tier_hierarchy_neurips2026.md (the paper),
cover_letter.md, submission_checklist.md, preregistration_v2.md,
neurips_reproducibility_checklist.md, and the new
`experiments/v11_runpod_pythia_moe/multiseed_v2_iter262_2026-05-07/V11_7_MULTISEED_STATS.md`.

**Frame:** every claim must fit one of the four Tier-1 .. Tier-4 buckets
(memory `project_nn_universe_proof_ladder.md`):
  - Tier 1: H200-scope, deliverable now
  - Tier 2: external infrastructure, 12–24 mo
  - Tier 3: real proof criteria (4D spacetime, Lorentz, quantum from
    information geometry, falsifiable GR deviation)
  - Tier 4: honest endpoint — "universe IS a neural network" not
    attainable empirically; only predictive advantage + ontological
    economy + 30-yr falsification survival count

## Findings

**No overclaim hits.** Every Tier 3+ surface in the repo is properly
hedged:

| Surface | Tier 3+ topic | Framing |
|---|---|---|
| `fim_tier_hierarchy_neurips2026.md:35` | claim hierarchy | explicit "(proved · empirically supported · conjectural)" stratification |
| `fim_tier_hierarchy_neurips2026.md:80` | mechanism | "closed-form proved (Theorem 1', Appendix B)" — math proof, valid |
| `fim_tier_hierarchy_neurips2026.md:126` | Hanin–Nica scope | "Scope clarification ... Its application to trained MLPs, boolean circuits, attention transformers, MERA tensor networks, and BatchNorm ResNets is *empirical* in this paper" — explicit narrowing |
| `fim_tier_hierarchy_neurips2026.md:517` | 4D spacetime | "not addressed empirically; remains an open theoretical question in the parent framework" |
| `fim_tier_hierarchy_neurips2026.md:525` | universality | "Empirical extension and explicit narrowing" — narrows to 5 substrate classes that pass + names GPT-Tiny tied-embedding as a directional failure |
| `cover_letter.md:40` | α-drift | "null expectation at current archival-data noise levels" |
| `submission_checklist.md:66` | UVES + HIRES α-drift | "out of scope for this paper" |
| `submission_checklist.md:67` | 4D spacetime emergence | "out of scope, noted as open" |
| `preregistration_v2.md:15`, `:55`, `:84` | α-drift real-data | three sections all framed as conjectural-pending-data; pre-registered decision rules locked before access |
| `V11_7_MULTISEED_STATS.md` (this loop's ship) | dense-vs-MoE separation | explicit "Honest framing" section listing what the run proves vs does NOT prove (Tier 1 only) |

**Tier 3 references** (cosmological constant, EFT black holes — `[38]
H. Cohen et al. 1999`) appear **only in the references list**, not as
results claims. Properly cited as parent-theory background.

**Tier 4 framing.** No surface in the repo claims "the universe IS a
neural network" without explicit conjectural prefix. The paper's
contribution is bounded as "FIM-tier-ratio universality across deep
sequential composition" (a Tier-1 empirical result with a
closed-form mechanism), not the Tier-4 endpoint.

## Pin candidate (future T1 work)

The honest-framing discipline is currently enforced by **author
discipline + reviewer eyes**. A grep-based pin would catch regressions:
flag any sentence in `*.md` that contains the bare phrase "universe is
a neural network" / "is a neural network" without an immediate
qualifier ("conjectural" / "hypothesized" / "open" / "Tier 4" /
"theoretical" within ±60 chars). Same iter-232 single-source-of-truth
→ derived-surface drift class — applied at the framing-discipline
layer.

Not adding the pin in this audit pass (scope creep). Logging as a
future iter T1 candidate for nn_universe parallel work.

## Audit verdict

**PASS** — no edits required. The paper, README, preregistration,
cover letter, submission checklist, and reproducibility checklist all
pass the four-tier framing discipline as of 2026-05-08.

## Reproduction

```bash
# (from nn_universe repo root)
grep -rnE "(prove|proven|proves|established|definitively|conclusive|demonstrates|universe is a|emergent.{0,30}spacetime|Lorentz|cosmological constant)" \
  *.md docs/*.md | grep -v README_archive | less
# Verify each hit either has a math-proof context, an explicit
# scope/narrowing qualifier, or is in the references list.
```
