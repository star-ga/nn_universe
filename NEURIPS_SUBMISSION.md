# NeurIPS 2026 — Submission Instructions

This document is the operator's runbook for submitting the paper to NeurIPS 2026 main track. The paper source is `docs/fim_tier_hierarchy_neurips2026.md`.

> **Author:** Nikolai Nedovodin · STARGA Inc. · `nikolai@star.ga`
> **Title:** *Fisher Information Tier Hierarchy: A Universality Signature of Deep Layered Sequential Computation*
> **Venue:** NeurIPS 2026 main track (primary) → NeurIPS 2026 Foundations of Deep Learning workshop (fallback) → ICLR 2027 (secondary fallback)

---

## 1. Where to submit

NeurIPS 2026 uses **OpenReview** for paper handling.

- **Submission portal:** `https://openreview.net/group?id=NeurIPS.cc/2026/Conference`
- **Author guide:** `https://neurips.cc/Conferences/2026/CallForPapers`
- **CMT? No** — NeurIPS has been on OpenReview since 2024 and remains there.

You will need:

1. An **OpenReview profile** linked to the corresponding-author email (`nikolai@star.ga`). If a profile already exists, log in. Otherwise create one at `https://openreview.net/signup`. Profile activation can take 24 hours, so do this *at least 48 hours* before the deadline.
2. A **Google Scholar / DBLP / ORCID** identifier linked to the profile (OpenReview cross-checks for conflict-of-interest).
3. The paper's **OpenReview submission form** filled out: title, abstract, authors (yourself), keywords, primary subject area (`cs.LG → Theory`), TLDR (1-sentence), conflict-of-interest disclosures.

## 2. Critical NeurIPS 2026 deadlines

> Verify these against the official CFP at submission time — dates below reflect the NeurIPS-2026 schedule typical for past years and **must be confirmed**.

- **Abstract registration:** ~7 days before full paper deadline. Register the title + abstract on OpenReview; the full paper PDF is uploaded after.
- **Full paper submission:** typically mid-to-late **May 2026** for NeurIPS 2026 December conference.
- **Reproducibility checklist:** required, completed inline in the submission. Ours is at `docs/neurips_reproducibility_checklist.md` — copy-paste into the OpenReview form.
- **Supplementary material:** uploaded with the main paper as a separate ZIP. Code-repository link is allowed; anonymized fork is recommended (see §5).
- **Author response window:** ~2 weeks after reviews. Limited rebuttal length.
- **Final notification:** typically September.
- **Camera-ready due:** typically October.

**Action:** check `https://neurips.cc/Conferences/2026` for the **exact** dates and put each on the calendar. Set reminder 14 days before each.

## 3. Paper format requirements

NeurIPS has a strict formatting policy. The official LaTeX style file is mandatory:

- **Style file:** `https://media.neurips.cc/Conferences/NeurIPS2026/Styles.zip` (will be released with the CFP — typically a `neurips_2026.sty` file).
- **Page limit:** 9 pages of main content + unlimited references. Appendices go in the supplementary material (no page limit on supplementary).
- **Paper size:** US Letter, single-column (NeurIPS-2026 inherits the 2024+ template).
- **Font / spacing:** as set by the .sty file. Do NOT modify margins, line spacing, or font.
- **Figures:** vector PDF/EPS preferred. Raster fine at ≥ 300 DPI.

### Converting our Markdown source to NeurIPS LaTeX

Our source is `docs/fim_tier_hierarchy_neurips2026.md`. To produce a NeurIPS-formatted PDF:

```bash
# 1. Get the official template (when CFP is open)
mkdir -p /tmp/neurips26 && cd /tmp/neurips26
wget https://media.neurips.cc/Conferences/NeurIPS2026/Styles.zip && unzip Styles.zip
# 2. Convert markdown → tex skeleton via pandoc
pandoc /home/n/nn_universe/docs/fim_tier_hierarchy_neurips2026.md \
    --bibliography=/home/n/nn_universe/docs/references.bib \
    --citeproc \
    -o /tmp/neurips26/paper.tex \
    --template=neurips_2026.tex
# 3. Manually port the §1.5 contributions, §3.5 falsifiability ladder,
#    §4.5 Proposition 1, and Appendix B Theorem 1 into LaTeX equation
#    environments (the markdown $$...$$ blocks need \begin{equation} ... \end{equation}).
# 4. Compile: lualatex paper.tex; lualatex paper.tex
```

Plan to spend ~1 day on the LaTeX port and ~half a day on figure regeneration.

## 4. Anonymisation (double-blind review)

NeurIPS is **double-blind**: the submitted PDF and supplementary must not identify the authors.

**What to strip from the paper PDF:**
- All "STARGA, Inc." mentions in the title block (replace with "Anonymous").
- The git commit string at the top.
- Author email + STARGA contact info.
- Repository URL `github.com/star-ga/nn_universe` — do NOT include the live URL. Instead, point to an *anonymised* GitHub fork or to OpenReview supplementary.

**Two safe options for code:**
1. **Anonymous GitHub fork:** create a new throwaway GitHub account (`fim-tier-anon-2026`), fork the repo as a zip, push to a public throwaway repo. Cite that URL in the paper. After acceptance, the deanonymised version replaces it.
2. **OpenReview-only supplementary:** upload `code.zip` directly to the OpenReview submission. No external URL needed.

We recommend option 2: it is fully self-contained, cannot leak through GitHub history, and is what most NeurIPS submissions do.

**Anonymisation checklist before upload:**
```bash
# In the paper draft and any supplementary doc:
grep -lirE "STARGA|Nedovodin|nikolai|star\.ga|Nikolai" \
   docs/fim_tier_hierarchy_neurips2026.md docs/h200_cluster_runbook.md \
   docs/v6_0_mechanism_hanin_nica.md docs/preregistration_v2.md \
   docs/neurips_reproducibility_checklist.md docs/v1_1_ntk_gap_closure.md
# → expect 0 results in the SUBMISSION COPY (work on a copy, not the master)
```

## 5. What to upload — submission package

Build the submission ZIP `fim_tier_hierarchy_neurips2026_submission.zip`:

```
fim_tier_hierarchy_neurips2026_submission.zip
├── paper.pdf                 # 9-page anonymised main paper
├── supplementary.pdf         # Appendix A–D + reproducibility checklist
├── code/
│   ├── README.md             # how to run; one-line `bash run_all.sh`
│   ├── run_all.sh
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── experiments/          # all V1–V9 experiment scripts
│   ├── tests/
│   └── docs/claim_manifest.json
└── data/
    └── (none required — all results regenerated by run_all.sh)
```

Build script:

```bash
cd /home/n/nn_universe
mkdir -p /tmp/neurips_submission/code
cp -r experiments tests pyproject.toml run_all.sh /tmp/neurips_submission/code/
cp docs/claim_manifest.json /tmp/neurips_submission/code/docs/
# Strip identifying info from code/README.md (use a separate anonymised copy)
# Build paper.pdf and supplementary.pdf via the LaTeX template (§3 above)
cp paper.pdf supplementary.pdf /tmp/neurips_submission/
cd /tmp && zip -r fim_tier_hierarchy_neurips2026_submission.zip neurips_submission/
```

## 6. Reproducibility checklist (mandatory)

Fill out `docs/neurips_reproducibility_checklist.md` directly into the OpenReview form. The form has separate fields for: experiment scope, dataset access, compute envelope, hyperparameters, statistical-significance methodology, code availability.

## 7. After submission

- **OpenReview emails** when reviews are posted (~6 weeks after submission deadline).
- **Author response window:** typically 1 week, 1 PDF response per paper, max ~5000 characters.
- **Reviewer discussion phase:** a further 1–2 weeks.
- **Final decision:** ~3 months after submission.
- If rejected from main track, NeurIPS automatically offers consideration for the workshop track — accept that path if main rejects.

## 8. Anti-checklist (things NOT to do)

- ❌ Do **not** force-push commits with non-STARGA authorship to the public repo before the camera-ready deadline.
- ❌ Do **not** include any reviewer-LLM-evaluation traces (already deleted from the repo as of the cleanup commit).
- ❌ Do **not** disclose the anonymous-fork URL on social media until after notification — that breaks blinding.
- ❌ Do **not** submit any work that has been arXiv'd less than 30 days before the submission deadline (NeurIPS dual-submission rule). Our paper has not been arXiv'd → safe.
- ❌ Do **not** miss the abstract-registration deadline. Title + abstract are required ~7 days before the full paper deadline; without it, you cannot submit the full paper.

## 9. Contacts and authority

- **Submitting author:** Nikolai Nedovodin (STARGA Inc., `nikolai@star.ga`).
- **Funding statement:** none (self-funded research at STARGA, Inc.).
- **Conflicts of interest:** no employment / consulting / family relationships with NeurIPS-2026 area chairs or PC members.
- **Ethical concerns:** none (synthetic data, no human subjects, no sensitive applications). Section 7 of the paper is the broader-impact statement.

---

*Operator runbook v1, 2026-04-25. STARGA, Inc.*
