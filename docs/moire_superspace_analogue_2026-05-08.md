# Moiré-metals superspace formalism as a conceptual analogue for FIM tier universality

> **Status:** speculative / conceptual-analogues bucket. Per the
> nn_universe honest-framing rule (see
> `honest_framing_audit_2026-05-08.md`), every claim below is prefixed
> *conjectured* unless explicitly tagged as established.
> **Date:** 2026-05-08
> **Audience:** internal — prep for a NeurIPS 2026 footnote, not a
> primary citation.

## Reference paper

Nuckolls, K. P. *et al.* "Higher-dimensional Fermiology in bulk moiré metals." *Nature*, March 2026. DOI [10.1038/s41586-026-10173-8](https://doi.org/10.1038/s41586-026-10173-8).

Verified metadata:
- Aperiodic composite crystals `(Sr6TaS8)1+δ(TaS2)8` — exfoliatable, incommensurate-lattice van der Waals materials.
- Quantum-oscillation measurements map a Fermi surface with **40+ distinct cross-sectional areas** in a structurally simple bulk moiré metal.
- Authors interpret this as the bulk metal **encoding electronic properties of higher-dimensional superspace crystals** in ways paralleling well-established crystallographic methods for incommensurate lattices.

## Why this is conceptually adjacent to nn_universe

Both programs make the same **shape** of claim:

> A complicated observable structure of effective dimension *D*<sub>eff</sub> is the projection of a simpler generative structure of higher dimension *D*<sub>gen</sub> > *D*<sub>eff</sub>.

| Axis | Nuckolls 2026 (moiré metals) | nn_universe |
|---|---|---|
| What is projected | Aperiodic 3D crystal | NN's Fisher-Information eigenstructure |
| What it is projected from | Periodic crystal in superspace ℝⁿ, n > 3 | Higher-dimensional task / data manifold (conjectured) |
| Observable signature | 40+ Fermi-surface cross-sections in a "structurally simple" metal | Tier universality across unrelated tasks (CV < 15 %) |
| Math machinery | de Wolff / Janner / Janssen superspace formalism (established 1970s) | NTK + FIM tier analysis + finite-N corrections |
| Why it matters | Finite-*D* measurements reveal higher-*D* generative structure | FIM tier ratios reveal hidden manifold structure beyond nominal width *N* (conjectured) |

The **superspace-to-observable** principle was formalised for quasicrystals
in the 1970s and is now considered a standard crystallographic technique.
Nuckolls 2026 extends it to bulk moiré metals. nn_universe's FIM tier
universality is *conjectured* to live in the same conceptual family,
but in NN parameter space rather than real-space lattice.

## Where it might help nn_universe (Tier-1 work, conjectural)

### 1. The 0.5 → 0.566 NTK anomaly (proof-ladder Tier-1 item 3)

Standard NTK predicts tier-ratio scaling **N^0.5**; we measure **N^0.566**.
The superspace formalism is precisely a framework where *the apparent
scaling exponent in D dimensions is wrong because the generative structure
lives in D' > D dimensions, and the discrepancy recovers the integer
value once the projection is accounted for*.

Concrete actions worth a literature pass before any claim:
- Aubry, S. (1980) — incommensurate-system exponents.
- Bak, P. (1982) — commensurate / incommensurate phase scaling.
- Lifshitz, R. (2003) — the modern statement of the superspace formalism.

**What we'd need to demonstrate to make this more than analogy:**
a calculation showing that an NTK-derived tier exponent shifted by a
finite-*N* projection-correction term recovers 0.566 (or its
nearest-rational embedding). Until that calculation exists, the
anomaly stands and the bridge is metaphor.

### 2. "40 cross-sections from a structurally simple metal" as motivation

This is a clean physical existence proof of the *principle* that "wider
or deeper systems with simple nominal architecture can encode more
effective modes than their parameter count suggests." It is *not* a
mechanism import — Bloch theory and magnetotransport do not translate
to NN training dynamics. But it can sit in the related-work paragraph
as a single sentence motivating why the FIM tier-hierarchy claim is not
unphysical on its face.

## Where it would NOT help nn_universe

- **Not a method import.** Bloch theory, Fermi liquid theory, magnetotransport — none of these map onto NTK or FIM analysis. The math is genuinely different.
- **Not a falsifier.** Tier-2 falsification of nn_universe still routes through ELT-HIRES α-drift (proof-ladder item 5), not through condensed-matter measurements.
- **Not a primary citation.** The paper does not appear in `references.bib` and should not be promoted to primary status. A footnote in the related-work section, prefixed *conjectured analogue*, is the maximum appropriate weight.
- **Don't conflate the two senses of "higher-dimensional."** Nuckolls's superspace lives in real-space ℝⁿ; nn_universe's higher dimensions are abstract task-manifold dimensions in NN parameter space. Mixing the two without distinguishing them will get the paper rejected by careful reviewers.

## Approved phrasing if cited

For the NeurIPS 2026 paper draft, the only acceptable bridge phrasing is:

> "Bulk moiré metals (Nuckolls *et al.* 2026) provide a recent physical
> realisation of the broader principle that finite-dimensional
> observables can carry the imprint of higher-dimensional generative
> structure; the FIM tier hierarchy is *conjectured* to be an
> analogous projection in NN parameter space, though the mechanisms
> are unrelated."

The word "conjectured" is required by the nn_universe honest-framing
rule and stays in the sentence until either:
1. We produce a Tier-1 calculation showing the projection structurally,
   or
2. We drop the analogy from the manuscript entirely.

## References (not yet in `references.bib`)

- Nuckolls, K. P. *et al.* (2026). Higher-dimensional Fermiology in bulk moiré metals. *Nature*, **March 2026**. DOI [10.1038/s41586-026-10173-8](https://doi.org/10.1038/s41586-026-10173-8).
- de Wolff, P. M. (1974). The pseudo-symmetry of modulated crystal structures. *Acta Crystallographica A* **30**, 777.
- Janner, A. & Janssen, T. (1977). Symmetry of periodically distorted crystals. *Phys. Rev. B* **15**, 643.
- Aubry, S. & André, G. (1980). Analyticity breaking and Anderson localization in incommensurate lattices. *Ann. Israel Phys. Soc.* **3**, 133.
- Lifshitz, R. (2003). Quasicrystals: a matter of definition. *Found. Phys.* **33**, 1703.
