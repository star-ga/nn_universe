# Simulation resolution limits: the lattice-spacing UV cutoff and its observable signature

> **Status:** speculative / conceptual-analogues bucket. Per the nn_universe
> honest-framing rule (see `honest_framing_audit_2026-05-08.md`), every claim
> below is prefixed *conjectured* unless tagged as established.
> **Date:** 2026-06-03
> **Audience:** internal — prior-art anchor for the Appendix D out-of-scope
> substrate framing; not a primary citation.
> **Source pointer (internal-only, per no-public-attribution rule):**
> Beane, Davoudi, Savage, "Constraints on the Universe as a Numerical
> Simulation," Eur. Phys. J. A 50 (2014) 148; arXiv:1210.1847 (2012).

## The intel

A foundational observation for any "universe-as-computation" framing:
**a numerical simulation on a discretised grid has a finite resolution.** If the
substrate is a lattice, the lattice spacing $a$ sets a shortest resolvable
length and therefore a highest resolvable momentum — an ultraviolet (UV) cutoff
at $\Lambda \sim \pi/a$. This is not metaphor; it is the standard, established
behaviour of lattice field theory.

The 2012 argument takes this seriously as a *physical* hypothesis: if our
universe is itself such a simulation, the underlying lattice would leave
**observable fingerprints**, and those fingerprints are in principle
falsifiable. That is the part worth recording — it converts a philosophical
question into a (hard, but real) empirical one.

## The core chain (established lattice physics → conjectured cosmological signal)

1. **(Established.)** Lattice QCD discretises spacetime; the spacing $a$ imposes
   a momentum-space cutoff $\Lambda \sim \pi/a$. All lattice observables carry
   $O(a)$ / $O(a^2)$ discretisation artefacts that vanish only in the continuum
   limit $a \to 0$. This is exactly the refinement series the repo already
   runs (`experiments/v2_0_lattice/lattice_refinement.py`: $a_n = a_0/2^n$).

2. **(Established, within the hypothesis.)** A simulated universe built on such a
   lattice would carry the same cutoff: there would be a maximum energy /
   momentum that any propagating mode can have.

3. **(Conjectured / model-dependent.)** Beane–Davoudi–Savage identify the
   highest-energy cosmic rays as the natural probe. A lattice cutoff would
   manifest as:
   - a **cutoff in the cosmic-ray energy spectrum** near the lattice scale, and
   - a **direction dependence (anisotropy)** in cosmic-ray propagation aligned
     with the underlying lattice axes — because a cubic lattice breaks exact
     rotational invariance, leaving a residual hypercubic symmetry.
   They note the suggestive coincidence that the observed Greisen–Zatsepin–Kuzmin
   (GZK) cutoff sits near the highest-energy events, while stressing the GZK
   cutoff has a conventional astrophysical explanation — so this is a
   *consistency target*, not evidence. (See the next section: the cutoff is a
   real measurement, but the *distinguishing* signature is the anisotropy, not
   the cutoff itself.)

## The empirical hook: the GZK cutoff is measured, not hypothetical

The strongest part of the argument is that the resolution-limit *probe already
exists in data*. The observable universe has a real, measured high-energy
cutoff in its cosmic-ray spectrum:

- **(Established — observation.)** The cosmic-ray flux falls off sharply above
  roughly $5\times10^{19}$ eV ($\sim$50 EeV). This suppression is observed by
  multiple independent experiments (HiRes 2008; Pierre Auger Observatory;
  Telescope Array). The existence of a cutoff near this energy is not in
  dispute.
- **(Established — standard explanation.)** The conventional account is the
  **Greisen–Zatsepin–Kuzmin (GZK) mechanism**: protons above the threshold lose
  energy by photo-pion production off cosmic-microwave-background photons
  ($p + \gamma_{\text{CMB}} \to \Delta^+ \to p/n + \pi$), so high-energy protons
  from cosmological distances are attenuated. This requires *no* new physics and
  fully accounts for the observed cutoff. Source composition (heavier nuclei at
  the highest energies) modifies the picture but does not need a lattice either.
- **(Conjectured / model-dependent — the simulation reading.)** Beane–Davoudi–
  Savage note only that *if* spacetime were a lattice, its UV cutoff
  $\Lambda \sim \pi/a$ would *also* truncate the cosmic-ray spectrum, and that
  the observed cutoff sits suggestively near where the highest-energy events
  occur. The honest framing: a measured cutoff is a **necessary-but-not-
  sufficient** consistency check. It is consistent with a lattice — but it is
  far better explained without one, so it is **not** evidence for the
  hypothesis. The genuinely *distinguishing* prediction would be the
  **hypercubic anisotropy** (a preferred-axis direction dependence in
  ultra-high-energy cosmic-ray arrival/propagation), for which there is no
  established positive signal.

**Bottom line.** The cosmic-ray cutoff is the right *kind* of observable — it is
real, it is at the right scale to probe an extreme UV cutoff, and it makes the
hypothesis falsifiable rather than philosophical. But the cutoff alone is fully
absorbed by GZK; only an axis-aligned anisotropy would point past conventional
astrophysics, and that is not seen.

## Why this belongs in nn_universe

nn_universe's whole framing is the universe as **deep-layered-sequential
computation** (Appendix D). The Beane line of argument is the *spatially-
parallel-lattice* sibling hypothesis: same "universe-as-computation" genus,
different computational class.

That contrast is the useful part:

- **The lattice hypothesis** predicts a *spatial* discreteness signature
  (UV momentum cutoff + rotational-symmetry breaking). Its falsification handle
  is high-energy astrophysics.
- **The deep-layered-sequential hypothesis** (this repo) predicts a *spectral /
  information-geometric* signature in the substrate's computational class
  (the FIM tier hierarchy), not a spatial-anisotropy signature.

*Conjectured.* The two are not mutually exclusive — a substrate could be both
sequentially deep and spatially discretised — but they are **distinguishable by
their predicted signatures**, and the repo's contribution is to articulate the
sequential-depth signature that the lattice literature does not address. The
lattice resolution-limit argument is the natural foil to position against.

## What is established vs. conjectured

- **Established:** lattice discretisation imposes a UV cutoff $\Lambda \sim \pi/a$
  and breaks continuous rotational invariance down to a hypercubic subgroup;
  discretisation artefacts scale as $O(a^n)$ and the continuum limit is $a\to 0$.
- **Established (intellectual-history fact):** the 2012 paper proposed cosmic-ray
  spectrum cutoff + anisotropy as falsifiable simulation-lattice fingerprints.
- **Conjectured / model-dependent:** that our universe *is* such a lattice; that
  the observed GZK cutoff is anything other than conventional astrophysics; that
  any current data constrains the hypothesis at a meaningful level.
- **Out of scope for nn_universe:** we make *no* claim about a spatial lattice or
  cosmic-ray signatures. We record this only as the contrasting hypothesis that
  sharpens what the sequential-depth (FIM-tier) signature is and is not.

## See also

- `experiments/v2_0_lattice/lattice_refinement.py` — the repo's own lattice
  refinement series ($a_n = a_0/2^n$); the same $a\to 0$ continuum-limit
  discipline the resolution-limit argument depends on.
- `docs/fim_tier_hierarchy_neurips2026.md` Appendix D — out-of-scope substrate
  framing; the sequential-depth signature this note contrasts against the
  lattice signature.
- `docs/free_parameters_26_constants_2026-06-03.md` — sibling
  conjectural-bucket note (free-parameter benchmark for the substrate hypothesis).
- `docs/moire_superspace_analogue_2026-05-08.md` — sibling conjectural-bucket
  note (same honest-framing discipline).
- `docs/honest_framing_audit_2026-05-08.md` — the rule that governs all of these.
