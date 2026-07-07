# The cosmological constant as the hardest free-parameter, and what a "missing law" would have to look like

> **Lead (the one-line takeaway):** *the missing law is **global, not local**.*
> The cosmological constant problem signals that a constraint is missing in how
> quantum field theory connects to gravity — and the survey-grounded best guess is
> that the constraint is **global / topological / measure-theoretic**, acting on
> the whole gravity–QFT seam, not a new *local* field. Weinberg's no-go theorem
> forbids only the **local** adjustment class, so it does not block a global
> constraint; the live programs that route around it — **unimodular gravity**,
> **sequestering (Kaloper–Padilla)**, and the **Donoghue** non-running-$\Lambda$ /
> non-local-partner results — all share exactly this global character. This is a
> research *direction*, not a derivation, and not a paradox or unified-physics
> claim.

> **Status:** speculative / conceptual-analogues bucket. Per the nn_universe
> honest-framing rule (`honest_framing_audit_2026-05-08.md`), every claim is
> tagged *established* or *conjectured*; nn_universe makes **no** derivational
> claim here.
> **Date:** 2026-06-03
> **Audience:** internal — context note. Survey sources are deep-research reports
> (`/home/n/.openclaw/workspace/research/deep-research-cosmological-constant-2026-06-03.md`,
> `.../deep-research-cosmological-constant-survey-2026-06-03.md`, and
> `.../deep-research-donoghue-vacuum-energy-2026-06-03.pdf`), literature digests,
> not nn_universe results.
> **Scope:** LOCAL ONLY. Nothing here is committed or pushed.

## Why this note exists

The 26-free-parameter benchmark (`free_parameters_26_constants_2026-06-03.md`)
flagged the dark-energy density $\Omega_\Lambda$ / cosmological constant
$\Lambda$ as the *optional 27th* parameter — added whenever cosmology is folded
into the count. It is also, by a wide margin, the **most fine-tuned and least
understood** of the lot. This note records the current state of the
"is there a missing physical law for $\Lambda$?" question, because it is the
single sharpest instance of the open problem the substrate framing would
ultimately have to face: *deriving (or constraining) a measured constant that
the Standard Model takes as input.*

## The intel (established physics)

**Established.** The cosmological constant problem is not merely "one unlucky
fine-tuning." In Einstein gravity, **any** shift of the matter-sector vacuum
energy renormalizes $\Lambda$, so the technically sharp statement is one of
**radiative instability**: $\Lambda_{\rm eff} \sim \Lambda_{\rm bare} +
\rho_{\rm vac}$ (up to $8\pi G$ conventions). A naive Planck-scale zero-point
estimate gives $\rho_{\rm vac} \sim M_P^4$, against an observed
$\rho_\Lambda \sim 2.5\times10^{-47}\,\mathrm{GeV}^4 \approx 10^{-123}\,M_P^4$.

**Established (hedge).** The familiar "$10^{-120}$" slogan is
**convention-dependent** — $\approx 10^{-123}\,M_P^4$ in unreduced Planck units,
$\approx 7\times10^{-121}\,\bar M_P^4$ in reduced units; the energy-density
mismatch is usually quoted as $10^{120}$–$10^{123}$. The *integer* depends on
densities-vs-scales and on the Planck convention. (This is the same
convention-caveat discipline already applied to the 26-vs-25–31 count.)

## The analytic bottom line of the survey (reported, not endorsed)

> *If* a genuinely missing physical law exists, the best current guess is that
> it is a **global, topological, measure-theoretic, or nonlocal constraint
> governing the gravitational response to vacuum energy — not a conventional
> new local field.*

Reasoning chain from the survey:

- **Weinberg-type no-go.** Ordinary *local* adjustment mechanisms in 4D
  effective field theory do not generically self-tune the vacuum to Minkowski
  without hidden fine-tuning. This rules out the "just add a new particle /
  local interaction" class as a clean law.
- **Most law-like candidates** are therefore the ones that alter the
  source–curvature relation through a *global* rule: **vacuum-energy
  sequestering** and some **unimodular / global-constraint** formulations.
  These target the *old* problem (why large loop terms don't gravitate).
- **Symmetry-based cancellations (SUSY)** work only in exact limits; once broken,
  the vacuum energy reappears far too large.
- **Anthropic / landscape** explains a small positive $\Lambda$
  *environmentally*, not via a unique dynamical law.
- **Quintessence / most modified gravity** mainly address the *new* problem
  (why acceleration looks $\Lambda$-like now), not the old radiative-instability
  problem.

## Two EFT-of-gravity results worth pinning (established)

The Donoghue-lens digest adds two technically sharp points the first survey
glossed; both are *established*, and both **narrow** what a "missing law" can be.

- **The cosmological constant and $G$ do not physically "run."** A recurring
  hope is that $\Lambda$ flows from $M_P^4$ down to the observed value along an
  RG trajectory. Donoghue's analysis: the $\mu$-dependence of these couplings
  in dimensional regularization is **not** physical running — in actual
  scattering amplitudes the $\mu$-dependence is absorbed by tadpoles and local
  counterterms, and the constants do not dynamically scale down. So "$\Lambda$
  just runs to small" is **closed** as a clean mechanism. (Also undercuts the
  naive quartic-cutoff slogan: a careful path-integral measure cancels the
  $\Lambda_{\rm cut}^4$ piece, removing one historical motivation for SUSY —
  but the phase-transition / massive-particle vacuum energy *remains*, so the
  problem is **not** dissolved.)
- **The non-local partner to $\Lambda$.** A massive loop in curved background
  generates, alongside the usual local counterterms, a **finite, non-local,
  zeroth-order ($O(p^0)$)** term in the gravitational effective action whose
  coefficient is **strictly fixed by the particle mass** ($\sim m^4$) and
  **cannot be fine-tuned away**. It mimics the geometric scale of $\Lambda$ but
  is not a free parameter. Reading reported, not endorsed: this is offered as
  evidence that gravity responds to vacuum structure **non-locally**, so any
  purely-local reductionist account of $\Lambda$ is incomplete — consistent with
  (but stronger-worded than) the first survey's "global/topological, not local"
  guess. nn_universe endorses neither the strength of the claim nor a mechanism.

## Observational state, June 2026 (established)

- Planck 2018 + BAO + SNe: $w_0 = -1.03 \pm 0.03$ — consistent with a true
  cosmological constant.
- DESI DR2 BAO: $2.8$–$4.2\sigma$ preference for **evolving** dark energy
  (depending on SNe compilation); extended analysis prefers a **phantom
  crossing** over canonical $w \ge -1$ quintessence.
- GW170817: $|c_g/c - 1| \le 5\times10^{-16}$ — kills broad scalar-tensor
  dark-energy classes; GWTC-5.0 LVK cosmology sees no modified GW propagation.

Net: the perturbation data is still very $\Lambda$-like, which *squeezes*
modified-gravity solutions and (per the survey) favours research into a global /
topological cancellation law that leaves $\Lambda$-like clustering intact.

## The three concrete sub-directions of the global-constraint program (reported, not endorsed)

If the missing law is global rather than a new local field, the survey-level
guess sharpens into **three** specific places to look. The common premise:
macroscopic gravity is **not** the blind sum of microscopic quantum parts, so
the constraint lives in how the whole couples, not in any local term.

- **Emergent / thermodynamic gravity.** Treat the Einstein equations as an
  equation of state of underlying degrees of freedom (Jacobson-type, entropic /
  holographic). Then $\Lambda$ is fixed by an emergent constraint on the
  macroscopic state, not summed from UV modes — the cancellation is a property
  of the emergent description, not a tuned counterterm.
- **Exact conformal symmetry.** A symmetry that forbids a fixed scale in the
  gravitational sector ('t Hooft-type exact local conformal / Weyl invariance)
  would make a hard $\Lambda$ symmetry-protected to zero; its small observed
  value then comes from *controlled* breaking, paralleling how chiral symmetry
  protects light fermion masses. (Note the SUSY caveat above: protection only
  holds in the exact limit — the open question is whether a conformal protection
  survives realistic breaking where SUSY does not.)
- **Macroscopic non-local interactions.** The Donoghue non-local partner term
  is the established footing here: gravity already responds to vacuum structure
  through finite, non-local pieces of the effective action. A missing law of
  this class would be an IR / horizon-scale non-local rule (Sorkin causal-set
  fluctuating-$\Lambda$, Barvinsky–Deser-type non-local gravity) that cancels
  the bulk vacuum energy globally while leaving local physics intact.

These three are **directions to watch, not a mechanism**. nn_universe endorses
none of them and derives nothing; they are recorded as the concrete content of
the "global, not local" lead.

## Structural relevance to the substrate hypothesis (conjectured, open)

*Conjectured.* In a deep-layered-sequential-computation framing (Appendix D),
the radiative-instability problem has a suggestive — and **only** suggestive —
shape: a quantity that *must* be cancelled across every UV contribution to an
exponentially smaller residual looks more like a **global normalization /
conservation constraint over the whole computation** than like a tunable local
weight. The survey's independent conclusion ("global/topological, not local")
rhymes with this, but:

- This is an **analogy, not a derivation.** nn_universe predicts nothing about
  $\Lambda$ and has no mechanism for sequestering.
- The substrate framing's *only* concrete signature claim remains the FIM tier
  hierarchy (deep-layered-sequential vs spatially-parallel QFT) — see the Beane
  contrast note. $\Lambda$ is **out of scope** of that signature.
- Recording this only sharpens the bar: the hardest of the ~26–27 constants is
  the one even mainstream theory cannot derive, so any strong-form "substrate
  derives the constants" claim must clear it last and hardest.

## What is established vs. conjectured

- **Established:** the cosmological constant problem is radiative instability,
  not just fine-tuning; $\rho_\Lambda \approx 10^{-123}\,M_P^4$; the
  "$10^{-120}$" figure is convention-dependent; local no-go arguments;
  current data is still close to $w=-1$ with a DESI-driven hint of evolution.
- **Conjectured / open:** that the missing law is global/topological
  (survey's *guess*, not a result); that the substrate framing bears on
  $\Lambda$ at all; that sequestering-like global constraints map onto any
  computational-normalization analogue.

## See also

- `docs/free_parameters_26_constants_2026-06-03.md` — the free-parameter
  benchmark; $\Lambda$ is the optional 27th and the hardest entry.
- `docs/beane_simulation_resolution_limits_2026-06-03.md` — the
  resolution-limit / cosmic-ray-cutoff signature; the substrate's *actual*
  falsifiable fingerprint (this note is explicitly out of that scope).
- `docs/fim_tier_hierarchy_neurips2026.md` Appendix D — out-of-scope
  cosmological framing.
- `docs/honest_framing_audit_2026-05-08.md` — the governing rule.
- Survey digests:
  `/home/n/.openclaw/workspace/research/deep-research-cosmological-constant-2026-06-03.md`,
  `.../deep-research-cosmological-constant-survey-2026-06-03.md`,
  `.../deep-research-donoghue-vacuum-energy-2026-06-03.pdf`.

## arXiv-worthiness verdict (internal, this batch)

Honest read after all six deep-research docs (#3 MIND arch; #4–#6 CC): **no
physics paper here.** Every load-bearing idea is already published by named
authors (Weinberg no-go, Padilla/Kaloper sequestering, Donoghue non-local
partner + no-physical-running, 't Hooft exact conformal, Brown–Teitelboim,
Sorkin causal-set). These are faithful *summaries*, not new derivations,
calculations, or predictions; entering a 40-year CC field at survey level is
not a fit. The **only** genuinely-defensible takeaway in the batch is the
research *direction* recorded in the lead above — **"the missing law is global,
not local"** — i.e. track the unimodular / sequestering / Donoghue global-constraint
cluster on the QFT–gravity seam. That is a direction to watch, not a result, and
nothing in this batch is publishable physics as-is.
