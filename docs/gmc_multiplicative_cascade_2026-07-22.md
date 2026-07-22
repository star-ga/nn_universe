# Gaussian multiplicative chaos: shared mechanism, unshared machinery

> **Lead (the one-line takeaway):** *the log-normal depth-cascade underlying the
> √L result is an instance of a well-studied probabilistic family —
> multiplicative chaos — which has its own critical threshold and its own
> mature theory.* This upgrades the mechanism framing from "we observed a
> scaling law with a cited mechanism" to "the mechanism sits inside a named
> family." It does **not** supply a proof of the √L law, and the specific
> recent theorem does **not** transfer. Both halves are recorded below.

> **Status:** conceptual-framing bucket, *not* a result and *not* a citation
> for the current submission. Per the nn_universe honest-framing rule (see
> `honest_framing_audit_2026-05-08.md`), claims below are tagged *established*
> or *conjectured*. nn_universe makes **no** derivational claim here.
> **Date:** 2026-07-22
> **Audience:** internal — context note; candidate for post-acceptance
> theoretical framing (Stage 3), not for the anonymous submission.
> **Source pointer (internal-only, per no-public-attribution rule):**
> Lin, Qiu & Tan, *Harmonic analysis of multiplicative chaos Part I: the proof
> of the Garban–Vargas conjecture for 1D GMC*, arXiv:2411.13923v3 (Part II:
> arXiv:2505.03298v4). Garban–Vargas conjecture posted 2023. Kahane's
> T-martingale theory, 1985/87. Origin of this note: a popular-science article
> flagged by the operator 2026-07-22; the article's framing is narrowed below.
> **Scope:** LOCAL ONLY unless explicitly cleared. Nothing here is public.

## The intel

**Established (probability theory, not a nn_universe result).** Gaussian
multiplicative chaos (GMC) is a theory of random measures built as limits of
products of independent random factors. Kahane's T-martingale framework, quoted
from the primary source (arXiv:2411.13923v3 §1), takes a domain `U ⊆ ℝᵈ` with a
finite Radon measure `ν` and a sequence of independent random non-negative
functions `Pₙ(t)` with `E[Pₙ(t)] ≡ 1`, and defines

```
μₙ(dt) := [ ∏ₖ₌₁ⁿ Pₖ(t) ] ν(dt)          μ∞ = lim μₙ  (a.s. weak limit)
```

Recent work established the exact Fourier dimensions of all standard
**sub-critical** GMC on the unit interval, via a vector-valued martingale
method descended from the study of **Mandelbrot cascade** random measures.

## Why it is adjacent to this repo

**The mechanism is the same probabilistic object.** Our mechanism (V6.0,
`v6_0_mechanism_hanin_nica.md`) is a log-normal distribution over Fisher
diagonal entries with depth-linear log-variance, giving `log(T₁/T₃) ∝ √L`.
That is a product of independent per-layer multiplicative factors accumulated
along depth. The T-martingale product above is the same construction with the
scale index replaced by the layer index. GMC is the exponential of a Gaussian
field, i.e. log-normal by construction. This is not an analogy — the two are
instances of one family.

**The family has a critical threshold, and so do we.** GMC's scaling picture
holds only in the sub-critical regime; push the randomness parameter past a
critical value and the measure degenerates. The recent theorem is explicitly
scoped to sub-critical, and the critical case is open. Our own result has a
failure boundary: attention architectures and time-unrolled RNN/LSTM sit in the
deep-sequential band by magnitude but do **not** follow √L
(`fim_tier_hierarchy_neurips2026.md`, explicit narrowing).

## What does NOT transfer — recorded so it is not re-derived optimistically

**The theorem is Fourier-analytic and one-dimensional.** The quantity computed
is the polynomial decay rate of Fourier coefficients `μ̂(n)` of a random measure
on `[0,1]`. The machinery named in the paper's own keywords is Pisier's
martingale type inequalities and Littlewood–Paley decomposition, feeding a
random Fourier decoupling estimate. All of it needs a frequency variable to act
on. The FIM spectrum has no domain, no background measure `ν`, and no natural
Fourier variable — so there is nothing to take coefficients of. **The martingale
technique is not separable from the harmonic analysis it exists to feed.**

An earlier internal read suggested the martingale method might be the missing
proof strategy for √L. That is **retracted**: shared mechanism, unshared
machinery.

**The phase-transition parallel is confounded.** It is tempting to read our
attention/RNN failure as the same phenomenon as GMC's critical collapse. There
is a competing and more mundane explanation: attention is not a layered product
of independent factors at all, so multiplicative-cascade structure should not be
expected to apply in the first place. That would make it a *modelling-assumption*
failure, not a *phase-transition* failure. The two are indistinguishable from the
outside and are entirely different claims. **Conjectured, unresolved, and not to
be asserted either way.**

## Prior-art position

An arXiv full-text search conjoining "Gaussian multiplicative chaos" with
"neural network" returns **zero** results (checked 2026-07-22). Encouraging but
not conclusive — the query was a strict two-phrase AND, and the connection could
exist under "log-normal cascade", "Fisher information spectrum", or similar
without either exact phrase. Before any public framing that leans on this,
re-run the search across those alternatives.

## What would make this actionable (in priority order)

1. **Read Part II (arXiv:2505.03298), not Part I.** Part II is the *axiomatic*
   theory covering generalized Mandelbrot cascades; Part I is deliberately the
   concrete 1D special case. The checkable question is whether the depth-cascade
   satisfies Part II's axioms. Part I cannot answer that.
2. **Look for a variance parameter mapping.** GMC's sub-critical condition is
   stated in terms of a randomness parameter. If the depth-linear log-variance
   maps onto it, "do attention nets sit past the critical value?" becomes a
   yes/no question rather than a metaphor.
3. **Watch Part III.** The announced Part III handles the abstract setting with
   *abstract background measures* — the only place in the programme where the
   Euclidean-domain requirement is relaxed. Unpublished.

## Constraints on use

- **Not for the current submission.** The submission is packaged and under
  anonymous review (Stage 0); the roadmap's own rule 1 says do not edit the
  paper absent a camera-ready trigger. This is Stage-3 material at the earliest.
- **Cite as preprint.** The proof is not peer-reviewed.
- **Framing sentence only, if used at all.** The defensible public statement is
  that the depth-cascade belongs to a studied family with a known critical
  regime. Anything stronger — a proof strategy, a transferred theorem, a shared
  phase transition — is not supported by the primary source.
- Adopting a result on structural resemblance is the "design driven from analogy
  alone" failure this repo's honest-framing discipline exists to prevent. The
  resemblance here is real and confirmed at the mechanism level; the machinery
  is not, and that line is the whole content of this note.
