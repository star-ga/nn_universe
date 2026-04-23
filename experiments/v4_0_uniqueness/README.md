# V4.0 — Uniqueness Test for the FIM Three-Tier Hierarchy

**Status:** Experiment executable; production results pending.

---

## The question this experiment answers

V1.1 proved an NTK continuum-limit theorem. V1.2 measured a robust three-tier
FIM hierarchy in a 5-layer ReLU MLP across ten widths (16 → 8,192) with cross-
seed CV ≈ 10%. V2.1 showed the same hierarchy reappears in an independent task
(toric-code syndrome decoding), with task-dependent exponents but identical
structural form.

That combined result supports a universality claim. But it leaves a specific
open hole:

> Does the FIM three-tier hierarchy appear *only* in neural networks, or is it
> a generic information-geometric phenomenon that also shows up in non-learning
> parameterized systems?

Until this question is answered, every claim in the parent paper about
*neural-network universality* has a loose definition: the claim could be about
neural networks specifically, or about any parameterized system. The two
framings have very different implications for the "universe as neural network"
program — the first requires a learning substrate, the second is agnostic.

---

## What V4.0 does

We construct six parameterized systems, all with comparable internal dimension
(128–3,500 parameters), and compute a unified *parameter importance* measure
on each. For a likelihood-based model this measure reduces to the FIM diagonal;
for a deterministic system it is the output-Jacobian Gram-diagonal. The
tier-ratio (top-1% mean / bottom-50% mean) is computed on each, across
multiple random seeds.

Systems:

1. **Neural network** — 5-layer ReLU MLP on self-prediction (control / reference)
2. **Random Hermitian matrix** — GOE-like ensemble, parameters = upper-triangle entries
3. **Ising chain** — 1D classical spin chain, parameters = local fields
4. **Harmonic oscillator chain** — coupled 1D chain, parameters = spring stiffnesses
5. **Boolean circuit** — random softmax-mixture gate circuit, parameters = gate logits
6. **Cellular automaton** — Rule-110 elementary CA, parameters = initial cell states

Each system produces a scalar or vector output via a physically meaningful map
from its parameters; importance is the squared sensitivity of that output to
each parameter, averaged over a bank of random probe inputs.

## Three uniqueness tests

After seed-averaged tier ratios are computed, we run three independent
statistical tests:

1. **Magnitude.** Is the NN tier ratio distinguishable (>2σ) from the pooled
   distribution of non-NN ratios?
2. **Stability.** Does the NN seed-CV lie below every non-NN seed-CV?
3. **Classifier.** Can a two-feature logistic regression
   (`log_tier_ratio`, `top_1%_mass`) separate NN from non-NN samples with
   leave-one-out accuracy > 80%?

Ill-conditioned systems (those whose bottom-50% importance collapses to
numerical zero, such as Boolean circuits with dominant softmax branches) are
flagged as *degenerate* and excluded from pooled statistics, since their tier
ratios are not physically meaningful under this definition.

## Outcomes and interpretation

**If all three tests pass →** the FIM tier hierarchy is specific enough to
neural-network training dynamics that it can be statistically distinguished
from five alternative information-theoretic systems. This strengthens the
neural-network framing of the parent paper's universality claim.

**If one or more tests fail →** the tier hierarchy is not NN-specific. The
universality claim in the parent paper should be rephrased in substrate-
agnostic terms (e.g. *"any sufficiently parameterized system exhibits the
three-tier information-geometric hierarchy"*), which is a weaker but more
honest claim.

Either outcome is useful. This is a genuine falsification test of the
NN-specificity framing.

## Running the experiment

```bash
cd experiments/v4_0_uniqueness

# Quick check (3 seeds, 16 probes, ~3 min on CPU):
python3 run_uniqueness.py --seeds 3 --probes 16

# Production run (6 seeds, 32 probes, ~10 min on CPU):
python3 run_uniqueness.py --seeds 6 --probes 32

# Analyze and write markdown summary:
python3 analyze.py
```

Outputs:

- `v4_0_uniqueness_results.json` — per-seed tier ratios, masses, timings
- `v4_0_uniqueness_analysis.md` — three-test verdict + per-baseline comparison table

## Caveats

- All six systems have comparable but not identical parameter counts.
  The tier-ratio definition uses fractional cuts (top 1%, bottom 50%), so
  it's scale-invariant in principle, but finite-size effects dominate for
  the smallest systems (N < 100). Results are interpretable at the level
  of "NN tier ratio is ~1000× larger than any non-NN system" rather than
  "NN tier ratio is exactly 61,983".
- The Boolean-circuit importance can become ill-conditioned when one
  softmax entry dominates a gate (dormant branches). We mitigate this with
  small-initial-weight construction and the degenerate-flag in analysis.
- "Parameter importance" under the Jacobian-Gram definition is not
  identical to the classical FIM definition for stochastic models. For
  deterministic baselines this is the most natural generalization, and
  for the NN reference both definitions coincide at the diagonal to
  leading order.

## What this does *not* prove

- It does not prove the FIM tier hierarchy is a fundamental property of
  physics. It tests whether the hierarchy is NN-specific among parameterized
  systems.
- It does not cover the full space of possible counterexamples. A stronger
  test would include trained non-NN models (Ising under mean-field learning,
  for example), stochastic CAs, or quantum circuits. Those are future work.
- A *failure* of this test does not disprove the parent paper's empirical
  finding — it re-scopes the finding's interpretation.

## Connection to the parent paper

This experiment is listed as roadmap item **3** in the discussion section of
the parent paper's V1.2 extended-results appendix. It is the cheapest of the
four outstanding verification tasks (items 1–4 are: derive spacetime from NN
dynamics, predict a physical constant in advance, run this uniqueness test,
find the FIM tier signature in cosmological observations). V4.0 is the item
we can run on available hardware in ~10 minutes.
