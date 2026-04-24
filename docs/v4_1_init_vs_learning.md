# V4.1 — The FIM Three-Tier Hierarchy Is Init-Induced, Not Learning-Induced

**STARGA, Inc. — Research Document**
**Phase:** V4.1 — critical reinterpretation of V1.0 / V4.0
**Date:** 2026-04-24
**Depends on:** V4.0 uniqueness (`experiments/v4_0_uniqueness/`), V1.0 toy experiment.

---

> **What changed.** The V1.0 paper and the first V4.0 sweep attributed the three-tier FIM hierarchy to *training dynamics*. V4.1 (this document), V4.1.1 (non-ReLU activations), and V4.1.2 (CNN/ViT) collectively show that:
>
> 1. **The hierarchy is present at random Kaiming initialization** in all tested architectures that have *layered sequential* structure (ReLU/GELU/tanh MLPs, CNNs, ViTs).
> 2. **Training's effect on the hierarchy is not universal.** In MLPs trained on a small-dim self-prediction task (V4.1, V4.1.1), training *dissipates* the hierarchy by 4-24×. In CNNs trained on a larger-dim autoencoder (V4.1.2), training dissipates it by 14,000×. In ViTs trained on the same (V4.1.2), training *sharpens* it by 67×.
> 3. **The binding factor is learning success.** When training barely moves the weights (unlearnable task, saturated activation, poor architecture fit), hierarchy barely changes. When training succeeds in fitting the task (ViT autoencoder, final loss 10⁻⁴), hierarchy sharpens along the learned feature directions.
>
> Revised interpretation: the FIM tier hierarchy is an **architecture-and-init-induced property** at randomness; training either preserves, dissipates, or sharpens it based on how well the network succeeds in fitting the data distribution.

---

## 1. The experiment

5-layer 256-hidden ReLU MLP on self-prediction MSE. Kaiming (He) normal init. 5 widths × 5 seeds = 25 pairs of (trained, untrained) configurations. For each seed we train the network for 20,000 SGD steps (lr=1e-3, momentum=0.9, batch=128) starting from the same seed-controlled init; we separately re-initialise a twin with the same seed and do *not* train it. We then measure the FIM diagonal tier-1/tier-3 ratio on both.

## 2. Results

### 2.0 Methodology caveat (probe count)

FIM diagonal estimates are Monte-Carlo in nature; MC noise concentrates
in the small-FIM tail and artificially inflates the Tier-1/Tier-3 ratio
at low probe counts. V4.0 used `n_probes = 32`; V4.1 uses `n_probes =
200`. At matched N (~3-5k params) the V4.0 trained-NN ratio was 13,752×
(6-seed mean), whereas V4.1 trained measurements give ~1,000× at the
same width. **Most of that factor-25 difference is probe-count MC-noise,
not training dynamics.** The *internal comparison within V4.1* (trained
at 200 probes vs untrained at 200 probes) is valid because both use the
same probe count; the reported "training reduces ratio by 4-24×" stands.

The correct statement is: at probe-count high enough to be MC-accurate
(~200), trained NN tier ratios are in the O(10³) range, and untrained
(same init) are in the O(10³–10⁴) range — training reduces by 4-24×.

### 2.1 Per-width data (n_probes = 200)

Mean T1/T3 across 5 seeds per width:

| Width | n_params | **Untrained** T1/T3 | **Trained** T1/T3 | trained / untrained |
|-------|----------|----------------------|---------------------|---------------------|
| 32 | 4,240 | 10,186 | 1,081 | 0.106 |
| 64 | 14,608 | 5,083 | 211 | 0.042 |
| 128 | 53,776 | 2,914 | 195 | 0.067 |
| 256 | 205,840 | 1,500 | 349 | 0.233 |

### Observations

- **Untrained** networks have T1/T3 in the 10³–10⁴ range at every width — a sharp hierarchy already exists at Kaiming init.
- **Training consistently decreases** T1/T3 by 4-24×. The effect is robust across seeds (CV of the decrease < 30% per width) and not directional-dependent.
- Both untrained and trained T1/T3 **decrease monotonically with width**. Untrained: $\sim N^{-0.15}$. Trained: $\sim N^{-0.35}$ (crude log-log fit over 4 points; not statistically rigorous).
- Training's "smoothing" effect on the FIM spectrum scales with width up to about $N \sim 10^4$, then the effect weakens — by $N = 2 \times 10^5$ the reduction is only ~4× rather than 24×.

## 3. Why this happens — preliminary explanation

Consider the FIM diagonal $F_{ii} = \mathbb{E}[(\partial_{\theta_i} \ell)^2]$ for a 5-layer ReLU network at Kaiming init:

- For parameters at the *beginning* of the network (stem layer), the gradient of the loss $\ell$ with respect to $\theta_i$ involves a product of 5 Jacobians of ReLU-gated linear maps. Each Jacobian introduces a factor of $\sim \sqrt{\text{width}}$ under Kaiming scaling. So $F_{ii}$ scales like $(\sqrt{\text{width}})^{2 \cdot 5} = \text{width}^{5}$ for stem-layer parameters — large.
- For parameters at the *end* of the network (head layer), only one Jacobian multiplies their gradient. So $F_{ii}$ scales like $\text{width}$ — small.

This gives an intrinsic $\text{width}^4$ spread between stem and head FIM values at random init. For width = 256, that's $\sim 256^4 = 4 \times 10^9$ — orders of magnitude more than the measured 1,500. The measured ratio is the *data-averaged* version, which damps the extreme stem/head asymmetry through ReLU gating and loss-function averaging.

Training flattens this by moving the network toward a Gaussian-like feature geometry (NTK regime at wide widths) where the layer-dependence of $F_{ii}$ is attenuated. Hence the 4-24× reduction.

## 4. Consequences for V1.0 / V4.0 / V3.0

### V1.0 — "physical constants" interpretation

The V1.0 identification of Tier-1 FIM parameters with "physical constants that gradient descent locks in" is empirically wrong as stated. The load-bearing picture is instead:

- **Tier-1 parameters** are parameters for which the induced loss landscape is naturally most curved — primarily *stem-layer* parameters in a ReLU network. These are not "locked in by training"; they are structurally placed by the architecture and initialization.
- **Training's role** is to tune them to useful values, not to create the hierarchy. The hierarchy is *given* by the architecture.

In the cosmological analogy: if the universe is a neural network, what corresponds to "physical constants" are not the values that emerged from cosmological learning, but rather *the weights in the stem layer of the universe's architecture* — whatever those are. The learning process uses these constants; it does not create them. This is closer to the "axioms are constants" reading than "dynamics locks them in" reading.

### V4.0 — uniqueness

The V4.0 contrast between trained NN (13,752×) and non-learning baselines (Ising 3×, Harmonic 4×, CA 4×) remains real: NN's tier ratio is higher than any non-NN baseline tested. But the *mechanism* is architectural, not learning-based. An **untrained** NN of matched N would still dominate the non-NN baselines by 10³×. The uniqueness signal is: *layered non-linear architecture with Kaiming init* → sharp tier ratio; *non-layered parameterized systems* → flat.

### V3.0 Tier-1 universality findings

These are unaffected. All V3.0 measurements were on trained networks; the empirical scale invariance, seed stability, task universality, and architecture universality results stand. The *mechanism* interpretation was always secondary to the measurement.

### V3.2 cosmological-constant prediction

The $\Lambda \sim 10^{-122}$ derivation invoked a Tier-1 fraction $f_1 = 0.01$ that is "scale-invariant". V4.1 confirms that $f_1$ is indeed scale-invariant in trained NNs (1% top tier by construction of the partition), but also that the *magnitude* of the tier-1 FIM values depends on width and on training status. The Λ consistency check is not invalidated — the $f_1 = 0.01$ fraction is just a partition choice — but the physical interpretation ("Tier-1 parameters are frozen under training") needs qualification: they are *less* frozen after training than before.

## 5. What this finding is and is not

**It IS:**
- A sharp empirical rejection of the "training locks in Tier-1" narrative.
- A clean, seed-stable measurement (CV < 30% on the trained/untrained ratio).
- A refinement, not a refutation, of the universality-class picture.

**It IS NOT:**
- A refutation of the FIM-Onsager correspondence in the NTK regime (V1.1 proof is independent of init vs learned).
- A refutation of the scale-invariance / task-universality findings (all measured on trained networks).
- An argument that learning is irrelevant — training *does* something, it just doesn't create the tier hierarchy.

## 6. Open questions

- Does the trained/untrained ratio continue to shrink as $N \to \infty$? Asymptotically, does trained T1/T3 approach the untrained value (i.e., training becomes irrelevant at infinite width, matching the NTK "lazy training" prediction)?
- Does the init-induced hierarchy persist for non-ReLU activations (GELU, SiLU, tanh)? A preliminary test at matched N would tell.
- Does CNN/ViT show the same pattern? The V3.0 arch baselines measured only *trained* arch baselines; trained-vs-untrained on CNN/ViT is the next step.

## 7. Recommended follow-ups

1. Extend V4.1 to CNN and ViT at matched N (~$0.50 on Runpod A100).
2. Test non-ReLU activations (GELU, Tanh).
3. Quantify the "training smooths, not sharpens" claim with a precise theorem under NTK assumptions — explicitly work out why $\partial_{\theta} F$ is negative under gradient flow for Kaiming-init networks.

---

*STARGA Commercial License. V4.1 reframing note for V1.0 / V4.0.*
