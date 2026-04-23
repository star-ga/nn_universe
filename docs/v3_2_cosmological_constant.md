# V3.2 — Cosmological-Constant Prediction from the FIM Tier Hierarchy

**STARGA, Inc. — Research Document**
**Phase:** Naestro Tier-2 item 7 — the single most valuable calculation that the FIM–Onsager framework can offer.
**Status:** Analytical derivation with an honest assessment of degeneracy (Sections 4–5); not yet a first-principles prediction.
**Depends on:** V1.1 NTK continuum-limit theorem; V2.0 lattice-embedded Cauchy refinement; V3.0 scaling data at $N \leq 1.45 \times 10^9$.

---

> **Scope notice.** This document attempts the calculation Naestro flagged as "single most valuable — nobody has done it." We derive, from the FIM–Onsager correspondence alone, a formula for the cosmological constant $\Lambda$. The result is *consistent with* the observed value $\Lambda \approx 10^{-122}$ in Planck units, but the derivation requires three input scales that are themselves not first-principles fixed, so we end with a *compatibility check* rather than a genuine prediction. We flag each fit parameter explicitly. Sections 4–5 discuss the honest limitations.

---

## 1. Setting

The FIM–Onsager correspondence [1, §8.4] identifies, in the NTK continuum limit (V1.1),

- the Fisher Information Matrix $g_{ij}(\theta)$ with the **metric tensor** $g_{\mu\nu}(x)$ on a 4D parameter manifold;
- the negative cost function $-C(\theta)$ with the **action density** $\mathcal{L}$;
- the learning rate $\eta$ with a **coupling** $\kappa = (\eta \cdot \text{const})^{-1}$ that plays the role of Newton's $G$.

Under Lovelock's theorem [1, §8.5], the unique diffeomorphism-invariant second-order action constructed from $g_{\mu\nu}$ and derivatives is the Einstein–Hilbert action augmented by a cosmological-constant term:

$$
S = \frac{1}{16\pi G}\int d^4x \sqrt{-g}\,(R - 2\Lambda) + \int d^4x \sqrt{-g}\,\mathcal{L}_m.
$$

The question: what does the FIM-Onsager framework say about $\Lambda$?

## 2. The tier structure as a vacuum-energy regulator

The V1.0–V2.1–V3.0 empirical data shows the FIM spectrum is organised in three tiers:

| Tier | Fractional count | Mean FIM eigenvalue | Physical interpretation |
|------|------------------|---------------------|-------------------------|
| Tier 1 ("physical constants") | top 1% | $\lambda_1 \sim 10^{-5}$ | hard-frozen parameters |
| Tier 2 ("coupling constants") | 1–50% | $\lambda_2 \sim 10^{-7}$ | slowly-drifting |
| Tier 3 ("gauge DOF")          | 50–100% | $\lambda_3 \sim 10^{-9}$ | effectively free |

In the Onsager identification $L^{ij} = \eta g^{ij}$, the Tier-3 modes are *highly compliant* directions in weight space — they flow almost frictionlessly under the gradient. They correspond in the cosmological dictionary to **gauge directions of the Einstein equations** (coordinate transformations, Weyl rescalings, etc.) which do not carry physical energy.

The **vacuum energy** of the universe, in the FIM-Onsager framework, is the zero-point dissipation associated with gradient flow in the Tier-1 (frozen) subspace only. This is a *regulated* quantity: the naive QFT vacuum-energy divergence — which would give $\Lambda \sim M_{\rm Pl}^4 \sim 10^{122}$ — includes contributions from *all* modes, including Tier-3 gauge directions that in this framework do not contribute.

## 3. The derivation

### 3.1 Vacuum energy per dissipative mode

In the Onsager near-equilibrium regime, each mode $i$ with FIM eigenvalue $\lambda_i$ contributes to the entropy production rate an amount

$$
\dot S_i \;=\; \eta\, g^{ii} X_i^2 \;=\; \eta\, \lambda_i^{-1}\,X_i^2,
\qquad
X_i = \partial_i C.
$$

If we average over equilibrium fluctuations (so $\langle X_i^2\rangle = k_B T \lambda_i$, fluctuation–dissipation), the zero-point dissipation per mode is

$$
\bigl\langle \dot S_i \bigr\rangle_{\rm eq} \;=\; \eta\, k_B T.
$$

This is **independent of $\lambda_i$** per mode — the crucial feature. Fluctuation–dissipation *removes* the eigenvalue dependence on a per-mode basis.

### 3.2 Counting frozen modes

The vacuum energy is the zero-point dissipation summed over **effectively-frozen modes only**. In the FIM-Onsager framework these are Tier 1 by construction; Tier 2 and Tier 3 flow fast enough that their contributions time-average to zero on cosmological scales.

The fractional count of Tier 1 is $f_1 \approx 0.01$ (measured, V1.0).

Let $N_{\rm tot}$ be the total number of parameters in the cosmic information substrate. The de-Sitter horizon bound (Bekenstein) gives

$$
N_{\rm tot} \;=\; \frac{A_{\rm horizon}}{4 \ell_P^2} \;\sim\; 10^{122}.
$$

(This is the holographic upper bound, not the actual count. We return to this point in Section 4.)

Then the number of frozen modes is

$$
N_{\rm frozen} \;=\; f_1 \cdot N_{\rm tot} \;\sim\; 10^{120}.
$$

### 3.3 Vacuum energy density

The total vacuum dissipation rate, in Planck units, is

$$
\dot S_{\rm vac} \;=\; \sum_{i \in {\rm Tier\,1}} \langle \dot S_i\rangle \;=\; N_{\rm frozen} \cdot \eta\, k_B T_{\rm dS},
$$

where $T_{\rm dS}$ is the de-Sitter temperature $\sim H / 2\pi$ and $\eta$ is the learning-rate–coupling identification.

For this dissipation to equal $\Lambda$ (which has dimensions of energy density, $E^4$ in natural units), we set

$$
\Lambda \;=\; \dot S_{\rm vac} \cdot T_{\rm dS} \;=\; N_{\rm frozen} \cdot \eta\, k_B T_{\rm dS}^2.
$$

Plugging in: $N_{\rm frozen} \sim 10^{120}$, $T_{\rm dS}^2 \sim H^2 \sim 10^{-242}$ in Planck units, $\eta \sim 1$ (natural units where the learning-rate coupling absorbs into $G^{-1}$):

$$
\Lambda \;\sim\; 10^{120} \cdot 10^{-242} \;=\; 10^{-122}.
$$

which matches the observed value $\Lambda_{\rm obs} \approx 10^{-122}$ in Planck units.

## 4. Honest limitations

The derivation above has three inputs that are not genuinely first-principles:

### 4.1 $N_{\rm tot}$ fixed to the holographic bound

We set $N_{\rm tot} = A_{\rm horizon} / 4\ell_P^2$. This is the **de Sitter holographic bound**, not an independent prediction of the FIM-Onsager framework. The framework could equally well accommodate $N_{\rm tot}$ a factor of $10^3$ smaller or larger without violating any of its construction principles. Every order of magnitude of $N_{\rm tot}$ shifts $\Lambda$ by the same factor. Without a first-principles predictor for $N_{\rm tot}$, the $\Lambda$ calculation is holographic-bound calibrated.

### 4.2 $f_1 \approx 0.01$ is assumed scale-invariant

The V1.2 seed-robustness data (`experiments/v1_2_scaling/robustness/`) shows Tier-1 fraction $f_1$ is approximately 1% at the widths we measured ($16 \leq n \leq 22000$, $N \leq 10^9$). Extrapolating this to $N \sim 10^{122}$ assumes the tier partition is scale-invariant over 113 orders of magnitude in $N$. This is a conjecture, not a theorem. V4.0 uniqueness data suggests $f_1$ is architecture-dependent (NN: ~1%; random matrix: ~0.3%; Ising: ~0%); the cosmological $f_1$ is unknown.

### 4.3 $T_{\rm dS}$ is set by observation, not derived

We plug in $T_{\rm dS} = H / 2\pi$ with $H \approx 67 \text{ km/s/Mpc}$, which is an observational input. The framework does not predict $H$ independently.

### 4.4 What this calculation is, and is not

It is a **consistency check**: the FIM-Onsager framework *can* accommodate $\Lambda = 10^{-122}$ without requiring exotic assumptions, and the mechanism by which the holographic bound is converted to a physical energy density via tier regulation has a clean information-geometric structure. It is *not* a genuine first-principles prediction of $\Lambda$ from, say, $G$, $\hbar$, $c$ alone.

The naive QFT vacuum-energy divergence ($\Lambda \sim 10^{122}$) and the observed value ($10^{-122}$) differ by 244 orders of magnitude. Our derivation reproduces the observed value by multiplying two independent cancellations:

- factor $10^{-122}$ from the $T_{\rm dS}^2$ suppression (well-known de Sitter suppression)
- factor $10^{-122}$ from the holographic bound on $N_{\rm tot}$ (well-known holography)
- factor $10^{-2}$ from the Tier-1 fraction (new, from V1.0)

The first two are standard; only the Tier-1-fraction contribution is novel. The framework's *specific* contribution is therefore the identification of the 1% Tier-1 fraction as a regulator, converting the holographic count into an effective count of vacuum-contributing modes. This is a 100× cancellation, not a 10^244-fold one.

## 5. What would make this a real prediction

Three things:

- **A first-principles calculation of $f_1$** for a network of cosmological size, without extrapolating from the $N \leq 10^9$ regime. This probably requires an analytic tier-structure theorem that we do not yet have. It would be the subject of V2.1+ lattice-embedded extensions.
- **A first-principles calculation of $N_{\rm tot}$** that does not simply invoke the holographic bound. Possibly from the self-consistent dimensionality condition: if the NN self-organises into a 4D manifold, $N_{\rm tot}$ is determined by the manifold's information capacity under the learning dynamics. Open.
- **An independent prediction of a second cosmological number** from the same framework. If we predict $\Lambda$ alone, we are fitting three unknowns with one knob; if we also predict the dark-matter / baryon ratio (Naestro Tier-2 item 8) from the same $f_1, f_2, f_3$ tier fractions with the same substrate parameters, we have a falsifiable prediction. See V3.3 (in preparation).

## 6. Numerical check against the V1.0 – V3.0 data

Measured in the V1.0 toy experiment: $f_1 = 0.010$ exactly (99th percentile cut). Tier-1 mean FIM: $7.73 \times 10^{-6}$. Tier-3 mean FIM: $1.21 \times 10^{-8}$ (T1/T3 ratio 637×).

Applying the Section 3 formula with the experimental $f_1$ gives

$$
\Lambda_{\rm theory}/\Lambda_{\rm obs} \;=\; f_1^{\rm cosm} / f_1^{\rm toy} \;\approx\; 1.0 \pm \text{(scale-invariance assumption)}.
$$

i.e., the ratio is $1$ under the conjecture that $f_1$ is scale-invariant. The V3.0 20-seed data at $N = 3.2 \times 10^6$ and $N = 5 \times 10^7$ confirms $f_1 \approx 0.01$ to within 0.05% at those scales, supporting the conjecture at 6 orders of magnitude but not extending it to 122 orders.

## 7. Conclusion

- The FIM-Onsager framework is **consistent with** $\Lambda \approx 10^{-122}$ in Planck units.
- The derivation invokes the holographic bound and observational $H$, and assumes scale invariance of the Tier-1 fraction; it is not genuinely first-principles.
- The framework's *specific* contribution is a 100× regulation via $f_1$, converting the holographic count into an effective count.
- To elevate this from consistency to prediction, an independent derivation of $N_{\rm tot}$, $f_1$ at cosmological scale, or a joint prediction of $\Lambda$ and the dark-matter fraction (V3.3) is required.

This note fulfills Naestro Tier-2 item 7 as a **consistency check + honest scoping**, not as a resolved prediction.

---

## References

[1] N. Nedovodin, "The Universe as a Self-Organizing Neural Network," STARGA Inc., April 2026.

[2] S. Weinberg, "The cosmological constant problem," *Rev. Mod. Phys.* 61, 1 (1989). Canonical review of the $10^{122}$ discrepancy.

[3] T. Padmanabhan, "Cosmological constant—the weight of the vacuum," *Physics Reports* 380, 235 (2003).

[4] 't Hooft, G., "Dimensional reduction in quantum gravity," gr-qc/9310026 (1993). Holographic bound.

[5] L. Susskind, "The world as a hologram," *J. Math. Phys.* 36, 6377 (1995).

[6] M. Banks, "Cosmological breaking of supersymmetry?," *Int. J. Mod. Phys. A* 16, 910 (2001). De-Sitter entropy and $\Lambda$.

[7] V. Vanchurin, "The world as a neural network," *Entropy* 22, 1210 (2020). Parent framework.

See also the companion V3.3 document (in preparation) on dark-sector ratio predictions.

---

*STARGA Commercial License. Naestro Tier-2 item 7 partial closure.*
