# V3.2 — Cosmological-Constant Prediction from the FIM Tier Hierarchy (v2, patched)

**STARGA, Inc. — Research Document**
**Phase:** Naestro Tier-2 item 7.
**Status:** Consistency check; the framework's contribution is a **~10× Tier-1 fraction correction** on top of the standard Cohen–Kaplan–Nelson / de Sitter-holography bound. The resulting prediction is off from observation by ~1 order of magnitude — much better than the naive 122-order QFT gap, but not a tuned match.
**Depends on:** V1.1 NTK continuum-limit theorem; V4.1 init-vs-learning reinterpretation.

---

## 0. Audit trail (v1 → v2)

The original V3.2 document (v1, committed in `14845e0`) contained a dimensional/arithmetic error: it attributed the full $10^{-122}$ suppression to the FIM tier-fraction argument alone. In fact the $10^{-122}$ suppression is the standard **CKN / de Sitter-entropy bound** on vacuum energy (Cohen, Kaplan & Nelson 1999; Banks 2001); our framework's *additional* contribution is a factor of $f_1 \sim 10^{-2}$ on top, not the full 122 orders. This patch (v2) fixes the mis-attribution and corrects the formula. The conclusion is weaker than v1 claimed: the framework gets Λ within one order of magnitude of the observed value, not exactly.

---

## 1. The cosmological-constant problem

**Observed:** $\rho_\Lambda \approx 6 \times 10^{-10}\ \mathrm{J/m^3} \approx 1.3 \times 10^{-123}$ in Planck units ($\hbar = c = G = 1$, $\rho_{\rm Planck} = M_{\rm Pl}^4 \approx 4.6 \times 10^{113}\ \mathrm{J/m^3}$).

**Naive QFT prediction:** summing vacuum fluctuations up to $\Lambda_{\rm UV} = M_{\rm Pl}$ gives $\rho_\Lambda^{\rm QFT} \sim M_{\rm Pl}^4 = 1$ in Planck units. Gap: **122 orders of magnitude**. This is the canonical "cosmological-constant problem" (Weinberg 1989).

## 2. The standard holographic/CKN bound (not our framework)

Cohen, Kaplan & Nelson (1999) observed that a QFT cannot be self-consistent up to $\Lambda_{\rm UV}$ in a region of size $L$ if the resulting energy would form a black hole, i.e., $\Lambda_{\rm UV}^4 L^3 \gtrsim L / G$, giving

$$
\Lambda_{\rm UV} \lesssim (M_{\rm Pl} / L)^{1/2}.
$$

Applied to the de Sitter horizon $L = H^{-1}$:

$$
\rho_\Lambda^{\rm CKN} \sim \Lambda_{\rm UV}^4 \sim H^2 M_{\rm Pl}^2.
$$

**In Planck units:** $H \approx 1.5 \times 10^{-33}\ \mathrm{eV} / M_{\rm Pl} = 1.2 \times 10^{-61}$, so $H^2 \approx 1.5 \times 10^{-122}$. This gives

$$
\rho_\Lambda^{\rm CKN} \sim 10^{-122}\ \text{(Planck units)}.
$$

**This already saturates the 122-order gap, without any FIM framework.** The CKN bound is standard GR+QFT physics and is the real reason the gap is 122 orders, not 10 orders or 500 orders. Any neural-network interpretation of cosmology inherits this bound as a starting point, not as an output.

## 3. The FIM-Onsager framework's additional contribution

The contribution of the FIM tier hierarchy is to *sub-select* which modes actually contribute to the CKN vacuum-energy count. In the FIM-Onsager picture (Nedovodin 2026; V1.1 NTK theorem), modes fall into three tiers by FIM diagonal entry:

- **Tier 1** ($f_1 \approx 0.01$ of total modes): high FIM diagonal entry ("stiff", potential-dominated, time-invariant under Onsager flux).
- **Tier 2** ($f_2 \approx 0.49$): intermediate FIM, drift slowly.
- **Tier 3** ($f_3 \approx 0.50$): small FIM, free-flowing, gauge-like.

**Empirically measured in V1.0**: Tier-1 has FIM mean $\sim 7.7 \times 10^{-6}$, Tier-3 mean $\sim 1.2 \times 10^{-8}$ — ratio 637×. The *V4.1 reinterpretation* clarifies that this hierarchy is architecture-induced (present at random init), not learning-locked; but the tier *fractions* $f_1, f_2, f_3$ are stable quantities under the partition definition.

**Physical claim.** Only Tier-1 modes contribute static vacuum energy to the cosmological constant — Tier-2 and Tier-3 modes are dynamical / gauge-like and their contributions time-average to zero on cosmological scales. Under this claim the effective CKN-admissible mode count for Λ is reduced by a factor $f_1$:

$$
\rho_\Lambda^{\rm FIM} \sim f_1 \cdot \rho_\Lambda^{\rm CKN} \sim f_1 \cdot H^2 M_{\rm Pl}^2.
$$

**In Planck units:** $f_1 \cdot H^2 \sim 0.01 \times 1.5 \times 10^{-122} \approx 1.5 \times 10^{-124}$.

## 4. Comparison with observation

$$
\boxed{\rho_\Lambda^{\rm FIM} \approx 1.5 \times 10^{-124}, \qquad \rho_\Lambda^{\rm obs} \approx 1.3 \times 10^{-123}.}
$$

The framework predicts Λ about an order of magnitude **smaller** than observed. Gap remaining: factor ~9.

## 5. Honest assessment

### 5.1 What's right

- The 122-order suppression is handled correctly (from CKN/holography, not our framework).
- The FIM-tier interpretation of "only Tier-1 contributes to vacuum energy" is physically motivated and gives an additional ~$f_1 = 10^{-2}$ suppression.
- Result (~$10^{-124}$) is within one order of magnitude of observation (~$10^{-123}$) — better than generic anthropic arguments (which allow any value above the observed one).

### 5.2 What's not right

- **Not a precise match.** Off by ~9×. The framework over-suppresses by one order of magnitude.
- **$f_1 = 10^{-2}$ is a convention**, not a first-principles prediction. The V1.0 tier partition defines Tier-1 as the top 1% by FIM diagonal entry. If we had defined it as top 5%, we'd get $f_1 = 0.05$ and exact agreement. The choice of 1% is thus a one-knob fit.
- **$N_{\rm tot}$ is the holographic bound**, not framework-derived.

So the framework contributes **one tunable knob** ($f_1$) in exchange for reproducing observation to within 1 order of magnitude. This is a modest improvement over the standard CKN result (which already gets within factor ~10 of observation depending on choice of cutoff volume) but is **not** a first-principles prediction.

### 5.3 What would make this a prediction

- Derive $f_1$ uniquely from the FIM-Onsager framework — i.e., show that exactly 1% of the modes have FIM diagonal values above a threshold *determined by* the Onsager dynamics, with no partition-definition freedom.
- Predict a second independent observable (e.g., $\Omega_{\rm DM}/\Omega_{\rm baryon}$; see V3.4) with the **same** $f_1$ value, no new knob.

V3.4 (dark-sector) uses the same $f_1$ but introduces a separate knob $\xi_{\rm bar}$. V3.2 + V3.4 jointly have two observables and two knobs — not yet a prediction.

## 6. The v1 → v2 change in plain language

The v1 document said the FIM framework **generates** the $10^{-122}$ suppression through a tier-fraction calculation. This was wrong — that suppression belongs to standard de Sitter holography (CKN), not to our framework. The v2 document claims only that the framework adds a factor of $f_1 \sim 10^{-2}$ on top of the CKN result, yielding $10^{-124}$ (close to but ~10× below observation).

Mechanism | Suppression from naive QFT ($M_{\rm Pl}^4$) | Cumulative | Source
---|---|---|---
CKN/holographic | $10^{-122}$ | $10^{-122}$ | Cohen-Kaplan-Nelson 1999; standard GR+QFT
FIM Tier-1 fraction $f_1$ | $10^{-2}$ | $10^{-124}$ | V1.0 partition definition (this framework)
Observation | — | $10^{-123}$ | Planck 2018

## 7. Conclusion (revised)

- $\rho_\Lambda \sim 10^{-124}$ in Planck units is the framework's prediction.
- Observed $\rho_\Lambda \sim 10^{-123}$.
- Gap: 1 order of magnitude (framework is too small).
- $f_1 = 10^{-2}$ is a partition-definition convention, not a first-principles prediction; this is the one tunable knob.
- Naestro Tier-2 item 7 is closed with a **"consistent within 1 order of magnitude, one free knob" status** — honest but not elevating the framework to predictive theory.

---

## References

[1] A. Cohen, D. Kaplan, A. Nelson, "Effective Field Theory, Black Holes, and the Cosmological Constant," *PRL* 82, 4971 (1999).

[2] T. Banks, "Cosmological breaking of supersymmetry?" *Int. J. Mod. Phys. A* 16, 910 (2001).

[3] S. Weinberg, "The cosmological constant problem," *Rev. Mod. Phys.* 61, 1 (1989).

[4] Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters," A&A 641, A6 (2020).

[5] G. 't Hooft, "Dimensional reduction in quantum gravity," gr-qc/9310026 (1993).

[6] L. Susskind, "The world as a hologram," *J. Math. Phys.* 36, 6377 (1995).

[7] N. Nedovodin, "The Universe as a Self-Organizing Neural Network," STARGA Inc., April 2026.

Companion: `docs/v3_4_dark_sector.md`, `docs/v4_1_init_vs_learning.md`.

---

*STARGA Commercial License. V3.2 v2 patch dated 2026-04-24; corrects dimensional mis-attribution in v1.*
