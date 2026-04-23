# V3.3 — Cluster-Core α-Drift Amplification

**STARGA, Inc. — Research Document**
**Phase:** Naestro Tier-2 item 6 — a sub-5-year falsifier independent of ELT-HIRES.
**Depends on:** `docs/v3_1_alpha_drift_prediction.md`; GR gravitational red/blue-shift formulae.

---

> **Scope notice.** This note computes the gravitational-potential amplification of the predicted α-drift inside a galaxy cluster potential well, and compares it to current atomic-clock sensitivity limits. If the amplified effect exceeds current bounds, the V3.1 prediction is *already falsified* by archival clock data. If it is within reach of the next generation of clock comparisons, V3.1 becomes a near-term falsifier rather than an ELT-HIRES 2028 target.

## 1. The amplification mechanism

In V3.1 the prediction is $\dot\alpha/\alpha = \kappa \rho_I(x)$ at a local point $x$, with $\rho_I$ the information density (Proxy B: baryon density).

Inside a gravitational potential well of depth $\Phi$, two independent amplification channels exist:

### 1.1 Gravitational redshift amplification

A drift measured at Earth of an emission line from inside the cluster potential experiences the redshift $1 + z_{\rm grav} = (1 - 2\Phi/c^2)^{-1/2}$. For a typical cluster, $\Phi / c^2 \sim 10^{-5}$ at core. This does **not** amplify the drift *rate* directly — it shifts the observed frequency, but $\dot\alpha/\alpha$ is a ratio and stays invariant to first order.

### 1.2 Local-density amplification (the actual V3.3 effect)

The V3.1 formula $\dot\alpha/\alpha = \kappa \rho_I$ is *local*. At a cluster core, $\rho_I$ is amplified by the local overdensity factor $\delta \equiv \rho_I / \bar\rho_I$, where $\bar\rho_I$ is the cosmic mean.

Observed cluster-core overdensities (from X-ray + lensing mass maps of e.g. Coma, Virgo, Perseus):

| Cluster region | $\delta = \rho/\bar\rho$ | Spatial scale |
|---|---|---|
| Cluster center ($R < 100\text{ kpc}$) | $10^{4}$ – $10^{5}$ | ~kpc |
| Cluster main body ($R < 1\text{ Mpc}$) | $10^{2}$ – $10^{3}$ | ~Mpc |
| Filament (supercluster) | $10$ – $30$ | ~10 Mpc |
| Void | $0.1$ – $0.3$ | ~10 Mpc |

So the predicted drift is amplified at cluster cores by up to $10^5$:

$$
\frac{\dot\alpha}{\alpha}\bigg|_{\rm core} \;\approx\; 10^5 \cdot \frac{\dot\alpha}{\alpha}\bigg|_{\rm mean} \;\approx\; 10^{5} \cdot 10^{-17}\,{\rm yr^{-1}} \;=\; 10^{-12}\,{\rm yr^{-1}}.
$$

## 2. Detectability: which telescopes / clock comparisons reach $10^{-12}$/yr?

### 2.1 Laboratory atomic clock comparisons (Earth)

Atomic clocks on Earth are **not** in a cluster core — they are in the Milky Way disk, at a moderate overdensity ($\delta \sim 10^2$ – $10^3$). Earth clock experiments therefore probe $\dot\alpha/\alpha \sim 10^{-14}$ to $10^{-15}$ yr$^{-1}$ (local amplification at our position).

Current best Earth-clock constraint: Lange et al. 2021 with Yb⁺ (E3/M1): $|\dot\alpha/\alpha| < 3.1 \times 10^{-18}$ yr$^{-1}$ at 2σ.

**Earth at $\delta \sim 10^3$ predicts $\dot\alpha/\alpha \sim 10^{-14}$ yr$^{-1}$, which is four orders of magnitude ABOVE the Lange 2021 bound.**

This is a serious constraint: either (a) the local overdensity proxy is wrong by $\sim 10^4$, (b) κ is smaller than assumed, or (c) the linear $\kappa \rho_I$ relation breaks down at high density (e.g., a saturation effect).

### 2.2 Quasar absorption spectroscopy

Quasar Mg II / Fe II absorption systems at $z \leq 3$ with sightlines through galaxy clusters provide the spatial resolution needed. UVES (VLT) and HIRES (Keck) measure $\alpha$ at $|\Delta\alpha/\alpha| \lesssim 10^{-6}$ per system; with $N \sim 200$ systems and a cluster-core subset, the effective per-bin sensitivity is $\lesssim 10^{-7}$ (stacked).

Over a lookback time of $\sim 5$ Gyr, a drift of $10^{-12}$ yr$^{-1}$ at the cluster cores accumulates to $\Delta\alpha/\alpha \sim 5 \times 10^{-3}$, which is **4 orders of magnitude above the Planck recombination bound** ($\sim 10^{-3}$).

**Planck 2018 at $z \sim 1100$** constrains $|\Delta\alpha/\alpha| < 1.8 \times 10^{-3}$. This corresponds to a time-averaged $\dot\alpha/\alpha < 1.3 \times 10^{-13}$ yr$^{-1}$ over the full 13.8 Gyr lookback, averaged over the last-scattering surface.

## 3. Honest assessment

The cluster-core amplification actually **creates a problem**, not a test:

- If $\delta = 10^5$ at cluster cores, the prediction is 5 orders of magnitude above the Earth-clock bound.
- If $\delta = 10^3$ at Earth, the prediction is already $10^{-14}$/yr, still 4 orders of magnitude above Earth-clock bounds.

**One of the following must be true:**

1. The V3.1 coupling κ is smaller than our fit by at least 4 orders of magnitude. This is possible — κ is only pinned to a cosmic-mean 10^{-17}/yr expectation, and the cluster-core amplification was not considered in the V3.1 derivation. A revised κ of $\sim 4 \times 10^{-63}$ instead of $4 \times 10^{-59}$ would make all observations compatible.
2. The linear $\kappa \rho_I$ relation saturates at high density; the physically-relevant quantity is $\rho_I$ normalised by some scale $\rho^\star$, with the linear regime valid only for $\rho_I < \rho^\star$. Earth and cluster-core both fall above $\rho^\star$. Open to theoretical modelling.
3. The mechanism is wrong. Earth-clock experiments already falsify the prediction at face value, and V3.1 must be substantially revised.

**All three options reduce the scientific interest of V3.1**: either κ is phenomenologically fit far below the observational scale, or the framework has an undetermined saturation parameter, or V3.1 is simply wrong.

## 4. Revised V3.1 in light of this analysis

### 4.1 Minimal revision

κ must be $\sim 10^{-63}$ to be compatible with Earth-clock bounds. With $\delta_{\rm Earth} \sim 10^3$:

$$
\frac{\dot\alpha}{\alpha}\bigg|_{\rm Earth} \;\sim\; 10^{-63} \cdot 10^{17} \cdot 10^{3} \;=\; 10^{-43}\,{\rm yr}^{-1}.
$$

This is **26 orders of magnitude below current best Earth-clock sensitivity**, hence unfalsifiable by clock data. At cluster cores ($\delta = 10^5$) it is $10^{-41}$/yr — still unfalsifiable by any present or near-future instrument.

### 4.2 Consequence

With the revised κ, **V3.1 is unfalsifiable in practice** even under cluster-core amplification. This is the *honest* reading of the cluster-amplification calculation: it closes the near-term-falsifier option that Naestro Tier-2 item 6 hoped for.

The 2028 ELT-HIRES target was based on *cosmic-mean* density sensitivity to κ $\sim 10^{-17}$/yr, not revised κ. Under the revised κ $\sim 10^{-63}$, even ELT-HIRES at $10^{-17}$/yr sensitivity will not detect the effect.

## 5. What this means for V3.1 as a falsifier

V3.1 does not currently have a realistic near-term falsifier pathway. The three open options:

- Derive κ from first principles (V2.0 lattice-embedded theorem + Naestro Tier-1 item 3 NTK-closure could give a principled κ). If that derivation gives κ $\sim 10^{-17}$ / mean-density, V3.1 *is* falsified by Earth-clock data today. If it gives κ $\leq 10^{-63}$, V3.1 is unfalsifiable.
- Replace the linear $\kappa \rho_I$ with a *differential* prediction: $\dot\alpha(\rho_I^{\rm cluster})/\dot\alpha(\rho_I^{\rm void})$, which avoids the magnitude-fitting. Under this form the test is a *ratio* (quasar cluster vs quasar void), which is cleaner. Proposal for V3.1.1.
- Move the observational target from present-day drift to *CMB-era* drift, which integrates over 13.8 Gyr and averages out the density dependence. The Planck bound of $10^{-3}$ at $z \sim 1100$ is the relevant number, and it is compatible with κ $\lesssim 10^{-21}$/yr at cosmic mean — narrower than quasar or clock bounds, but not conclusive.

## 6. Conclusion

- The cluster-core amplification does not open a new falsification channel.
- Existing Earth-clock and Planck CMB bounds already constrain κ to $\lesssim 10^{-63}$ under the linear $\kappa \rho_I$ model.
- Under a realistic κ, the predicted drift is unfalsifiable by any present or near-future instrument.
- V3.1 requires restructuring toward (a) a first-principles κ derivation, (b) a differential test across density environments, or (c) an entirely different observational channel.
- V3.3 as drafted *reduces* the predictive status of V3.1 rather than strengthening it; this is the honest finding.

Naestro Tier-2 item 6 closed with a negative result — cluster amplification does not create a sub-5-year falsifier.

---

## References

[1] N. Nedovodin, "The Universe as a Self-Organizing Neural Network," STARGA Inc., April 2026.

[2] T. Lange et al., "Improved Limits for Violations of Local Position Invariance from Atomic Clock Comparisons," *PRL* 126, 011102 (2021).

[3] D. J. E. Marsh, "Axion cosmology," *Phys. Reports* 643, 1 (2016). Cluster-core dark matter profiles.

[4] Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters," A&A 641, A6 (2020).

---

*STARGA Commercial License. Naestro Tier-2 item 6 closed.*
