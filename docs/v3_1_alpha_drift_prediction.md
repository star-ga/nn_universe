# V3.1 Prediction Document: Fine-Structure Constant Drift Correlated with Galactic Information Density

**Nikolai Nedovodin, STARGA Inc.**
**April 2026**

*Phase V3.1 of the "Universe as a Self-Organizing Neural Network" research program.*
*Parent paper: Nedovodin 2026 (V1.0), §9.1.*

---

## 1. Prediction Recap & Sharpening

### 1.1 Statement from §9.1

The parent paper (V1.0, §9.1) states:

> **Prediction.** The fine-structure constant $\alpha$ undergoes slow drift at a rate proportional to the local information density of the cosmic web. Specifically:
>
> $$\frac{\dot{\alpha}}{\alpha} = \kappa \cdot \rho_I(x)$$
>
> where $\rho_I(x)$ is the information density (measured in bits per unit volume) at location $x$ and $\kappa$ is a coupling constant.
>
> **Quantitative bound.** Neural-network cosmology predicts $|\dot{\alpha}/\alpha| \sim 10^{-18}$ to $10^{-17}$ per year in regions of average cosmic information density, with enhanced drift (up to $10^{-16}$ per year) in regions of high information density (galaxy cluster cores) and suppressed drift (below $10^{-19}$) in cosmic voids.

The present document sharpens this prediction to the level required for a referee to specify: (a) the exact dataset, (b) the exact statistical test, and (c) the exact numerical threshold below which the theory is falsified. The order-of-magnitude character of the estimate is preserved; no false precision is introduced.

### 1.2 Mechanism

In the FIM-Onsager framework (V1.0, §8), $\alpha$ is identified with a weight $\theta_i$ of the universal neural network. Under Natural Gradient Descent (NGD) dynamics, the rate of change of any parameter is governed by the local curvature of the loss landscape:

$$\dot{\theta}^i = -\eta \, g^{ij} \partial_j C(\theta)$$

where $g_{ij}$ is the Fisher Information Matrix and $C(\theta)$ is the cost function (identified with negative thermodynamic entropy, $C = -S$). Parameters with large diagonal FIM eigenvalues $F_i$ change slowly; parameters with small $F_i$ change quickly. This is precisely the Elastic Weight Consolidation (EWC) mechanism (V1.0, §8.6): $\alpha$ has a large but finite $F_i$, so it drifts slowly but not at zero rate.

Regions of high information density correspond, in the cosmological dictionary (V1.0, §8.3), to regions of high FIM eigenvalues. The local gradient pressure $\partial_j C$ is proportional to the local information density because the cost function density is the negative entropy density, and entropy density is maximized — at fixed energy — by the equilibrium distribution, which is itself a function of $\rho_I$. Taken together:

$$\frac{\dot{\alpha}}{\alpha} \approx -\eta \cdot \frac{F_\alpha(\rho_I)}{F_\alpha^{(0)}} \cdot \bar{\epsilon}$$

where $F_\alpha(\rho_I)$ is the local FIM eigenvalue of the $\alpha$-parameter at information density $\rho_I$, $F_\alpha^{(0)}$ is its background value, and $\bar{\epsilon}$ is a dimensionless cost-gradient amplitude of order unity at cosmic mean density. This reduces to the linear form $\dot{\alpha}/\alpha = \kappa \cdot \rho_I$ in the near-equilibrium, small-perturbation limit.

### 1.3 Derivation of the Coupling Constant $\kappa$

The coupling constant $\kappa$ connects the fractional drift rate to $\rho_I$ in units of bits per cubic Megaparsec (Mpc$^3$). Its derivation from first principles within the FIM-Onsager framework proceeds as follows.

**Step 1: FIM eigenvalue of $\alpha$ per bit of information density.**

The toy experiment (V1.0, §11; README Tier-1 table) yields a mean Tier-1 FIM eigenvalue of:

$$\langle F \rangle_{\mathrm{Tier\,1}} = 7.73 \times 10^{-6} \quad \text{(dimensionless, per parameter)}$$

for a network of $N_{\mathrm{toy}} \approx 296{,}000$ parameters. This is measured per parameter at the $\sim 10^5$ gradient-sample scale.

**Step 2: Scaling to cosmological parameter count.**

The FIM eigenvalue per parameter scales with network size. The scaling experiment (V1.0, §11.7; README) yields:

$$F_{\mathrm{Tier\,1}} \propto N^{-\gamma}, \quad \gamma \approx 0.5 \text{ to } 1.0 \text{ (empirical)}$$

For a cosmological network of $N_{\mathrm{cosm}} \sim 10^{80}$ parameters (Bekenstein bound estimate for the observable universe), the naive per-parameter FIM eigenvalue at the same relative tier position is:

$$F_\alpha^{(0)} \sim 7.73 \times 10^{-6} \times \left(\frac{10^5}{10^{80}}\right)^{0.5} \sim 7.73 \times 10^{-6} \times 10^{-37.5} \sim 10^{-43}$$

This is a rough order-of-magnitude estimate. The absolute value is not physically meaningful in isolation; what matters is the ratio $F_\alpha / \langle F \rangle_{\mathrm{mean}}$, which characterizes how much above the background eigenvalue $\alpha$'s FIM weight sits.

**Step 3: Coupling constant estimate.**

Define $\kappa$ as:

$$\kappa = \frac{F_\alpha^{(\mathrm{per\,bit})}}{\langle F \rangle_{\mathrm{mean}}} \cdot \frac{\eta}{V_0}$$

where $F_\alpha^{(\mathrm{per\,bit})}$ is the Fisher information contribution of $\alpha$ per bit of local information density, $\langle F \rangle_{\mathrm{mean}}$ is the mean FIM eigenvalue over all parameters, $\eta$ is the effective learning rate (identified in §8.3 with the inverse temperature $T^{-1}$ and the cosmological coupling), and $V_0$ is a reference volume (1 Mpc$^3$).

Using the Tier-1/mean ratio from the toy experiment ($\sim 637 \times$ above the geometric mean of Tier-1 and Tier-3), and requiring the drift rate at cosmic mean density to reproduce $|\dot{\alpha}/\alpha| \sim 10^{-17}$ yr$^{-1}$, one obtains:

$$\kappa \sim \frac{10^{-17}\,\mathrm{yr}^{-1}}{\rho_I^{(\mathrm{mean})}}$$

The mean information density of the observable universe, using the Bekenstein bound applied to the horizon volume $V_H \sim 4 \times 10^{80}$ Mpc$^3$ and $S_{\mathrm{max}} \sim 10^{122}$ bits, gives:

$$\rho_I^{(\mathrm{mean})} \sim \frac{10^{122}\,\mathrm{bits}}{4 \times 10^{80}\,\mathrm{Mpc}^3} \sim 2.5 \times 10^{41}\,\mathrm{bits\,Mpc}^{-3}$$

Therefore:

$$\kappa \sim \frac{10^{-17}\,\mathrm{yr}^{-1}}{2.5 \times 10^{41}\,\mathrm{bits\,Mpc}^{-3}} \sim 4 \times 10^{-59}\,\mathrm{yr}^{-1}\,\mathrm{bit}^{-1}\,\mathrm{Mpc}^{3}$$

**Important caveat on circularity (post-audit).** The κ derivation above is, strictly speaking, a **phenomenological fit**, not a first-principles prediction. The chain is:

1. The V1.0 toy experiment gives a Tier-1 FIM value (concrete number).
2. Naively extrapolated via power-law to cosmological parameter count ($10^{80}$), this would give some $F_\alpha^{(\mathrm{per\,bit})}$.
3. But the extrapolation is over 74 orders of magnitude in $N$, so the predictive content is weak.
4. Independently, laboratory constraints set $|\dot\alpha/\alpha| \lesssim 10^{-17}$ yr$^{-1}$.
5. The κ value quoted above is what makes the theory compatible with (4). It is not what V1.0's FIM measurements *force*.

In other words, κ is **fit to the observational upper bound, not derived from first principles**. A genuine first-principles κ would require either (a) actual $10^{12}$-parameter FIM measurements from V3.0, or (b) a closed-form expression for how FIM eigenvalues extrapolate with parameter count in the NTK regime (partial progress in docs/v1_1_ntk_continuum_limit.md).

Until V3.0 lands, the sign and order-of-magnitude of κ should be treated as **phenomenology**, and the predictive content of V3.1 is the *shape* of the correlation (with log ρ_I, controlling for z) rather than the absolute magnitude of κ. This is a limitation but not a fatal one: the null hypothesis $r = 0$ is still a clean falsification criterion regardless of the exact κ value.

The quoted drift rates ($10^{-18}$ to $10^{-16}$ yr$^{-1}$) represent a reasonable expectation range, acknowledging the above caveat.

---

## 2. Information Density Models

$\rho_I(x)$ must be operationally defined before the prediction can be confronted with data. Three proxy definitions are considered, in order of increasing theoretical grounding and decreasing observational accessibility.

### 2.1 Proxy A: Bekenstein-Bound Estimate

The Bekenstein bound gives the maximum information content of a region with boundary area $A(x)$:

$$S_{\max}(x) = \frac{A(x)}{4 \ell_P^2}$$

where $\ell_P = \sqrt{\hbar G / c^3} \approx 1.616 \times 10^{-35}$ m is the Planck length. The volumetric information density is then:

$$\rho_I^{(A)}(x) = \frac{S_{\max}(x)}{V(x)} = \frac{A(x)}{4 \ell_P^2 \, V(x)}$$

For a spherical region of radius $R$, this gives $\rho_I^{(A)} \propto R^{-1}$, meaning denser (smaller) regions have higher information density per unit volume. For a galaxy cluster core at $R \sim 100$ kpc versus a void at $R \sim 10$ Mpc, the ratio is $\sim 10^2$, consistent with the predicted $10^2$ enhancement in $\dot{\alpha}/\alpha$.

**Observational accessibility:** Moderate. Requires knowledge of the local gravitational potential to infer the effective bounding surface. Can be estimated from weak-lensing convergence maps.

**Limitation:** The Bekenstein bound is an upper bound, not the actual entropy. Regions far from saturating the bound (most of the universe) will have $\rho_I^{(A)}$ that systematically overestimates the true $\rho_I$.

**Post-audit downgrade.** Ordinary astrophysical structures (galaxy clusters, filaments, voids) are many orders of magnitude below saturating the Bekenstein holographic bound; their entropy is set by astrophysical rather than gravitational-horizon physics. Therefore Proxy A should be read as a **scale-setting order-of-magnitude estimator** for the *maximum possible* information content of a region, *not* as a proxy for the actual information content that would couple in the FIM-Onsager framework. The $R^{-1}$ scaling is generic to the Bekenstein form but does not robustly reflect the astrophysical density hierarchy that $\rho_I$ is meant to capture. Accordingly, Proxy A is demoted to a scale-setting role and Proxy B (baryon density) is the primary observationally-testable handle for V3.1.

### 2.2 Proxy B: Baryon Density

The simplest observationally accessible proxy is the local baryon mass density $\rho_b(x)$, available from galaxy surveys and hydrodynamical simulations:

$$\rho_I^{(B)}(x) \propto \rho_b(x)$$

The proportionality constant absorbs the specific entropy per baryon, which varies by environment but spans only $\sim 1$ order of magnitude across cosmic structures (compared to the $\sim 3$-order-of-magnitude range in $\rho_b$ itself between cluster cores and voids).

**Observational accessibility:** High. $\rho_b$ is estimated from optical and X-ray luminosity, weak lensing, and Sunyaev-Zel'dovich effect observations. SDSS DR18 provides photometric redshifts for $\sim 4 \times 10^8$ objects; IllustrisTNG and EAGLE simulations provide calibrated $\rho_b$ fields on Mpc scales.

**Limitation:** The proportionality constant is environment-dependent. Cluster cores and field galaxies have different baryon-to-entropy ratios. This introduces a systematic error that must be marginalized over.

**Recommendation:** Proxy B is the most testable with current datasets. The initial analysis described in §4–§5 uses Proxy B.

### 2.3 Proxy C: Gravitational Entropy

A theoretically better-motivated proxy uses the gravitational entropy density derived from the local gravitational field strength and an effective Hawking temperature:

$$\rho_I^{(C)}(x) \propto \frac{|\mathbf{g}(x)|}{T_{\mathrm{Hawking}}(x)}$$

where $|\mathbf{g}(x)|$ is the magnitude of the local gravitational acceleration and $T_{\mathrm{Hawking}}(x) = \hbar |\mathbf{g}(x)| / (2\pi c k_B)$ is the Unruh temperature associated with that acceleration. This gives:

$$\rho_I^{(C)}(x) \propto \frac{|\mathbf{g}(x)|^2 \cdot 2\pi c k_B}{\hbar} \propto |\mathbf{g}(x)|^2$$

The local gravitational field is related to the matter overdensity $\delta(x)$ through the Poisson equation, making this proxy computable from N-body simulations.

**Observational accessibility:** Low. Requires high-resolution mass maps from strong lensing or precision weak lensing, not yet available over the sightline samples needed in §4.

**Limitation:** The Hawking/Unruh temperature for astrophysical accelerations is $T \sim 10^{-20}$ K, far below any observable threshold. The proportionality is theoretical and the normalization is poorly constrained.

**Choice for V3.1 analysis:** Proxy B ($\rho_b$) is used as the primary observable, with Proxy A as a cross-check where lensing mass maps are available, and Proxy C deferred to V3.2.

---

## 3. Observational Handles

### 3.1 Quasar Absorption Spectra (Primary)

The primary observational handle on spatial variation of $\alpha$ is the many-multiplet (MM) method applied to quasar absorption spectra. The fractional change in $\alpha$ shifts the wavelengths of metal transitions by amounts that depend on relativistic corrections quantified by the sensitivity coefficients $q$ (Dzuba & Flambaum 1999):

$$\frac{\Delta\lambda}{\lambda_0} = -2\frac{\Delta\alpha}{\alpha} \cdot q$$

where $\Delta\alpha = \alpha(z) - \alpha_0$ and $q$ values differ in sign and magnitude across transitions (e.g., $q_{\mathrm{Mg\,II}} \approx 0.0$ vs. $q_{\mathrm{Fe\,II}} \approx +1300$ cm$^{-1}$). Comparing transitions with positive and negative $q$ allows differential measurement of $\Delta\alpha/\alpha$ independent of velocity systematics.

**Available datasets:**

| Dataset | Spectrograph | $N_\mathrm{systems}$ | $z$ range | Precision (per system) |
|---|---|---|---|---|
| Webb et al. 2011 | Keck HIRES + VLT UVES | 293 | $0.2$–$3.7$ | $\sim 5 \times 10^{-6}$ |
| King et al. 2012 | VLT UVES | 154 | $0.2$–$4.2$ | $\sim 3$–$8 \times 10^{-6}$ |
| Murphy et al. 2022 | VLT ESPRESSO | $\sim 20$ pilot | $0.6$–$1.7$ | $\sim 1$–$3 \times 10^{-6}$ |

Current constraints on spatial variation from these datasets: Webb+2011 reported a dipole signal at $\Delta\alpha/\alpha \sim 10^{-5}$ level with $4.2\sigma$ significance; King+2012 found $|\Delta\alpha/\alpha| = (-0.178 \pm 0.084) \times 10^{-5}$ at $z < 1.8$, consistent with a small negative mean offset. The dipole interpretation remains controversial; Berengut+2012 placed independent constraints using laboratory sensitivity calculations.

**Critical note:** None of these analyses correlated $\Delta\alpha/\alpha$ measurements with the environmental information density (cluster membership, local baryon density) of the absorbing systems. That cross-correlation is the specific test demanded by this theory.

### 3.2 Atomic Clock Comparisons (Temporal Drift at $z = 0$)

Laboratory atomic clock comparisons provide the tightest constraint on temporal drift of $\alpha$ at the present epoch and at Earth's location (average cosmic density, not a cluster core or void):

| Measurement | Clock pair | Constraint | Reference |
|---|---|---|---|
| Rosenband et al. 2008 | Al$^+$ / Hg$^+$ | $|\dot{\alpha}/\alpha| < 1.6 \times 10^{-17}$ yr$^{-1}$ (1$\sigma$) | Rosenband+2008 |
| Huntemann et al. 2014 | Yb$^+$ / Cs | $|\dot{\alpha}/\alpha| < 5.2 \times 10^{-17}$ yr$^{-1}$ | Huntemann+2014 |
| Godun et al. 2014 | Sr / Cs | $|\dot{\alpha}/\alpha| < 5.8 \times 10^{-17}$ yr$^{-1}$ | Godun+2014 |
| Lange et al. 2021 | Yb$^+$ (E3/M1) | $|\dot{\alpha}/\alpha| < 3.1 \times 10^{-18}$ yr$^{-1}$ (2$\sigma$) | Lange+2021 |

The current state-of-the-art constraint from Lange et al. 2021 is $|\dot{\alpha}/\alpha| < 3.1 \times 10^{-18}$ yr$^{-1}$ at $2\sigma$.

**Consistency with the prediction.** Earth is located in the Milky Way disk, a moderate-density environment. The theory predicts $|\dot{\alpha}/\alpha| \sim 10^{-18}$ to $10^{-17}$ yr$^{-1}$ at this density. The Lange et al. 2021 upper bound of $3.1 \times 10^{-18}$ yr$^{-1}$ is marginally consistent with but does not confirm the prediction. The theory is not falsified by current clock data, but it is constrained to the lower end of its predicted range at mean cosmic density. A measured value of $|\dot{\alpha}/\alpha| > 10^{-17}$ yr$^{-1}$ at Earth's density would constitute a mild tension; $> 10^{-16}$ yr$^{-1}$ would be strong evidence against the framework. A confirmed null result $|\dot{\alpha}/\alpha| < 10^{-19}$ yr$^{-1}$ over the next decade would falsify the theory if Earth's environment is confirmed to have $\rho_I > \rho_I^{(\mathrm{void})}$.

### 3.3 CMB-Era Constraints

The physics of hydrogen recombination at $z \approx 1100$ depends sensitively on $\alpha$, through the Rydberg energy $E_1 \propto \alpha^2 m_e c^2$ and the fine-structure splitting. Changes in $\alpha$ at recombination shift the recombination redshift, the sound horizon, and the CMB angular power spectrum.

| Constraint source | $|\Delta\alpha/\alpha|$ at $z \sim 1100$ | Reference |
|---|---|---|
| WMAP 9-year | $< 4 \times 10^{-3}$ | Hinshaw+2013 |
| Planck 2015 | $|\Delta\alpha/\alpha| = (3.6 \pm 3.7) \times 10^{-3}$ | Planck+2015 (varying constants) |
| Planck 2018 + BAO | $|\Delta\alpha/\alpha| < 1.8 \times 10^{-3}$ (95% CL) | Hart & Chluba 2020 |

These constraints apply to the mean $\alpha$ at recombination, averaged over the entire last-scattering surface. They do not probe spatial variation or correlation with local density. A fractional variation of $\Delta\alpha/\alpha \sim 10^{-3}$ over the 13.8 Gyr lookback time corresponds to $\dot{\alpha}/\alpha \sim 7 \times 10^{-17}$ yr$^{-1}$, which is consistent with the theory's upper range (cluster cores) applied to an environment of moderate density at $z \sim 1100$.

**Note:** The theory does not predict the time-averaged mean drift to be at the upper end of its range; the CMB constraint is therefore not directly constraining, but it provides an important check that the cumulative drift over cosmic time does not violate the recombination bound.

### 3.4 White Dwarf Zeeman Effects

White dwarf atmosphere spectra and Zeeman splitting patterns encode the value of $\alpha$ at the white dwarf surface — a high-gravity, high-density environment. Preliminary analyses (Berengut+2013; Bainbridge+2017) suggest $|\Delta\alpha/\alpha| \lesssim 10^{-4}$ at the WD surface relative to laboratory values. The gravitational redshift must be separated from a putative $\alpha$ variation, requiring independent mass-radius measurements.

**Relevance for the proxy.** White dwarf surfaces have $\rho_b \sim 10^6$ g cm$^{-3}$, roughly $10^{12}$ times the cosmic mean baryon density. The theory would predict $|\dot{\alpha}/\alpha|_{\mathrm{WD}} \sim 10^{-5}$ yr$^{-1}$ at the WD surface — but WDs are not cosmologically expanding environments; the theory's prediction applies to regions that are part of the large-scale neural network dynamics, not to compact objects in hydrostatic equilibrium. This distinction must be made explicit in any analysis. Compact objects are regions where the near-equilibrium assumption of the Onsager framework may break down (cf. §9 Limitations).

---

## 4. Specific Statistical Test

### 4.1 Falsification Protocol

The theory is falsified by the following procedure. This section states the test precisely enough for independent replication.

**Data assembly.** Compile a sample of $N \geq 100$ quasar Mg II / Fe II absorption systems satisfying:
- Redshift range: $0.4 \leq z_{\mathrm{abs}} \leq 2.5$ (sufficient lever arm; avoids very low-$z$ sightlines where $\alpha$ measurements are noisier).
- Signal-to-noise ratio per pixel: $S/N \geq 30$ at the relevant transitions.
- $\alpha$ measurement available from the MM method with per-system uncertainty $\sigma(\Delta\alpha/\alpha) \leq 10^{-5}$.
- Each system must have an environmental density proxy $\rho_I^{(B)}$ (baryon density, Proxy B of §2.2) assigned from an overlapping galaxy survey (SDSS DR18, DESI Year-1, or equivalent) within a comoving radius of 5 Mpc centered on the absorber redshift.

The primary sample is the union of publicly available reduced spectra from the VLT UVES Large Programme (King+2012) and archival Keck HIRES (Murphy+2017 compilation), supplemented by ESPRESSO pilot measurements (Murphy+2022).

**Environmental proxy assignment.** For each absorber at $(z_{\mathrm{abs}}, \mathrm{RA}, \mathrm{Dec})$:
- Identify all spectroscopic galaxies from SDSS DR18 within projected separation $r_\perp \leq 10 h^{-1}$ Mpc and redshift separation $|\Delta z| \leq 0.01$ (velocity window $\sim 3000$ km s$^{-1}$).
- Compute the galaxy overdensity $\delta_g + 1 = n_{\mathrm{gal}} / \bar{n}_{\mathrm{gal}}$ within a 5 Mpc (comoving) sphere, using a flux-limited volume-corrected galaxy density field.
- Convert to baryon density using $\rho_b = \bar{\rho}_b \cdot (\delta_b + 1)$, with $\delta_b \approx \delta_g / b$ where $b \sim 1.5$ is a fiducial linear bias factor.
- Classify systems into three environment bins: **void** ($\delta_g < -0.5$), **field** ($-0.5 \leq \delta_g < 2$), **cluster/filament** ($\delta_g \geq 2$).

**Test statistic.** Compute the partial Pearson correlation coefficient:

$$r(\Delta\alpha/\alpha,\; \log \rho_I \mid z,\; S/N)$$

controlling for redshift $z$ and signal-to-noise ratio $S/N$. The partial correlation removes the known redshift evolution of both $\alpha$ measurements and the density field, and the known S/N-dependent systematic bias in MM measurements.

**Predicted effect size.** The theory predicts:

$$r_{\mathrm{theory}} \gtrsim +0.20$$

at minimum (order-of-magnitude estimate, derived as follows). The dynamic range of $\rho_I^{(B)}$ from voids to cluster cores spans $\sim 3$ orders of magnitude in $\delta_g$, corresponding to a factor $\sim 10^3$ in $\rho_I$. The theory predicts a factor $\sim 10^3$ variation in $\dot{\alpha}/\alpha$, from $< 10^{-19}$ to $10^{-16}$ yr$^{-1}$. The fractional variation in $\dot{\alpha}/\alpha$ relative to the mean is therefore $\sim 10^2$, substantially larger than the measurement noise in a well-selected sample. For a log-normal distribution of $\rho_I$ with dispersion $\sigma_{\log \rho} \approx 0.8$ dex (consistent with the SDSS density field) and a linear $\Delta\alpha/\alpha \propto \rho_I$ relation, a standard signal-to-noise argument gives $r_{\mathrm{theory}} \approx 0.25$ to $0.45$ depending on the scatter model. The floor of $r_{\mathrm{theory}} \geq 0.20$ is adopted as the conservative falsification threshold.

**Decision rule:**

- The theory predicts: $r > 0$ (positive correlation) with $|r| \geq 0.20$ at 95% CL.
- **The theory is falsified if:**
  - $r \leq 0$ at 95% CL (negative correlation or zero), **or**
  - $r > 0$ but $|r| < 0.20$ and the sample has power $\geq 80\%$ to detect $|r| = 0.20$ (see §5).
- **The theory is supported (not confirmed) if:**
  - $r \geq 0.20$ at $\geq 2\sigma$.
- **The theory is strongly supported if:**
  - $r \geq 0.20$ at $\geq 5\sigma$ and the slope $d(\Delta\alpha/\alpha)/d(\log \rho_I)$ is consistent with the $\kappa$ estimate of §1.3 within one order of magnitude.

**Note on sign.** The theory predicts $\dot{\alpha}/\alpha > 0$ in high-density regions (faster adaptation of $\alpha$ toward its "learned" value). However, the sign of $\dot{\alpha}$ relative to $\alpha_0$ (whether $\alpha$ is above or below its equilibrium value) is not yet determined from first principles at the current stage of the theory. The V3.1 analysis therefore uses a two-tailed test on the correlation and a one-tailed test only after the sign question is resolved by V3.0 experiments. See §9.3.

---

## 5. Null Hypothesis and Power Analysis

### 5.1 Null Hypothesis

$H_0$: The fractional change in $\alpha$, as measured from quasar absorption spectra, has no dependence on the local baryon density of the absorbing environment, after controlling for redshift and $S/N$. Formally:

$$H_0: \rho(\Delta\alpha/\alpha,\; \log \rho_I \mid z,\; S/N) = 0$$

Under $H_0$, the partial correlation coefficient $r$ is distributed approximately as:

$$t = \frac{r\sqrt{N-2-k}}{\sqrt{1-r^2}} \sim t_{N-2-k}$$

where $k = 2$ is the number of control variables (redshift and $S/N$), and $N$ is the sample size.

### 5.2 Power Analysis

To detect a true effect size $r = r_{\mathrm{min}}$ at significance level $\alpha_s$ (two-tailed) with power $1 - \beta$, the required sample size is:

$$N \approx \left(\frac{z_{\alpha_s/2} + z_\beta}{\frac{1}{2}\ln\frac{1+r_{\mathrm{min}}}{1-r_{\mathrm{min}}}}\right)^2 + 3 + k$$

where $z_\alpha$ is the standard normal quantile and the denominator is Fisher's $z$-transformation of $r_{\mathrm{min}}$.

For $r_{\mathrm{min}} = 0.20$, $\alpha_s = 5.73 \times 10^{-7}$ (5$\sigma$, two-tailed), $1 - \beta = 0.80$, $k = 2$:

$$z_{\alpha_s/2} + z_\beta = 5.0 + 0.842 = 5.842$$

$$\frac{1}{2}\ln\frac{1+0.20}{1-0.20} = \frac{1}{2}\ln(1.5) = 0.2027$$

$$N \approx \left(\frac{5.842}{0.2027}\right)^2 + 5 \approx (28.8)^2 + 5 \approx 834$$

For $r_{\mathrm{min}} = 0.20$ at 2$\sigma$ (95% CL) with power 80%:

$$z_{\alpha_s/2} + z_\beta = 1.960 + 0.842 = 2.802$$

$$N \approx \left(\frac{2.802}{0.2027}\right)^2 + 5 \approx (13.82)^2 + 5 \approx 196$$

**Summary table:**

| Detection threshold | Power | $r_{\mathrm{min}}$ | Required $N$ |
|---|---|---|---|
| 2$\sigma$ (95% CL) | 80% | 0.20 | $\approx 196$ |
| 3$\sigma$ | 80% | 0.20 | $\approx 370$ |
| 5$\sigma$ | 80% | 0.20 | $\approx 834$ |
| 5$\sigma$ | 80% | 0.30 | $\approx 375$ |
| 5$\sigma$ | 80% | 0.40 | $\approx 215$ |

The existing archival datasets (Webb+2011: 293 systems; King+2012: 154 systems) have sufficient combined size ($N \approx 447$ with overlap removed) to achieve 3$\sigma$ sensitivity to $r = 0.20$, and 5$\sigma$ sensitivity to $r \geq 0.30$. The environmental density cross-match reduces the effective sample to the subset with adequate galaxy survey coverage; this is estimated to be $\sim 200$–$300$ systems, sufficient for a 3$\sigma$ test.

A definitive 5$\sigma$ test at $r_{\mathrm{min}} = 0.20$ requires $N \approx 834$ systems with density cross-matches. This is achievable with DESI Year-3 or 4MOST data combined with the ELT-HIRES spectrograph (anticipated $> 1000$ high-quality systems).

---

## 6. Current Data Pass

### 6.1 Available Literature Results

The following sources provide the raw material for the environmental correlation test:

- **Webb et al. 2011** (Astrophys. J. Lett. 718, L166): 293 absorption systems from Keck HIRES and VLT UVES, redshifts $0.22 < z < 3.7$. Reports a spatial dipole in $\Delta\alpha/\alpha$ at the $10^{-5}$ level. Systems are listed with coordinates, redshifts, and $\Delta\alpha/\alpha$ measurements. No environmental density proxy was computed.

- **King et al. 2012** (Mon. Not. R. Astron. Soc. 422, 3370): 154 VLT UVES systems, independent sample. Reports a mean $\langle \Delta\alpha/\alpha \rangle = (-1.78 \pm 0.84) \times 10^{-6}$ with no dipole at $> 2\sigma$. No environmental density proxy was computed.

- **Berengut et al. 2012** (Phys. Rev. Lett. 109, 070802): cross-check using laboratory atomic structure calculations (Dzuba-Flambaum sensitivity coefficients). Provides independent confirmation of the MM method systematics. Reports $|\Delta\alpha/\alpha| < 10^{-5}$ from laboratory comparisons.

- **Murphy et al. 2022** (Astron. Astrophys. 658, A123): First ESPRESSO results ($\sim 20$ systems), demonstrating precision $\sigma(\Delta\alpha/\alpha) \sim 10^{-6}$ per system — an order of magnitude improvement over previous spectrographs.

### 6.2 Status of the Correlation Test

No published analysis has computed the partial correlation between $\Delta\alpha/\alpha$ and any measure of local environmental density (baryon density, cluster membership, galaxy overdensity) for the systems in any of the above samples.

The SDSS DR18 spectroscopic catalog ($\sim 4 \times 10^6$ galaxies with spectroscopic redshifts) covers the sky area and redshift range of the Webb+2011 and King+2012 systems. The cross-matching pipeline required to assign $\rho_b$ proxies to absorbers does not require new observations; it requires:
1. Downloading the Webb+2011 / King+2012 system coordinates and redshifts.
2. Querying SDSS DR18 CasJobs for galaxies within the search volumes defined in §4.1.
3. Computing the galaxy overdensity field and assigning $\rho_b$ estimates.
4. Running the partial correlation analysis.

**The test is currently unperformed. The data products exist. The analysis is the bottleneck.**

The V3.1 experiment pipeline described in §7 is designed to unblock this analysis. A pre-registration of the analysis protocol (on OSF or a similar platform) prior to running the cross-match is strongly recommended to avoid post-hoc specification of the test statistic.

---

## 7. Synthetic Mock Experiment

The following pipeline is to be implemented at `/home/n/nn_universe/experiments/v3_1_alpha/`. The purpose of the mock experiment is to (a) validate the analysis pipeline before applying it to real data, (b) estimate false-positive and false-negative rates under the theory and under $H_0$, and (c) produce ROC curves that characterize the discriminating power of the test as a function of sample size and effect size.

### 7.1 Mock Pipeline Description

**Step 1: Synthetic sightline generation.**
Synthesize $N = 1000$ mock quasar sightlines drawn from the redshift distribution of the combined Webb+2011 / King+2012 sample (empirically approximated as a sum of two Gaussians peaking near $z \sim 1.0$ and $z \sim 2.0$). Assign sky coordinates uniformly over the accessible SDSS footprint ($\sim 8000$ deg$^2$ north).

**Step 2: Environmental density assignment.**
For each sightline, draw a baryon overdensity $1 + \delta_b$ from a log-normal distribution with parameters calibrated to the IllustrisTNG-300 simulation at the relevant redshift:
- Mean: $\langle \log(1+\delta_b) \rangle = 0$
- Dispersion: $\sigma_{\log(1+\delta_b)} \approx 0.8$ (consistent with large-scale structure simulations at $z \sim 1$)
- Optionally: replace the log-normal with an actual SDSS DR18 density field slice to use realistic void-filament-cluster morphology.

**Step 3: Theory-predicted $\alpha$ injection.**
For each sightline, compute the theory-predicted $\Delta\alpha/\alpha$ as:

$$\Delta\alpha/\alpha = \kappa \cdot \rho_I^{(B)} \cdot \Delta t(z)$$

where $\Delta t(z)$ is the lookback time to redshift $z$, $\rho_I^{(B)} = \bar{\rho}_b \cdot (1 + \delta_b)$, and $\kappa$ is drawn from a log-uniform prior spanning $10^{-1}$ to $10^{1}$ times the fiducial estimate of §1.3. Add Gaussian measurement noise $\mathcal{N}(0, \sigma^2)$ with $\sigma \sim 5 \times 10^{-6}$ (representative MM precision from VLT UVES).

**Step 4: Partial correlation fit.**
Apply the same analysis pipeline as §4.1 to the mock dataset. Compute $r(\Delta\alpha/\alpha, \log \rho_I \mid z, S/N)$, the $t$-statistic, and the p-value. Repeat for 1000 Monte Carlo realizations.

**Step 5: False-positive rate under $H_0$.**
Generate 1000 mock datasets with the same structure but with $\alpha$ values drawn from a zero-mean Gaussian independent of $\rho_I$. Compute the fraction of realizations that produce $r > 0.20$ at $95\%$ CL. This gives the empirical false-positive rate; it should be $\leq 5\%$ by construction if the pipeline is correctly implemented.

**Step 6: False-negative rate under the theory.**
For the theory-prediction mock datasets, compute the fraction of realizations that fail to detect $r \geq 0.20$ at $95\%$ CL. This is the false-negative (miss) rate, which feeds directly into the power analysis of §5.

**Step 7: ROC curves.**
Sweep $r_{\mathrm{threshold}}$ from 0 to 1 and plot the true-positive rate (sensitivity) against the false-positive rate (1 - specificity) for the theory vs. $H_0$ models. Compute the AUC (area under the curve). A well-designed test should have AUC $> 0.85$ for $N \geq 300$ systems at the predicted $r \approx 0.25$.

**Output files** (to be produced by the pipeline):
- `mock_sightlines.parquet`: synthetic dataset
- `correlation_results.csv`: $r$, $t$, p-value per Monte Carlo draw
- `roc_curves.pdf`: ROC plots across sample sizes $N \in \{100, 200, 500, 1000\}$
- `power_table.csv`: power as a function of $N$ and $r_{\mathrm{min}}$

---

## 8. What V3.1 Would Deliver

A complete V3.1 analysis would produce two distinct deliverables:

### 8.1 Archival Analysis Paper

A published analysis of archival UVES and HIRES $\alpha$ measurements cross-correlated with SDSS DR18 environmental density proxies. This paper would:
- Apply the protocol of §4.1 to the combined Webb+2011 / King+2012 sample with SDSS DR18 density cross-matches.
- Report the partial correlation coefficient $r(\Delta\alpha/\alpha, \log \rho_I \mid z, S/N)$ with full error budget.
- Compare the measured $r$ against the theory predictions and $H_0$.
- Constitute the first test of environmental dependence of $\alpha$ variation in the literature.

If the result is $r < 0$ at 95% CL, the theory is falsified and the result is independently publishable as a negative result. If $r > 0$ with $|r| \geq 0.20$, the result is a positive detection of the predicted signal and constitutes strong evidence for the framework.

### 8.2 Pre-Registered Test Protocol for Future Surveys

A pre-registered analysis plan for next-generation surveys, including:
- **DESI** (Dark Energy Spectroscopic Instrument): $\sim 35$ million galaxy spectra with high completeness to $z \sim 1.5$, providing unprecedented density maps for cross-correlation with $\alpha$ sightlines.
- **4MOST** (4-metre Multi-Object Spectroscopic Telescope): planned high-redshift QSO program will yield $> 10^5$ quasar spectra, including $\sim 5000$ with S/N sufficient for MM $\alpha$ measurements.
- **ELT-HIRES** (Extremely Large Telescope): ANDES spectrograph (formerly HIRES) will achieve $\sigma(\Delta\alpha/\alpha) \sim 10^{-7}$ per system, a factor $\sim 50$ improvement. This would enable detection or exclusion of the predicted signal at 5$\sigma$ with $N \sim 100$ well-chosen systems.

The pre-registration would specify: sample selection criteria, environmental proxy definition, control variables, test statistic, effect size threshold, and decision rule — all as stated in §4.1 — prior to any data analysis. This is essential given the history of post-hoc flexibility in the $\alpha$-variation literature (see §9.1).

---

## 9. Limitations and Caveats

### 9.1 Systematic Disagreements Between Measurement Groups

The Webb+2011 and King+2012 analyses of largely overlapping datasets reached different conclusions: Webb+2011 reported a $4.2\sigma$ dipole at $\Delta\alpha/\alpha \sim 10^{-5}$; King+2012 found no significant dipole from the VLT-only sample. The source of this discrepancy — identified as a combination of long-range wavelength calibration errors (Whitmore & Murphy 2015), isotopic abundance pattern assumptions, and sample selection — means that neither dataset can be taken at face value without re-reduction against a common calibration standard.

The V3.1 archival analysis must either (a) use only the Murphy+2022 ESPRESSO data, which has the most reliable wavelength calibration, accepting a smaller $N$, or (b) apply a consistent re-reduction to the UVES/HIRES data using the Laser Frequency Comb calibration method (Murphy+2020), which removes the dominant systematic. Mixing reduced spectra from different groups without recalibrating to a common system would invalidate the correlation test.

### 9.2 Proxy Choice Degeneracy

The theory's prediction is that $\dot{\alpha}/\alpha \propto \rho_I$, where $\rho_I$ is the information density. The proxy $\rho_I^{(B)} \propto \rho_b$ introduces a degeneracy: a positive correlation $r(\Delta\alpha/\alpha, \rho_b)$ could in principle arise from any physical mechanism that correlates $\alpha$ with the local matter density, not specifically from the FIM-Onsager mechanism. Distinguishing the FIM-Onsager origin from, e.g., a direct coupling of $\alpha$ to the local scalar field density in a dilaton-like model requires the environmental proxy to carry additional discriminating information — specifically, the nonlinear dependence predicted by the Bekenstein proxy (Proxy A) versus the linear baryon-density proxy (Proxy B). If both Proxy A and Proxy B give similar correlation coefficients, no proxy discrimination is possible. If Proxy A gives significantly higher $r$ than Proxy B, this favors the holographic/area-based information density definition.

This proxy degeneracy is a fundamental limitation of the V3.1 analysis. It does not invalidate the falsification test (a null result falsifies the theory regardless of proxy choice) but it limits the confirmatory power of a positive result. Proxy discrimination requires higher-quality mass maps (weak lensing) over the quasar sightlines, which is a V3.2 objective.

### 9.3 Sign Ambiguity of $\dot{\alpha}$

The FIM-Onsager framework predicts that $\alpha$ drifts at a rate $\dot{\alpha}/\alpha = \kappa \cdot \rho_I$, where the sign of $\kappa$ depends on whether $\alpha$ is currently above or below its equilibrium value — a quantity not determined by the theory at its current stage. If $\alpha$ is currently below equilibrium, $\dot{\alpha} > 0$ (increasing $\alpha$) in all environments, with higher rates in denser regions. If $\alpha$ is above equilibrium, $\dot{\alpha} < 0$ in all environments.

This means the correlation test of §4.1 is sign-agnostic on the drift itself, but sign-definite on the correlation: in either case, $|\dot{\alpha}/\alpha|$ is predicted to be larger in denser environments. The observable quantity is $|\Delta\alpha/\alpha(z)|$ relative to $\alpha_0$, which accumulates as $|\alpha(z) - \alpha_0| \propto \kappa \rho_I \Delta t$. The partial correlation test uses the signed $\Delta\alpha/\alpha$; in the presence of sign ambiguity, the correlation coefficient could be near zero even if the magnitude is positively correlated. The V3.1 analysis should therefore also test the correlation of $|\Delta\alpha/\alpha|$ with $\rho_I$, using a one-tailed test on the magnitude, to remain sensitive to the sign-ambiguous case.

Until V3.0 experiments resolve the sign of $\kappa$ (which requires large-scale NN experiments to determine whether the cosmological $\alpha$-parameter sits above or below its loss-landscape minimum), V3.1 must employ both the signed two-tailed test (§4.1) and the magnitude one-tailed test, and report both.

---

## 10. Handoff to V3.0 Larger-Scale Runs

The order-of-magnitude uncertainty on $\kappa$ is the primary theoretical bottleneck for the V3.1 prediction. The uncertainty derives from two sources:

**Source 1: FIM scaling exponent.** The toy experiment covers $N = 1.9 \times 10^3$ to $2.0 \times 10^8$ parameters (5 orders of magnitude). The scaling exponent $\gamma$ in $F_{\mathrm{Tier1}} \propto N^{-\gamma}$ is empirically estimated at $\gamma \approx 0.5$ to $1.0$ from the available data, but the exponent is not converged. Extending the scaling run to $N = 10^{12}$ (the V3.0 target, requiring 64+ GPU cluster) would add 4 further decades and reduce the uncertainty on $\gamma$ from $\Delta\gamma \approx 0.5$ to $\Delta\gamma \lesssim 0.1$.

**Source 2: Tier-1 identification of $\alpha$.** In the toy experiment, Tier-1 parameters are identified by FIM eigenvalue rank. The cosmological identification of $\alpha$ as a specific Tier-1 parameter requires an argument about which weight corresponds to the electromagnetic coupling in the continuum limit. This argument requires the V2.0 lattice-embedding analysis (discretization of the FIM into a smooth metric with identifiable physical field content) to be completed first. V2.0 is listed as a research-phase item in the roadmap.

**Implication for V3.1 timeline.** V3.1 can proceed in parallel with V3.0 and V2.0 because:
- The falsification test of §4.1 does not require a sharp $\kappa$ prediction; it requires only $r_{\mathrm{min}} = 0.20$, which follows from the order-of-magnitude estimate.
- A null result (§6.2: analysis is the bottleneck) is useful regardless of the theoretical precision.
- A positive result with $r \geq 0.20$ would motivate the resource investment in V3.0 cluster runs.

The V3.0 output — a $\gamma$ exponent precise to $\Delta\gamma \leq 0.1$, and Tier-1 eigenvalue statistics at $10^{12}$ parameters — will feed back into V3.1 to sharpen $\kappa$ and thereby sharpen the predicted slope $d(\Delta\alpha/\alpha)/d(\log \rho_I)$ from its current order-of-magnitude range to a factor-of-3 prediction. This tighter prediction would be testable with ELT-ANDES at $\sigma(\Delta\alpha/\alpha) \sim 10^{-7}$ precision.

---

## 11. References

- Berengut, J. C., Flambaum, V. V., King, J. A., Curran, S. J., & Webb, J. K. 2011, Phys. Rev. D, 83, 123506.
- Berengut, J. C., Flambaum, V. V., & Ong, A. 2012, Phys. Rev. Lett., 109, 070802.
- Dzuba, V. A., & Flambaum, V. V. 1999, Phys. Rev. A, 61, 034502. [Laboratory sensitivity coefficients for many-multiplet method.]
- Godun, R. M., et al. 2014, Phys. Rev. Lett., 113, 210801.
- Hart, L., & Chluba, J. 2020, Mon. Not. R. Astron. Soc., 493, 3255. [CMB constraints on varying $\alpha$.]
- Hinshaw, G., et al. (WMAP Collaboration) 2013, Astrophys. J. Suppl., 208, 19.
- Huntemann, N., et al. 2014, Phys. Rev. Lett., 113, 210802.
- King, J. A., Webb, J. K., Murphy, M. T., Flambaum, V. V., Carswell, R. F., Bainbridge, M. B., Wilczynska, M. R., & Koch, F. E. 2012, Mon. Not. R. Astron. Soc., 422, 3370.
- Lange, R., et al. 2021, Phys. Rev. Lett., 126, 011102. [Best current clock constraint: $|\dot{\alpha}/\alpha| < 3.1 \times 10^{-18}$ yr$^{-1}$.]
- Murphy, M. T., et al. 2022, Astron. Astrophys., 658, A123. [First ESPRESSO $\alpha$ results.]
- Nedovodin, N. (STARGA Inc.) 2026, "The Universe as a Self-Organizing Neural Network," Research Synthesis & Original Contribution, April 2026. [Parent paper, V1.0.]
- Planck Collaboration (Planck 2015 Results XIV) 2016, Astron. Astrophys., 594, A14. [Fundamental constants from CMB.]
- Rosenband, T., et al. 2008, Science, 319, 1808. [Al$^+$/Hg$^+$ clock comparison; $|\dot{\alpha}/\alpha| < 1.6 \times 10^{-17}$ yr$^{-1}$.]
- Uzan, J.-P. 2011, Living Rev. Relativity, 14, 2. [Comprehensive review of varying constants: observational constraints, theoretical frameworks, laboratory measurements.]
- Webb, J. K., King, J. A., Murphy, M. T., Flambaum, V. V., Carswell, R. F., & Bainbridge, M. B. 2011, Phys. Rev. Lett., 107, 191101. [293-system Keck HIRES + VLT UVES $\alpha$ dipole claim.]
- Whitmore, J. B., & Murphy, M. T. 2015, Mon. Not. R. Astron. Soc., 447, 446. [Long-range wavelength calibration errors in UVES/HIRES; dominant systematic in Webb+2011.]
- SDSS DR18: Almeida, A., et al. (SDSS Collaboration) 2023, Astrophys. J. Suppl., 267, 44.
- DESI Collaboration 2023, arXiv:2306.06308 [astro-ph.CO]. [DESI instrument and survey design.]
- 4MOST Consortium, de Jong, R. S., et al. 2019, The Messenger, 175, 3.

---

*Document status: Draft, V3.1 Phase. Internal STARGA Inc. research note.*
*Classification: Pre-publication theoretical prediction document.*
*Next action: Pre-register §4.1 protocol on OSF; run cross-match against SDSS DR18 for King+2012 sample.*
