# V2.0 Lattice-Embedded Numerics

Companion numerics for `docs/v2_0_lattice_embedded.md`.

## Scripts

| Script | Purpose |
|--------|---------|
| `lattice_analytic.py` | **Primary V2.0 demonstration.** Untrained Gaussian-receptive-field lattice; compares discrete bilinear form `u^T G_a u` against the analytic continuum limit at five halving-refinement levels. Produces `lattice_analytic_results.json`. |
| `lattice_refinement.py` | Training-based refinement — same lattice model under SGD training. Produces `lattice_refinement_results.json`. Useful for probing the training-confounded Cauchy regime; less clean than `lattice_analytic.py`. |

## Key Result

From `lattice_analytic.py` with `d=2`, four refinement levels:

| Level | Spacing $a$ | u^T G u | |err| | rel_err |
|-------|-------------|---------|------|---------|
| 0 | 1.0000 | 2.5220e+00 | 4.51e-02 | 1.82e-02 |
| 1 | 0.5000 | 2.4887e+00 | 1.19e-02 | 4.80e-03 |
| 2 | 0.2500 | 2.4832e+00 | 6.36e-03 | 2.57e-03 |
| 3 | 0.1250 | 2.4797e+00 | 2.87e-03 | 1.16e-03 |

Observed convergence rate: $|err| \sim a^{1.28}$ — sub-quadratic, consistent with theorem's $O(a^2)$ prediction modulo finite-density reference integration.

## Historical Files

- `lattice_refinement_high_quality.json` — earlier run on remote CPU with the *old* lattice-scaled receptive field (divide-by-$a$ interpretation). Now superseded by the physical-units receptive field in the current `lattice_refinement.py`. Kept for historical reference; the "primary" clean result is `lattice_analytic_results.json`.
