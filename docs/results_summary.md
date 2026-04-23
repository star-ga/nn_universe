# nn_universe — Experimental Results Summary


_Auto-generated from JSON outputs in the repository. Do not edit manually._


### V1.0 Per-Layer SV (seed 42)

| Layer | Shape | Top 3 SV | SV Ratio | SV Std |
|-------|-------|----------|----------|--------|
| net.0.weight | [256, 64] | [1.8277, 1.7888, 1.7648] | 2.87x | 0.326 |
| net.2.weight | [256, 256] | [2.0758, 1.2525, 1.2472] | 847.13x | 0.3477 |
| net.4.weight | [256, 256] | [1.5362, 1.2949, 1.275] | 570.29x | 0.3458 |
| net.6.weight | [256, 256] | [1.3663, 1.3142, 1.2795] | 1921.51x | 0.3438 |
| net.8.weight | [256, 256] | [1.3126, 1.2971, 1.273] | 560.55x | 0.3444 |
| net.10.weight | [64, 256] | [1.1874, 1.1619, 1.1377] | 3.52x | 0.2524 |

### V1.2 Width Sweep

| Width | Params | SV Ratio | FIM T1/T3 |
|-------|--------|----------|-----------|
| 16 | 1,888 | 377.2x | 191.0x |
| 32 | 5,280 | 383.4x | 502.3x |
| 64 | 16,672 | 1047.6x | 615.7x |
| 128 | 57,888 | 2172.0x | 370.8x |
| 256 | 214,048 | 9490.6x | 149.6x |
| 512 | 821,280 | 9115.1x | 247.8x |
| 1024 | 3,215,392 | 32228.3x | 335.3x |
| 2048 | 12,722,208 | 554885.2x | 378.7x |
| 4096 | 50,610,208 | 77169.1x | 416.9x |
| 8192 | 201,883,680 | 59364.2x | 452.7x |
**SV power law:** $N^{0.566}$, $R^2 = 0.84$. **FIM power law:** $N^{0.02}$, $R^2 = 0.031$.

### V1.2 Seed Robustness (width=256)

| Seed | SV Ratio | FIM T1/T3 |
|------|----------|-----------|
| 0 | 30491.7x | 354.1x |
| 1 | 13003.0x | 416.8x |
| 2 | 66397.9x | 416.0x |
| 3 | 2956.7x | 435.2x |
| 4 | 6326.5x | 354.4x |
| 5 | 1739.4x | 447.8x |
**Mean SV:** 20152.5 ± 24990.2 (CV 124.0%). **Mean FIM T1/T3:** 404.1 ± 40.4 (CV 10.0%).

### V1.2 Depth Sweep (fixed width 256)

| Depth | Params | SV Ratio | FIM T1/T3 | Train time |
|-------|--------|----------|-----------|------------|
| 2 | 82,464 | 1347.7x | 63.6x | 41.2s |
| 3 | 148,256 | 36800.7x | 98.4x | 49.2s |
| 5 | 279,840 | 8793.6x | 765.7x | 64.3s |
| 8 | 477,216 | 4144.1x | 8.2e+06 | 86.6s |
| 12 | 740,384 | 2979.0x | 5.8e+14 | 117.7s |
| 20 | 1,266,720 | 53629.4x | 3.5e+14 | 178.5s |
> Note: FIM T1/T3 ratios for depth ≥ 8 are numerically unreliable because Tier-3 FIM values underflow float32 at these depths. The monotone upward trend is physical; the absolute magnitudes at depth ≥ 8 are lower bounds only.

### V2.0 Lattice Cauchy Convergence

| Level | Spacing $a$ | u^T G u | |err| | rel_err |
|-------|-------------|---------|------|---------|
| 0 | 1.0000 | 2.5220e+00 | 4.51e-02 | 1.82e-02 |
| 1 | 0.5000 | 2.4887e+00 | 1.19e-02 | 4.80e-03 |
| 2 | 0.2500 | 2.4832e+00 | 6.36e-03 | 2.57e-03 |
| 3 | 0.1250 | 2.4797e+00 | 2.87e-03 | 1.16e-03 |
**Observed convergence rate:** $|err| \sim a^{1.283}$ (theoretical: $O(a^2)$).

### V2.1 QEC Decoder Width Sweep

| Width | Params | SV Ratio | FIM T1/T3 | Final BCE loss |
|-------|--------|----------|-----------|----------------|
| 32 | 6,706 | 567.5x | 92.5x | 0.100515 |
| 64 | 21,554 | 1045.2x | 201.4x | 0.071288 |
| 128 | 75,826 | 6191.1x | 420.8x | 0.056881 |
| 256 | 282,674 | 36094.1x | 1762.3x | 0.04517 |
| 512 | 1,089,586 | 70289.7x | 46206.2x | 0.041288 |
| 1024 | 4,276,274 | 50250.6x | 704389.4x | 0.038971 |
**QEC SV power law:** $N^{0.807}$, $R^2 = 0.89$. **QEC FIM power law:** $N^{1.386}$, $R^2 = 0.93$.

### V3.1 Mock Power Analysis

| Threshold | α | FPR | Power |
|-----------|---|-----|-------|
| p05 | 5.00e-02 | 0.090 | 1.000 |
| p01 | 1.00e-02 | 0.030 | 1.000 |
| p001 | 1.00e-03 | 0.000 | 1.000 |
| p3sigma | 2.70e-03 | 0.010 | 1.000 |
| p5sigma | 5.73e-07 | 0.000 | 1.000 |