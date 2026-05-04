# FIM Tier Hierarchy in Deep Layered Sequential Computation

Computational and statistical validation of the Fisher Information Matrix
diagonal tier hierarchy as a panel-bounded empirical regularity of deep
layered sequential computation.

## Headline result

**The FIM three-tier diagonal hierarchy is a mechanism-backed universality
signature of deep layered sequential computation.** Across 12 parameterised
substrate classes — trained / untrained neural networks, random boolean
circuits, four shallow parameterised learners, U(1) and SU(2) lattice gauge
fields, three dynamical-system controls, and a random-matrix ensemble — the
tier ratio $T_1/T_3$ separates into two groups with **complete rank
separation** (one-sided Mann–Whitney $U$: $p = 1.7 \times 10^{-17}$,
rank-biserial $r = 1.000$, $n_{\text{deep}} = 46$, $n_{\text{rest}} = 50$).
The split is quantitatively predicted by **Hanin & Nica 2020** (Comm. Math.
Phys. 376, 287–322): log-normal $F_{ii}$ with depth-linear variance →
$\log(T_1/T_3) \propto \sqrt{L}$. We empirically confirm this scaling across
**six independent substrate classes** (untrained MLP $R^2 = 0.98$, trained
MLP $R^2 = 0.94$, random boolean circuits $R^2 = 0.98$, transformers
$R^2 = 0.97$, balanced binary tensor networks $R^2 = 0.99$, ResNet
residual stacks $R^2 = 0.999$ slope 16.74 over 4–32 blocks). The mechanism
is also confirmed across four activations (ReLU $R^2 = 0.965$, GELU/Swish
$R^2 = 0.990$, tanh $R^2 = 0.969$). **Honest scope narrowing**: GPT-Tiny
attention architectures sit in the deep-sequential band by magnitude but
the $\sqrt{L}$ scaling itself does not hold for either tied- or
untied-embedding configuration; time-unrolled RNN/LSTM also do not follow
the scaling. The mechanism's $\sqrt{L}$ universality is therefore scoped to
non-attention layered-stack architectures; the dichotomy magnitude claim is
universal across all tested substrates.

**Real-data verification.** ResNet-18 trained 10 epochs on CIFAR-10 (81.4 %
test accuracy, 11.2 M params) gives **$T_1/T_3 = 778$**, Gini = 0.84,
top-1 % FIM mass = 47.8 %. Same measurement protocol as the synthetic-task
panel. Production-scale measurements at ResNet-50 V1+V2 on ImageNet
(76.13 / 80.86 % top-1), ViT-L/16 (304 M params), GPT-2-medium (5-seed
$\log_{10}(T_1/T_3) = 4.83 \pm 0.003$), GPT-2-large (774 M), and
Pythia-1.4 B all sit firmly in the deep-sequential band.

The universality class is **deep layered sequential composition** — not
neural networks, not learning, not optimisation. See
[`docs/fim_tier_hierarchy_neurips2026.md`](docs/fim_tier_hierarchy_neurips2026.md)
for the full draft and [`docs/v6_0_mechanism_hanin_nica.md`](docs/v6_0_mechanism_hanin_nica.md)
for the mechanism derivation.

## Reproduction (one command)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/reproduce_main_results.sh
```

Wall-clock on a single CPU is ~12 minutes; on a CUDA GPU ~3 minutes. The
script runs the four load-bearing experiments and writes JSONs alongside
the committed reference outputs for byte-level comparison.

| Experiment | Script | Reference output |
|---|---|---|
| V5.0 dichotomy stats | `experiments/v5_0_dichotomy_stats/dichotomy_stats.py` | `dichotomy_stats_results.json` |
| V5.1 threshold sensitivity | `experiments/v5_0_dichotomy_stats/threshold_sensitivity.py` | `v5_1_threshold_sensitivity_results.json` |
| V5.2 multi-seed bootstrap | `experiments/v5_0_dichotomy_stats/mw_bootstrap.py` | `v5_2_mw_bootstrap_results.json` |
| V6.0 mechanism (depth sweep) | `experiments/v6_0_depth_mechanism/depth_sweep.py` | `v6_0_depth_sweep.json` |
| V4.5 partition-invariant | `experiments/v4_3_statistics/partition_invariant_dichotomy.py` | `v4_3_partition_invariant_dichotomy.json` |
| V6.0c pooling-error bound | `experiments/v6_0_mechanism/pooling_error_bound.py` | `v6_0c_pooling_error_bound_results.json` |

For a faster first pass that skips the V6.0 depth sweep (~7 min savings):

```bash
bash scripts/reproduce_main_results.sh --quick
```

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **V1.0** | 296k-param toy experiment + 6-scale sweep, 3 falsifiable predictions validated | **Done** |
| **V1.1** | NTK continuum limit — rigorous theorem for $L$-layer ReLU FC networks | **Done** ([gap closure](docs/v1_1_ntk_gap_closure.md)) |
| **V1.2** | Extended scaling: 10 widths, seed-robustness, depth sweep | **Done** ([scripts](experiments/v1_2_scaling/)) |
| **V2.0** | Lattice-embedded subclass: Cauchy-refinement theorem + numerical demo | **Done** ([numerics](experiments/v2_0_lattice/)) |
| **V2.1** | QEC decoder spectral analysis — universality test across 2 tasks | **Done** ([experiment](experiments/v2_1_qec/)) |
| **V3.0** | Cluster-scale runs + symbolic-regression task + arch baselines | **Done** ([task-3](experiments/v3_0_task3_symbolic/), [arch baselines](experiments/v3_0_arch_baselines/)) |
| **V4.0** | Uniqueness test — FIM tier hierarchy vs 5 non-NN parameterised systems | **Done** ([experiment](experiments/v4_0_uniqueness/)) |
| **V4.1** | Trained-vs-untrained: hierarchy is init-induced, training dissipates 4–24× | **Done** ([experiment](experiments/v4_0_uniqueness/run_trained_vs_untrained.py)) |
| **V4.2** | FIM diagonal vs full spectrum (Lanczos on small MLP) | **Done** ([experiment](experiments/v4_2_fim_spectrum_validation/)) |
| **V4.3** | Tier-partition sensitivity + bootstrap 95 % CI on all exponents | **Done** ([experiments](experiments/v4_3_statistics/)) |
| **V4.4** | 4 non-deep learners (linear/kernel/logistic/GP) — decisive dichotomy | **Done** ([experiment](experiments/v4_0_uniqueness/learning_baselines.py)) |
| **V5.0** | U(1) pure-gauge lattice FIM (non-deep, spatially-parallel control) | **Done** ([experiment](experiments/v5_0_lattice_qcd/)) |
| **V5.0-stats** | Bootstrap CIs + Mann–Whitney U test on the 12-system dichotomy ($p = 1.7 \times 10^{-17}$, complete rank separation, $n_{\text{deep}} = 46$, $n_{\text{rest}} = 50$) | **Done** ([stats](experiments/v5_0_dichotomy_stats/)) |
| **V6.0** | Mechanism — Hanin–Nica log-normal theorem + depth-sweep empirical confirmation (H1 $R^2 = 0.91$, H2 $R^2 = 0.98$) | **Done** ([doc](docs/v6_0_mechanism_hanin_nica.md), [experiment](experiments/v6_0_depth_mechanism/)) |
| **V6.1** | Width sweep — confirms Hanin–Nica width-independence | **Done** ([JSON](experiments/v6_0_depth_mechanism/v6_1_width_sweep.json)) |
| **V6.2** | Trained-NN depth sweep — $\sqrt{L}$ scaling survives training ($R^2 = 0.94$) | **Done** ([experiment](experiments/v6_0_depth_mechanism/trained_depth_sweep.py)) |
| **V6.3** | Layered boolean-circuit depth sweep — substrate-independent $\sqrt{L}$ scaling | **Done** ([experiment](experiments/v6_0_depth_mechanism/bc_depth_sweep.py)) |
| **V6.4** | Transformer depth sweep — attention + residuals preserve $\sqrt{L}$ scaling ($R^2 = 0.97$) | **Done** ([experiment](experiments/v6_0_depth_mechanism/transformer_depth_sweep.py)) |
| **V6.5** | Activation-function depth sweep — ReLU/GELU/tanh/Swish all pass $\sqrt{L}$ ($R^2 \geq 0.97$) with activation-dependent $\sigma$ | **Done** ([experiment](experiments/v6_0_depth_mechanism/activation_sweep.py)) |
| **V7.0** | SU(2) non-abelian lattice gauge — $T_1/T_3 = 4.85$ (CV 3.1 %, 3 seeds), still in O(1–10) non-deep band | **Done** ([experiment](experiments/v7_0_lattice_su2/)) |
| **V7.1** | U(1) lattice $\beta$-sweep (5 couplings 0.1 → 5.0 across deconfinement) — $T_1/T_3 = 1.72$–1.79, gauge-coupling-invariant | **Done** ([experiment](experiments/v5_0_lattice_qcd/beta_sweep.py)) |
| **V8.0** | Binary-tree tensor-network depth sweep — MERA-style substrate test passes $\sqrt{L}$ at $R^2 = 0.99$ | **Done** ([experiment](experiments/v8_0_tensor_network/)) |
| **V9** | Modern-architecture coverage: ResNet, GPT-Tiny, ResNet-50 ImageNet, ViT trajectories | **Done** ([experiments](experiments/v9_modern_arch/)) |
| **V10** | Production-scale baselines | **Done** ([experiments](experiments/v10_baselines/)) |

## Key documents

- [`docs/fim_tier_hierarchy_neurips2026.md`](docs/fim_tier_hierarchy_neurips2026.md) — full paper draft synthesising V1.0 through V10 (panel + mechanism + falsifier + scope narrowing).
- [`docs/v6_0_mechanism_hanin_nica.md`](docs/v6_0_mechanism_hanin_nica.md) — V6.0 derivation of $\log(T_1/T_3) \propto \sqrt{L}$ from Hanin–Nica + log-normal quantile algebra, with V6.1 / V6.2 / V6.4 sub-results.
- [`docs/v1_1_ntk_gap_closure.md`](docs/v1_1_ntk_gap_closure.md) — NTK continuum-limit gap-closure note for the SV-exponent compatibility check.
- [`docs/preregistration_v2.md`](docs/preregistration_v2.md) — pre-registered hypotheses and falsifiers.

## Repository layout

```
docs/                  # Paper drafts and mechanism notes
experiments/           # Per-phase experiment scripts and committed JSON outputs
plots/                 # Figures (PNG)
scripts/               # Reproduction infrastructure (one-command rerun)
runtime/               # Optional: tier-hierarchy runtime hooks (research only)
tests/                 # pytest suite (markers: unit/integration/slow)
```

## License

This repository is released for academic reuse with citation under the terms
in `LICENSE`.

## Citation

```bibtex
@article{anonymous2026fimtier,
  title  = {Fisher Information Tier Hierarchy: A Panel-Bounded Empirical
            Regularity of Deep Layered Sequential Computation},
  author = {Anonymous},
  year   = {2026},
  note   = {Submission under double-blind review.}
}
```
