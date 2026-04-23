# V3.0 — Cluster Recipe for 10^12-Parameter Scaling

**STARGA, Inc. — Research Document**
**Series:** Universe as a Self-Organizing Neural Network
**Phase:** V3.0 (Recipe, not executable on single-workstation hardware)
**Depends on:** `scaling_experiment.py`, `scaling_experiment_extended.py`, `docs/v1_1_ntk_continuum_limit.md`

---

> **Scope notice.** This document is a *recipe* — a reproducible protocol for
> running the V3.0 scaling sweep on a multi-GPU / multi-node cluster. No
> experimental numbers are reported here; the single RTX 3080 workstation
> used for V1.0/V1.2 cannot host 10^12-parameter runs. The protocol is
> written to be directly executable on an 8× to 64× H200 / H100 / MI300X
> cluster by a competent ops engineer.

---

## 1. Goal

Extend the V1.2 width-sweep power-law fit (SV $\sim N^{0.566}$, $R^2 = 0.84$ over 10 widths;
V1.0 was SV $\sim N^{0.47}$, $R^2 = 0.935$ over 6 widths) by two additional orders of
magnitude: widths producing $10^{11}$–$10^{12}$ parameter counts. This is the empirical
test of the V1.1 NTK continuum-limit theorem at a scale where finite-width corrections
have measurably decayed.

In parallel, V3.0 should also establish the **FIM T1/T3 scaling**, which (per V2.1 QEC
results, $\text{T1/T3} \sim N^{1.386}$, $R^2 = 0.93$) is a more robust observable than
the SV ratio. On the cosmology task the FIM scaling is currently flat across 5 decades;
V3.0 should confirm whether this flatness persists into the cluster-scale regime or
whether a crossover appears.

## 2. Architecture

Same 5-hidden-layer ReLU MLP as V1.0, with width $W$. Parameter count scales
dominantly as $N \approx 3 W^2 + 2 W d$ (three internal $W \times W$ matrices,
two stem/head matrices of size $W \times d$). For $d = 32$:

| Width $W$ | Params $N$       | Approx. FLOPs / step (batch 128) |
|-----------|------------------|----------------------------------|
| 45,000    | $6.08 \times 10^9$   | $3.1 \times 10^{12}$            |
| 100,000   | $3.00 \times 10^{10}$ | $1.5 \times 10^{13}$            |
| 316,000   | $3.00 \times 10^{11}$ | $1.5 \times 10^{14}$            |
| 577,000   | $1.00 \times 10^{12}$ | $5.0 \times 10^{14}$            |

## 3. Hardware Requirements

For each target:

- **$10^{10}$ params** — single H200 SXM (141 GB HBM3e) in bf16 with gradient checkpointing. ~30 min training, ~10 min FIM sampling.
- **$10^{11}$ params** — 8× H200, tensor-parallel across a single node (NVLink). ~2 h training, ~30 min FIM.
- **$10^{12}$ params** — 64× H200 (8-node pod), TP × PP hybrid, NVSwitch + IB. ~8 h training, ~2 h FIM.

Storage: 8 TB NVMe for checkpoints (bf16 params + fp32 optimizer state + FIM diagonal).

## 4. Distributed Strategy

**Tensor parallel**: partition each internal $W \times W$ matrix along the row dimension across the TP group. The self-prediction loss is embarrassingly data-parallel, so we combine with **data parallel** at batch granularity. For $10^{12}$ params we need **pipeline parallel** over the 5 hidden layers: each PP stage holds one $W \times W$ layer.

Suggested config:

| Scale     | TP | PP | DP | GPUs |
|-----------|----|----|----|------|
| $10^{10}$ | 1  | 1  | 1  | 1    |
| $10^{11}$ | 8  | 1  | 1  | 8    |
| $10^{12}$ | 8  | 4  | 2  | 64   |

Recommended frameworks: **Megatron-LM** or **DeepSpeed ZeRO-3 + Megatron-TP**.

## 5. Precision

- **Weights**: bf16 (for memory).
- **Gradients**: accumulate in fp32.
- **Optimizer state (SGD+momentum)**: fp32 (momentum buffer).
- **SVD / FIM post-processing**: compute in fp32, transferring the final weight tensors to fp32 on the analysis node.

Loss in bf16 is fine for MSE self-prediction on $\mathcal{N}(0, I)$ inputs; log-scale loss stabilizes within ~100 steps.

## 6. Training Protocol (per width)

```bash
# On the cluster head node, from the nn_universe repo root:
torchrun \
    --nnodes=$NODES --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HEAD:$PORT \
  experiments/v1_2_scaling/distributed_width_sweep.py \
    --width $WIDTH --dim 32 --hidden-layers 5 \
    --steps 20000 --batch 128 --lr 1e-3 --momentum 0.9 \
    --bf16 --grad-ckpt --tp $TP --pp $PP --dp $DP \
    --fim-samples 500 --seed 42 \
    --out-dir /data/runs/v3_0_$WIDTH
```

(`distributed_width_sweep.py` is a planned deliverable; on v1.0 of this repo it does not yet exist. The single-GPU `scaling_experiment_extended.py` is the template.)

Each run commits a JSON result fragment to `/data/runs/v3_0_<width>/result.json` with the same schema as `scaling_results.json`. A final merge step appends all fragments, re-fits the power law, and writes the V3.0 master result.

## 7. FIM Diagonal at Scale

Diagonal FIM estimation requires per-sample backward passes. At $W = 10^5$, a single backward pass is ~50 GFLOPs, so 500 samples = 25 TFLOPs ≈ 30 s on one H200. At $W = 577{,}000$ ($10^{12}$ params), per-sample backward is ~2.5 TFLOPs; 500 samples = 1.25 PFLOPs ≈ 10 min on an 8× TP node.

Important: the FIM gradient must use a batch size of 1 to be a proper empirical Fisher; DP replication does not help here. Instead we **shard FIM samples** across DP workers (each worker computes 500/DP samples, then average). This gives linear DP speedup on FIM sampling without breaking the per-sample requirement.

## 8. SVD at Scale

For $W = 577{,}000$, each of the 3 internal weight matrices is $5.77 \times 10^5 \times 5.77 \times 10^5$, costing ~200 TFLOPs per full SVD. Only top-3 singular values are needed, so:

- Use randomized SVD (Halko, Martinsson, Tropp 2011). Target rank $k = 32$, oversample $p = 16$, power iterations $q = 3$. Cost: $O(k W^2)$ = ~300 GFLOPs, runs in seconds on a single H200.
- For the smallest singular value (needed for the SV ratio), use inverse iteration on the shifted matrix $A^T A - \sigma I$ with $\sigma$ = 0. This is **numerically hostile** at $W \sim 10^5$+; we instead report $\sigma_1 / \sigma_{k}$ where $k = W$ is the *last* SV computed via a bidiagonalization sweep. Document the $k$ used clearly in results.

## 9. Checkpoint & Re-entry

Each run writes a checkpoint after every 5000 steps, containing the full model state + optimizer state + RNG state. On OOM or node failure, re-running with the same `--seed` and same config resumes from the last checkpoint. Idempotent: **if `/data/runs/v3_0_<width>/result.json` already exists with `train_steps >= config.steps`, re-running is a no-op**.

## 10. Cost Estimate

| Scale      | GPU-hours | Approx. cost (H200 @ $3/h) |
|------------|-----------|-----------------------------|
| $10^{10}$  | 1          | $3                        |
| $10^{11}$  | 16         | $48                       |
| $10^{12}$  | 512        | $1,540                    |

The V3.0 sweep touches 6 widths × 3 seeds = 18 runs, of which ~3 are at $10^{12}$. Total: ~$5,000 in compute at published H200 rates. One week of wall-clock on an 8-node pod.

## 11. Expected Outcome

Under the V1.1 NTK theorem the FIM spectrum is controlled by the deterministic NTK integral operator as $W \to \infty$. The observed V1.0/V1.2 power-law exponent ($\alpha = 0.47$) must either:

- **Saturate** at some $\alpha^\star < 0.5$ — evidence that finite-width corrections are subdominant and the NTK continuum limit is attained. *This is the prediction.*
- **Keep growing** — evidence that the restricted-class theorem is not capturing the full dynamics. Would require V1.1 to be revisited with a broader class (e.g. mean-field).
- **Collapse** — evidence the power-law is a finite-width artefact. Would falsify §11.7 "scaling behaviour" claim.

## 12. Data Handling & Publication

- Weights: not published (12 × 577k × 577k = 4 TB; impractical + no scientific value).
- Per-layer SVD top-3 and bottom-3 singular values: published.
- FIM diagonal histograms (not raw): published.
- Training loss curves (every 100 steps): published.
- All JSON results under STARGA Commercial License, following the V1.0 template.

## 13. Risks

- **NumPy / SciPy SVD instability** at $W \gg 10^4$: mitigated by randomized SVD.
- **bf16 gradient underflow** on small per-sample loss: use dynamic loss scaling.
- **Network stalls** during AllReduce at TP=8: test with NCCL_DEBUG=INFO first; allocate at least 200 Gb/s/GPU of effective IB bandwidth.
- **Heterogeneous cluster**: pin kernels with `NVIDIA_TF32_OVERRIDE=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for bitwise-reproducible runs across nodes. Seed-robustness is required for V3.0 credibility.

## 14. Open Handoffs

- V3.0 produces the input to refine the V1.1 NTK proof (finite-width correction analysis).
- The $10^{12}$ FIM spectrum, if consistent with the V1.1 prediction, tightens the $\kappa$ coupling constant for V3.1 α-drift to within one order of magnitude.
- V3.0 data is also useful as a baseline for QEC-decoder scaling (V2.1) — the two sweeps at matched parameter counts allow a direct architectural vs. task comparison.

---

**STARGA Commercial License. No co-authorship or AI attribution.**
