---
name: optimize-compute
description: Strategies for compute-bound Primus training (kernels not utilizing GPU). Use when bottleneck-diagnose returned COMPUTE_BOUND, when gpu_util_avg is low (< 0.7) but comm/bubble/mem are all fine, when MFU is far below cluster GEMM peak, when the user asks about kernel utilization, mbs scaling, fused kernels, kernel hints, flash attention, transformer engine, fp8, hipblaslt tuning, or threading env (OMP_NUM_THREADS, MKL_NUM_THREADS, numactl).
---

# Optimize for Compute Bottleneck

The other axes are quiet but the GPU is not running at peak. Make the kernels bigger / better fused / more cache-friendly.

## Strategy Decision Table

| Tier | Move | When | Cost | Typical gain |
|---|---|---|---|---|
| 0 | Threading env (OMP/MKL) | host-side starvation suspected | env-sweep | 2–8% |
| 0 | HSA / queue env | known good defaults vary by ROCm | env-sweep | 1–5% |
| 1 | Raise `mbs` | mem headroom, small kernels | flag | 5–25% |
| 1 | Enable / fix fusion | available but not enabled | flag | 5–15% |
| 1 | FP8 (where supported) | model + kernels support it | flag | 30–60% on T_comp |
| 2 | Drop `tp` | `tp` too big for kernel size | reshape | matrix sizes ↑ |
| 2 | hipblaslt / kernel autotune | repeated shape, peakable | offline tuning | 5–20% |
| 3 | Custom kernel / model-side change | last resort | engineering work | varies |

## Tier 0 — Host-side / threading env

When CPU side starves the GPU (e.g. data loading or torch dispatch), raising thread counts helps.

| Flag | Try | Effect |
|---|---|---|
| `OMP_NUM_THREADS` | `4` / `8` / `16` (depend on cores/rank) | parallelizes CPU ops |
| `MKL_NUM_THREADS` | match `OMP_NUM_THREADS` | MKL parallelism |
| `GPU_MAX_HW_QUEUES` | `2` (default) / `4` / `8` | more concurrent kernel queues |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` | better PCIe granularity |
| `numactl --cpunodebind / --membind` | bind to local NUMA node | DRAM locality |
| `HSA_NO_SCRATCH_RECLAIM` | typically `1` for ROCm | avoid scratch thrash |

Sweep these in one env-sweep round when COMPUTE_BOUND with `gpu_util < 0.65`.

## Tier 1 — Raise mbs (the dominant Tier-1 move)

Bigger microbatches → bigger GEMM tiles → higher MFU.

| Override |
|---|
| `--micro_batch_size <mbs>` |

Order of operations:

1. Check `optimize-memory.OOM gate`: predicted mem at new mbs ≤ 0.92 × hbm.
2. If safe, try `mbs × 2`. Re-diagnose afterward — sometimes you hit MEMORY_BOUND or PIPELINE_BOUND next.
3. Diminishing returns past `mbs ≈ 8` for most dense models.

## Tier 1 — Fusion / kernel flags

The Primus / Primus Turbo fusion catalog (CE / RoPE / grad-accum / Turbo attention / parallel linear / RMS norm) is owned by `primus-defaults` and is normally already on after `round_1`. Re-check it here only if a flag is missing or was disabled by the user's YAML:

| Flag | Owned by `primus-defaults` | Use here |
|---|---|---|
| `cross_entropy_fusion_impl: te` + `cross_entropy_loss_fusion: true` | yes | confirm enabled |
| `apply_rope_fusion: true` (+ `enable_experimental: true`) | yes | confirm enabled |
| `gradient_accumulation_fusion: true` | yes (model-class-conditional — some small dense YAMLs disable it) | check the user's YAML before flipping |
| `enable_primus_turbo: true` + `use_turbo_attention: true` | yes | confirm enabled |
| `use_turbo_parallel_linear: true` | yes (TP > 1 only) | confirm when TP > 1 |
| `use_turbo_grouped_mlp: true` | yes (MoE) | confirm when MoE |
| `use_turbo_rms_norm` | **stays off** (known bug) | leave off |
| `--use_flash_attn` (or backend equivalent) | n/a — model-side | enable when supported and Turbo attention is off |

If a flag is missing, the right action is to re-run `round_1` (primus-defaults batch) rather than do a one-flag candidate here. This skill takes over only when COMPUTE_BOUND persists *after* primus-defaults has been applied.

## Tier 1 — FP8 (where supported)

If the cluster reports `peak_tflops_fp8 ≈ 2 × peak_tflops_bf16` (preflight) and the model + backend support it:

| Override | Effect |
|---|---|
| `--fp8 hybrid` (TE) / backend equivalent | mixed BF16/FP8 GEMMs |
| FP8 recipe flags (`--fp8_amax_history_len`, `--fp8_amax_compute_algo`) | numerical stability knobs |

Risks:

- Numerical drift on long runs — re-baseline correctness with a short run that compares loss vs BF16 reference.
- Some models are not yet FP8-stable; check `output/pilot/cluster-*.md` notes for known-good models.

## Tier 2 — Drop tp (when `tp` is too big)

If `tp = 8` but the per-rank GEMM size is now small enough to leave SMs idle, dropping `tp` makes the kernels bigger:

| Move | Side effect |
|---|---|
| `tp 8 → 4` | per-rank matrix doubles → MFU ↑; M_param/M_grad ÷ tp doubles → memory ↑ |
| `tp 4 → 2` | further doubling; very mem-aware |

Always re-check `optimize-memory.OOM gate` after dropping tp.

## Tier 2 — hipblaslt offline tuning

Primus ships a HipblasLt auto-tune flow (see `examples/README.md` "HipblasLT Auto Tuning"). When the same training shape will run many times, it pays off:

```
Stage 1: dump GEMM shapes (one short run with HIPBLASLT_LOG_LEVEL=4)
Stage 2: tune (offline, separate job)
Stage 3: train with the tuned kernel cache
```

Apply only when the champion plan is stable and you expect the model to run repeatedly. One-off tuning runs don't repay the cost.

## Tier 3 — Model-side changes

Last resort: change the model code (different attention impl, different fused ops). Out of scope for Pilot — escalate to the model owner with the profiler trace as evidence.

## env_diff template

```yaml
env_diff:
  OMP_NUM_THREADS: 8
  MKL_NUM_THREADS: 8
  GPU_MAX_HW_QUEUES: 4
  HSA_FORCE_FINE_GRAIN_PCIE: 1
```

(See `env-catalog` for the full HSA / threading dictionary.)

## Constraints

| Combo | Issue |
|---|---|
| `OMP_NUM_THREADS` very large (≥ 32) | thread contention, can hurt |
| `GPU_MAX_HW_QUEUES > 8` | usually no benefit, wastes resources |
| FP8 + small mbs | FP8 kernel fixed cost dominates; bf16 may be faster |
| `--use_flash_attn True` on unsupported model | silent fallback to slow path |

## Worked example

```
Champion P_c: tp=4 pp=1 mbs=2; comm_ratio=0.10, bubble=0.0, mem=120 GB, gpu_util=0.55
  → bottleneck-diagnose: COMPUTE_BOUND

Tier-0 env-sweep (host/threading):
  E1: OMP_NUM_THREADS=8
  E2: + GPU_MAX_HW_QUEUES=4
  E3: + HSA_FORCE_FINE_GRAIN_PCIE=1
  Winner E3: tps +4%

Tier-1 mbs scaling (with E3 merged):
  P1: mbs 2 → 4   pred mem = 145 GB ≤ OOM gate; tps +18%; gpu_util 0.72  ← champion
  P2: mbs 2 → 8   pred mem = 195 GB > 0.92 × 192 = 176 → REJECT (predicted OOM)

Re-diagnose on P1: gpu_util 0.72 — still some headroom but diminishing.
Optionally Tier-1: enable TE attention if available; otherwise stop.
```

## Important Notes

- **Raising `mbs` is the dominant compute move** — try it before kernel hints.
- **Watch the OOM gate every time you raise mbs**. Compute optimization routinely produces MEMORY_BOUND in the next round.
- **Threading env is cheap and easy to forget**. The first COMPUTE_BOUND env-sweep should always include OMP / GPU_MAX_HW_QUEUES.
- **FP8 gains are real but require numerical re-validation**. Always pair with a CORRECTNESS-style smoke that compares loss curves to the BF16 reference.
- **Don't enable kernel hints you cannot verify**. Silent fallback flags waste rounds.
- **HipblasLt tuning is for steady-state production**. Skip it during exploratory tuning unless the user has explicitly asked for it.
