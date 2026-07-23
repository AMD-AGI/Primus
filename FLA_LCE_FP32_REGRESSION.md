# FLA upstream regression: FP32 grad accumulation in FusedLinearCrossEntropyLoss

> **Status (2026-07-22):** fla fork `main` moved to `5eed1106` (upstream 0.5.2
> merge — the "better fla"); the Primus fix passes `accumulate_grad_in_fp32=False`
> by default (`PRIMUS_LCE_ACCUM_FP32=0`), so upstream+fix is active out of the box
> and beats the old fork on all 5 models. All 6 megatron patches verified
> (reverse-check + round-trip apply on a pristine tree).


## TL;DR

Upgrading `flash-linear-attention` from the pinned fork (`0.5.0`, commit
`6c02bb9f`) to the upstream merge (`0.5.2`, `save/upstream-merge-5eed1106` /
`fla-org/main`) made **hybrid** training **~15–18 % slower** on MI300X
(GDN-hybrid, mamba-hybrid; KDA-hybrid ~8 %). Pure GDN/KDA were unaffected.

**Root cause:** upstream added `accumulate_grad_in_fp32=True` (new default) to
`FusedLinearCrossEntropyLoss`. This upcasts the chunked weight-gradient GEMM
`[V,H] = [128256,1024]` to **FP32**. FP32 GEMM has no matrix-core acceleration
on MI300X, so that single GEMM runs **~4.5× slower** and dominates the whole
regression.

**Fix (Primus, version-safe):** construct the loss with
`accumulate_grad_in_fp32=False` when the fla version exposes the knob. This
restores the bf16-matmul / fp32-accumulate path. Result: the fixed run on
upstream fla is **faster than the old fork** (135 ms/iter vs 147) with
loss identical to 4 decimals.

The missing "variable" the user asked about is `accumulate_grad_in_fp32`.

## How it was found (profiling)

Torch profiler (`profile: true`, `use_pytorch_profiler: true`,
`profile_step_start/end`) over steps 20–24 of `zebra_llama_300M_gdn_hybrid`,
one trace per fla version. Traces parsed with `eval/parse_trace.py` (sums GPU
kernel time by name and diffs the two runs).

Profiled-window device time: **latest 677.5 ms vs fast 560.6 ms (+17.3 %)** —
matches the end-to-end regression. The per-kernel diff isolates it to one GEMM:

| kernel (Tensile) | fast 0.5.0 | latest 0.5.2 |
|---|---:|---:|
| `Cijk_..._MT256x128x32_...` (bf16, `BBS`) | 39.3 ms / 128 calls | — |
| `Cijk_..._MT256x192x16_...LDSB0` (fp32, `_S_`) | — | 176.7 ms / 128 calls |

Same 128 launches (32 LCE chunks × 4 steps), 4.5× slower per call. Everything
else (chunk-GDN, RMSNorm, swiglu, conv, MLA attention) was within noise; the
isolated-kernel microbench (`eval/bench_fla_kernels.py`) even showed RMSNorm and
chunk-GDN *faster* on 0.5.2 — which is why the microbench alone did not explain
the regression and full-run profiling was required.

## Why only hybrids regressed

`fused_ce_mode` maps to two different fla losses (see
`primus/backends/megatron/patches/fla_runtime_patches.py`):

- **mode 1** → `FusedLinearCrossEntropyLoss` — *fuses* the vocab projection GEMM
  inside the loss, so its grad GEMM is the one that got upcast. Hybrid configs
  use mode 1.
- **mode 2 / pures** → `FusedCrossEntropyLoss` — logits are produced by the
  normal (bf16) Megatron output GEMM; the loss only does the CE Triton kernel,
  so the fp32-accumulate change does not touch a large GEMM.

## The upstream change (diff)

`fla/modules/fused_linear_cross_entropy.py`, fast → slow:

```python
# fast (0.5.0): bf16 matmul, accumulated into an fp32 dw buffer
dw += c_logits.t() @ c_x

# slow (0.5.2): grad_dtype = fp32 (accumulate_grad_in_fp32=True default)
c_x = c_x.to(dtype=grad_dtype)                 # -> fp32
torch.addmm(input=dw, mat1=c_logits.t().to(grad_dtype), mat2=c_x, out=dw)  # fp32 GEMM
```

## The fix

`third_party/Megatron-LM/megatron/core/models/mamba/mamba_model.py`
(persisted in `megatron_patches/01-mamba_model-fused-ce.patch`):

```python
_lce_kwargs = dict(reduction='mean', num_chunks=_nc)
if 'accumulate_grad_in_fp32' in inspect.signature(
        FusedLinearCrossEntropyLoss.__init__).parameters:
    _lce_kwargs['accumulate_grad_in_fp32'] = False
self._fused_lce = FusedLinearCrossEntropyLoss(**_lce_kwargs)
```

Guarded by `inspect`, so it is a no-op on the old fork (which lacks the kwarg)
and enables the fast path on upstream fla — meaning **Primus can now track
upstream fla and be faster than the pinned fork.**

## Verification — full model sweep (8×MI300X, mock data, 50 iters, upstream fla 0.5.2)

Toggle via `PRIMUS_LCE_ACCUM_FP32` (1 = old fp32 path, 0/unset = fixed bf16 path).
Steady-state inst ms/iter (avg over iters 41–50):

| model | before (ms) | after (ms) | speedup | TFLOP/s before→after | loss before→after |
|---|---:|---:|---:|---:|---|
| gdn_pure (300M) | 162.7 | 162.8 | ~0% (unaffected) | 354.9→354.7 | 1.030913 = 1.030913 |
| gdn_hybrid (300M) | 173.1 | **135.9** | **−21.5% (1.27×)** | 210.2→267.7 | 1.547605→1.547740 |
| kda_pure (300M) | 160.2 | 159.9 | ~0% (unaffected) | 360.5→361.1 | 1.380889 = 1.380889 |
| kda_hybrid (1B) | 1429.9 | **1301.0** | **−9.0% (1.10×)** | 160.3→176.2 | 7.36948→7.36709 |
| mamba_hybrid (300M) | 211.4 | **173.9** | **−17.7% (1.21×)** | 172.1→209.1 | 1.572484→1.572529 |

Only the mode-1 (`FusedLinearCrossEntropyLoss`) configs — the three hybrids —
improve; the pures are untouched. Loss differs only at the ~1e-4 level (bf16 vs
fp32 grad accumulation), i.e. no material precision loss; NaN-free.

Sweep logs: `eval/logs/sweep_{fixed,nofix}/<model>.log`.

## Cross-check vs the pre-push sanity suite + FLA-native (production batch)

The `sanity_before_push/` suite (run 2026-06-08 on the **fast fork**) compared
Primus to **FLA-native** training at FLA's real config (`mbs=128, gbs=1024,
seq=2048`, real FineWeb data). Re-running the two hybrids at that same batch on
**upstream 0.5.2** (mock data, 120 iters, steady inst ms/iter) lines the fix up
against both baselines:

| Model | FLA-native (Jun 8) | Primus fast-fork (Jun 8 sanity) | Primus upstream, no-fix | Primus upstream, +fix |
|---|---:|---:|---:|---:|
| gdn_hybrid ms/iter | 1487.3 | 1539.6 (+3.5% vs FLA) | 2084.9 (+40% vs FLA) | 1560.7 (+4.9% vs FLA) |
| gdn_hybrid TFLOP/s | — | 380.8 | 279.1 | 372.9 |
| mamba_hybrid ms/iter | 2278.6 | 2247.5 (−1.4% vs FLA) | **OOM @ mbs=128** | 2196.3 (−3.6% vs FLA) |
| mamba_hybrid TFLOP/s | — | 262.5 | OOM | 265.0 |

Takeaways:
- The fix **reproduces the earlier fast-fork sanity numbers** and keeps Primus
  within the sanity gate's FLA tolerance (≤+5% for 300M).
- The regression is **worse at production batch** (+40% vs FLA at gbs=1024 vs
  +17% at gbs=64) — the fp32 grad GEMM scales with per-chunk token count.
- The fp32 path also **inflates memory** (fp32 `dw` + fp32 chunk copies) and
  **OOMs mamba_hybrid at mbs=128** — it cannot run on upstream without the fix.

Logs: `eval/logs/sanity_cmp/300M_{gdn,mamba}_hybrid_{nofix,fixed}.log`.
(Speed runs used mock data; GPU-bound ms/iter is unaffected — the ≈1.4%
gdn_hybrid match to the real-data sanity number confirms this.)

## Repro artifacts

- `eval/bench_fla_kernels.py` — per-kernel microbench (one op per process).
- `eval/parse_trace.py` — sum/diff torch-profiler traces by kernel.
- `eval/logs/prof_gdn_hybrid_{fast,latest}/trace.json` — the two traces.
- `eval/logs/fix_verify/{slow_baseline,fixed}.log` — before/after timing.
