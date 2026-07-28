# MXFP4 training for Flux models

Guide for training Flux diffusion models in **MXFP4** (E2M1 mantissa + E8M0 block-of-32 scales) on AMD MI355X GPUs using Primus's local-spec MXFP4 implementation backed by Primus-Turbo and AITER.

## Overview

MXFP4 stores activations and weights in 4-bit microscale floating-point with one E8M0 exponent shared per block of 32 elements. The Primus integration:

- Uses a **local spec** (`PrimusTurboMXFP4LocalSpecProvider`) with **no Transformer Engine dependency**—MXFP4 linear layers are self-contained autograd `Function`s that call Primus-Turbo's `gemm_fp4_impl` directly, so the path is `torch.compile`-friendly with minimal graph breaks.
- Keeps **attention, optimizer state / main params, and inter-rank communication in BF16**. Only the MMA inputs of the column- and row-parallel linears are quantized.
- Supports two backward modes via `mxfp4_backward_precision`: pure **MXFP4** (default) or **FP8** hybrid (E5M2 backward with tensorwise scaling on HipBLASLt).
- Dispatches the FP4 GEMM through Primus-Turbo's pluggable backend layer, which can route to either AITER (recommended for MI355X) or HipBLASLt.

## Table of contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Primus-Turbo backend selection](#primus-turbo-backend-selection)
- [Tuned GEMMs](#tuned-gemms)
- [Troubleshooting](#troubleshooting)
- [Verification status](#verification-status)

---

## Prerequisites

### Hardware

- **AMD Instinct MI355X** (gfx950) with FP4 tensor-core support. The MXFP4 linear-layer modules assert `check_mxfp4_support()` at construction and will refuse to initialize on unsupported devices ([`primus/backends/megatron/core/extensions/primus_turbo_mxfp4_local.py`](../../../primus/backends/megatron/core/extensions/primus_turbo_mxfp4_local.py)).
- Single node (the local-spec layers require `tensor_model_parallel_size: 1`).

### Software

- ROCm-compatible install of `aiter` (provides `aiter.gemm_a4w4` and the tuned-config loader in `aiter/jit/core.py`).
- Primus-Turbo with FP4 backend registered (`primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl`).
- `enable_primus_turbo: true` and `use_turbo_attention: true` in the training config.

---

## Quick start

The verified MXFP4 config is `examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_mxfp4.yaml`. Launch with the AITER backend and the pre-tuned GEMM CSV:

```bash
# Path to a checkout of the `tuned_gemm_configs` directory.
# Set TUNED_GEMM_DIR to wherever you have the tuned configs available.
export TUNED_GEMM_DIR=${TUNED_GEMM_DIR:-/path/to/tuned_gemm_configs}

export EXP=examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_mxfp4.yaml
export PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER
export AITER_CONFIG_GEMM_A4W4=$TUNED_GEMM_DIR/mi355x/flux_12b.csv
export AITER_LOG_TUNED_CONFIG=1   # recommended: confirms each shape hits the CSV

bash examples/run_pretrain.sh
```

The pre-tuned CSV is distributed via an internal tuned-config source (`tuned_gemm_configs/mi355x/flux_12b.csv`). If you do not have access, omit `AITER_CONFIG_GEMM_A4W4` and AITER will fall back to its bundled `a4w4_blockscale_tuned_gemm.csv` (slower for Flux 12B shapes).

---

## Configuration

The relevant overrides in [`examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_mxfp4.yaml`](../../../examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_mxfp4.yaml):

```yaml
# MXFP4 precision
fp4: "mxfp4"
fp4_recipe: "mxfp4"              # default is "nvfp4" in trainer_base.yaml; must override
mxfp4_backward_precision: "mxfp4"  # "mxfp4" (pure) or "fp8" (hybrid)

# Local spec + Primus-Turbo
transformer_impl: "local"
enable_primus_turbo: true
use_turbo_attention: true

# Required by the MXFP4 linear-layer modules
tensor_model_parallel_size: 1
gradient_accumulation_fusion: false
# sequence_parallel must remain false
```

### Knob semantics

| Knob | Values | Notes |
|------|--------|-------|
| `fp4` | `"mxfp4"` | Top-level switch to enable FP4. |
| `fp4_recipe` | `"mxfp4"` for this guide | Default in [`primus/configs/modules/megatron/trainer_base.yaml`](../../../primus/configs/modules/megatron/trainer_base.yaml) is `nvfp4`; the MXFP4 config overrides it. |
| `mxfp4_backward_precision` | `"mxfp4"` or `"fp8"` | Exhaustive set (checked by branch in [`primus_turbo_mxfp4_local.py`](../../../primus/backends/megatron/core/extensions/primus_turbo_mxfp4_local.py)). `"fp8"` uses E5M2 with tensorwise HipBLASLt for backward. |
| `mxfp4_gradient_stochastic_rounding` | `true` / `false` | Optional. Enables SR on FP4 gradient quantization. |

---

## Primus-Turbo backend selection

The FP4 GEMM call is routed by `GEMMFP4KernelDispatcher` in `Primus-Turbo/primus_turbo/pytorch/kernels/gemm/gemm_fp4_impl.py`. Backends are selected with the precision-scoped env var `PRIMUS_TURBO_GEMM_BACKEND` (declared in `Primus-Turbo/primus_turbo/common/constants.py`):

```bash
# Single backend for every precision:
export PRIMUS_TURBO_GEMM_BACKEND=AITER

# Precision-scoped (recommended): route FP4 GEMMs to AITER, leave others to defaults:
export PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER

# Per-precision routing:
export PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER,FP8:HIPBLASLT
```

The dispatcher (`GlobalBackendManager` / `AutoKernelDispatcher.dispatch` in `Primus-Turbo/primus_turbo/pytorch/core/backend.py`) resolves the backend in this order: **explicit env > code-set > auto-tune > registered default > fallback**.

### Preshuffle fast path

When **all** of the following are true, MXFP4 GEMMs take the preshuffled fast path with no per-call shuffle overhead:

- `PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER` (or `AITER`) is set.
- `PRIMUS_TURBO_AUTO_TUNE` is unset or `0`.

The `_enable_preshuffle()` helper in `primus/backends/megatron/core/extensions/primus_turbo_mxfp4_local.py` returns `True` under these conditions (FP4 backend pinned to AITER and auto-tune off), and the call becomes `aiter.gemm_a4w4(..., bpreshuffle=True)`. This helper reproduces the upstream `enable_preshuffle()` that Primus-Turbo removed in PR #383 ("refactor preshuffle ..."), which moved per-call preshuffle control onto `Float4QuantConfig.use_preshuffle`; Primus keeps the runtime probe locally because `MXFP4LinearFunction` passes a plain `bool` into its custom ops.

> **Do not combine `PRIMUS_TURBO_AUTO_TUNE=1` with a tuned CSV.** Auto-tune disables the preshuffle fast path, so each call pays the shuffle cost while AITER still picks the same kernel internally. For production runs, leave `PRIMUS_TURBO_AUTO_TUNE` unset.

---

## Tuned GEMMs

AITER reads its tuned-GEMM CSV from the `AITER_CONFIG_GEMM_A4W4` env var (handled in `aiter/jit/core.py`; the default is the bundled `aiter/configs/a4w4_blockscale_tuned_gemm.csv`). Each row maps `(cu_num, M, N, K)` to a profiled kernel and split-K factor.

For Flux 12B on MI355X, the pre-tuned CSV is provided by an internal tuned-config source (`tuned_gemm_configs/mi355x/flux_12b.csv`). See that directory's `README.md` for the tuning runbook, CSV schema, ASM-vs-CK kernel distinction, and re-tuning triggers.

### Verifying the CSV is being used

Set `AITER_LOG_TUNED_CONFIG=1`. AITER will log one line per **hit**:

```
shape is M:16384, N:9216, K:3072, found padded_M: 16384, N:9216, K:3072 is tuned on cu_num = 256 in /path/to/tuned_gemm_configs/mi355x/flux_12b.csv, kernel name is _ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E, splitK is 0!
```

**Miss** lines are printed unconditionally (no env var required) and look like:

```
shape is M:..., N:..., K:..., not found tuned config in /path/to/flux_12b.csv, will use default config!
```

Any miss line means the CSV needs re-tuning for that shape—follow the runbook in `tuned_gemm_configs/README.md`.

### First-run JIT compile

The two `a4w4_blockscale_*_intrawave_v3` CK kernels are JIT-compiled on first use (~2-5 min). Subsequent runs reuse the cached `.so` files. The ASM `f4gemm_bf16_per1x32Fp4_BpreShuffle_*` kernels are pre-compiled blobs shipped with AITER and incur no JIT cost.

---

## Troubleshooting

### `not found tuned config in {file}, will use default config!`

The (M, N, K) shape is missing from your CSV. AITER will fall back to its compiled-in default, which is typically slow. Re-tune for that shape per the internal `tuned_gemm_configs/README.md` runbook (capture the shape via the same log line, append to the untuned CSV, re-run the AITER tuner, commit the new CSV).

### Slow first iteration (~minutes), normal afterwards

Expected—the first call to a CK-based `a4w4_blockscale_*` kernel triggers JIT compilation. Cached `.so` files are reused on subsequent starts.

### `User specified backend AITER cannot handle the given inputs`

Raised by `AutoKernelDispatcher.dispatch` when `GEMMFP4AITERBackend.can_handle` rejects the input. Common causes:

- `M` not a multiple of 16, or `N` not a multiple of 16 (constants `AITER_FP4GEMM_M_MULTIPLE` / `AITER_FP4GEMM_N_MULTIPLE` in `gemm_fp4_impl.py`).
- Unsupported dtype combination (only `(float4_e2m1fn_x2, float4_e2m1fn_x2, fp16/bf16)` is supported).
- Non-NT layout (`trans_a=False, trans_b=True, trans_c=False`).

Workaround: switch to `PRIMUS_TURBO_GEMM_BACKEND=FP4:HIPBLASLT` for unsupported shapes, or pad/reshape inputs.

### `MXFP4ColumnParallelLinear requires tensor_model_parallel_size=1`

The MXFP4 linear-layer modules assert on `tensor_model_parallel_size == 1`, `gradient_accumulation_fusion == False`, and `sequence_parallel == False` ([`primus_turbo_mxfp4_local.py`](../../../primus/backends/megatron/core/extensions/primus_turbo_mxfp4_local.py)). Adjust the config accordingly.

### NaN losses

Switch to the hybrid backward mode, which keeps the FP4 forward but does the gradient GEMM in FP8 (E5M2 tensorwise on HipBLASLt):

```yaml
mxfp4_backward_precision: "fp8"
```

If NaNs persist, also try `mxfp4_gradient_stochastic_rounding: true`.

---

## Verification status

The public config has been smoke-tested end-to-end: 1000 iters on 8x MI355X (single node, micro-batch 64 / global 512, sequence length 512) completes in ~16-20 minutes with `PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER` and the tuned CSV. No errors across ranks; `pretrain() completed successfully`.

Formal A/B benchmarks vs BF16 and FP8 (delayed and tensorwise) are pending and will be published once a representative suite is run; do not rely on the wall-clock numbers above as performance characterizations.

---

## Source code pointers

- MXFP4 spec provider: [`primus/backends/megatron/core/extensions/primus_turbo_local_spec.py`](../../../primus/backends/megatron/core/extensions/primus_turbo_local_spec.py) (`PrimusTurboMXFP4LocalSpecProvider`).
- MXFP4 linear-layer autograd / fwd-bwd: [`primus/backends/megatron/core/extensions/primus_turbo_mxfp4_local.py`](../../../primus/backends/megatron/core/extensions/primus_turbo_mxfp4_local.py).
- Config schema defaults: [`primus/configs/modules/megatron/trainer_base.yaml`](../../../primus/configs/modules/megatron/trainer_base.yaml).
- Dataclass field `mxfp4_backward_precision`: [`primus/backends/megatron/core/models/diffusion/common/config.py`](../../../primus/backends/megatron/core/models/diffusion/common/config.py).
- FP4 backend selection (Primus-Turbo): `primus_turbo/common/constants.py`, `primus_turbo/pytorch/core/backend.py`, `primus_turbo/pytorch/kernels/gemm/gemm_fp4_impl.py`.
- AITER tuned-config loader: `aiter/jit/core.py` (`AITER_CONFIG_GEMM_A4W4`).
- AITER A4W4 dispatch + hit/miss logging: `aiter/ops/gemm_op_a4w4.py`.

## Related documentation

- [FP8 Training Guide](fp8_training.md)—companion guide for FP8.
- [Diffusion Architecture / Developer Guide](README.md).
- [Diffusion Examples README](../../../examples/megatron/diffusion/README.md).
