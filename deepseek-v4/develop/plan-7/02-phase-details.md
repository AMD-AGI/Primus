# 02 — Plan-7 Phase Details

## Phase 45 — Custom Triton fused Adam (absorb ε-add into master functor)

> Sourced from `progress/p41/p41-candidates.md` §3.2 row #1 + #2
> (combined 217 ms / iter Adam residual).

P45 replaces the TE / Apex `multi_tensor_adam_master_param_remainder`
+ the separate BF16 ε-add chain with a single Triton kernel per
multi-tensor group that does the full Adam step in registers.

### Tasks

1. **`primus/backends/megatron/extensions/_triton/fused_adam.py`** —
   new file with `_fused_adam_master_kernel` and a
   `FusedAdamMasterParamRemainder` callable matching the upstream
   signature.  The kernel signature is:

   ```python
   @triton.jit
   def _fused_adam_master_kernel(
       g_ptr, p_ptr, m_ptr, v_ptr, master_p_ptr, remainder_ptr,
       lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
       grad_scale, step, n_elements, BLOCK_SIZE: tl.constexpr,
   ): ...
   ```

   Internally: `m = beta1*m + (1-beta1)*g`,
   `v = beta2*v + (1-beta2)*g**2`, `m_hat = m / bias_correction1`,
   `v_hat = v / bias_correction2`,
   `update = lr * m_hat / (sqrt(v_hat) + eps) + lr * wd * master_p`,
   `master_p -= update`, `(p, remainder) = bf16_cast_round(master_p, remainder)`.
   All math in registers — the ε-add is fused with the
   `sqrt(v_hat)` and the bf16-cast remainder accumulation is
   fused at the end.

2. **`primus/backends/megatron/patches/turbo_adam_patches.py`** —
   new monkey-patch.  Probes for the upstream multi-tensor functor
   (TE's `transformer_engine.optimizers.multi_tensor_adam_master_param_remainder`
   first; Apex's `apex.optimizers.multi_tensor_adam` fallback).
   Wraps it with a dispatcher: when `PRIMUS_FUSED_ADAM_TRITON=1`,
   route through `FusedAdamMasterParamRemainder`; else fall through
   to the upstream functor.  Patch installs in
   `before_train` (same phase as the V4 flops patch).

3. **Microbench `progress/p45/bench_fused_adam.py`** — covers the
   V4-Flash production param-list shape distribution (captured from
   a real smoke step's chunk-size histogram).  Reports wall-clock +
   GB/s + speedup vs upstream.  `iters=20, warmup=5,
   n_input_copies=4, l2_flush_mb=512`.

4. **EP8 proxy A/B trace** — `progress/p45/run_baseline_trace_ep8_p45.sh`
   with `PRIMUS_FUSED_ADAM_TRITON=1`; render
   `develop/profile/profile-after-p45-ep8-<YYYYMMDD>.{md,html}`.
   A side is `PRIMUS_FUSED_ADAM_TRITON=0` (P44 production).

5. **G47 — unit tests** —
   `tests/unit_tests/megatron/extensions/test_p45_fused_adam_triton.py`:
   - FWD bit-equal vs upstream Adam at fast tier (`[8 params × 4096
     elements]`, fp32, fp32 master).
   - BF16 master + remainder: ULP-difference ≤ 1 vs upstream at
     fast tier; bit-equal upper 16 bits.
   - 10-step micro-rollup: max abs diff in param ≤ 1e-3 vs upstream.
   - Release-tier slow: full V4-Flash param list, 100 steps,
     loss-curve diff ≤ 1e-3 vs upstream at fixed seed.

6. **Default flip note** — `PRIMUS_FUSED_ADAM_TRITON` default is
   `"1"` only if the proxy A/B confirms a positive delta.

### Design notes

- **Why fuse master-param + remainder in one kernel.** The upstream
  TE / Apex multi-tensor Adam returns the master-param update and
  then a separate kernel applies the BF16 cast + remainder
  accumulation.  Two HBM round-trips for the master-param tensor.
  The fused kernel keeps the master-param in registers between
  the update and the cast, saving 1 HBM read + 1 HBM write per
  parameter per step.
- **Why probe both TE and Apex.** Primus uses TE's fused Adam when
  `enable_primus_turbo=True`; the eager / debug path falls back
  to Apex's `multi_tensor_adam`.  The patch supports both so a
  future fallback re-enable doesn't require a Primus-side change.
- **Bias-correction handling.** The upstream functor passes
  `bias_correction1 / bias_correction2` as host-side scalars
  (Python floats); the Triton kernel takes them as `tl.float32`
  arguments.  Computing them inside the kernel would require
  reading the `step` counter as a tensor, which complicates the
  call signature; the host-side approach is simpler and bit-equal
  per the TE reference.

### Edge cases

- **`master_param == None`** — caller didn't allocate the fp32
  master copy.  The patch falls back to the upstream functor
  (this is the standard "skip Adam fusion" code path).
- **`remainder == None`** — older Apex variants don't have the
  remainder accumulator.  Patch falls back to upstream.
- **`weight_decay == 0`** — kernel branches on a `tl.constexpr` so
  the weight-decay branch is dead-eliminated.

---

## Phase 46 — Fused grad-scale Triton kernel

> Sourced from `progress/p41/p41-candidates.md` §3.2 row #3
> (`multi_tensor<scale>` 10.96 ms / 321 launches).

P46 absorbs the per-param `multi_tensor<scale>` calls into a
single Triton kernel that runs once per multi-tensor batch group.

### Tasks

1. **`primus/backends/megatron/extensions/_triton/fused_grad_scale.py`** —
   new file with `_fused_grad_scale_kernel` and a
   `FusedGradScale` callable.  The kernel scales gradients in
   place via `g.mul_(scale)`.

2. **`primus/backends/megatron/patches/turbo_adam_patches.py`** —
   extend with a `multi_tensor_scale` dispatcher.  Gate
   `PRIMUS_FUSED_GRAD_SCALE=1` (default `"0"` then `"1"` after
   A/B).

3. **Microbench `progress/p46/bench_fused_grad_scale.py`** —
   covers the V4-Flash production param-list shape distribution.

4. **EP8 proxy A/B trace** —
   `progress/p46/run_baseline_trace_ep8_p46.sh`.

5. **G48 — unit tests** —
   `tests/unit_tests/megatron/extensions/test_p46_fused_grad_scale.py`:
   - FWD bit-equal vs upstream `multi_tensor_scale` at fast tier
     fp32 + bf16.

6. **Default flip note** — `PRIMUS_FUSED_GRAD_SCALE` default is
   `"1"` only if the proxy A/B confirms a positive delta.

### Design notes

- This is the simplest of the plan-7 phases.  The win comes from
  collapsing 321 small launches into one.

---

## Phase 47 — Fused grad-norm clip Triton kernel

> Sourced from `progress/p41/p41-candidates.md` §3.2 row #4
> (`reduce<l2norm_bf16>` 7.76 ms + `multi_tensor<l2norm>` 6.72 ms).

P47 fuses the L2-norm reduce + max-with-existing-norm +
clip-scale derivation + apply-clip chain into a 3-kernel pipeline.

### Tasks

1. **`primus/backends/megatron/extensions/_triton/fused_grad_norm_clip.py`** —
   new file with three Triton kernels:
   - `_grad_norm_l2_partial_kernel`: per-param reduce, emits
     partial L2 norm into a small scratch tensor.
   - `_grad_norm_global_kernel`: reduces partials across params,
     applies `max_norm` clamp, derives clip-scale.
   - `_grad_clip_apply_kernel`: applies clip-scale to each
     gradient in place.

2. **`primus/backends/megatron/patches/turbo_adam_patches.py`** —
   extend with a `clip_grad_norm` dispatcher.  Gate
   `PRIMUS_FUSED_GRAD_NORM_CLIP=1` (default `"0"` then `"1"` after
   A/B).

3. **Microbench `progress/p47/bench_fused_grad_norm_clip.py`** —
   covers the V4-Flash production param-list shape distribution.

4. **EP8 proxy A/B trace** —
   `progress/p47/run_baseline_trace_ep8_p47.sh`.

5. **G49 — unit tests** —
   `tests/unit_tests/megatron/extensions/test_p47_fused_grad_norm_clip.py`:
   - FWD bit-equal vs upstream `clip_grad_norm_fp32` at fast tier
     fp32 + bf16.  L2 norm is associative so the reduction order
     matters; the kernel uses the same order as the upstream
     multi-tensor functor (left-to-right, no block-shuffle).

6. **Default flip note** — `PRIMUS_FUSED_GRAD_NORM_CLIP` default
   is `"1"` only if the proxy A/B confirms a positive delta.

### Design notes

- **Why 3 kernels and not 1.** The L2 norm requires a global reduce
  across all params + ranks (cross-rank via NCCL allreduce).  A
  single-kernel approach would either need a custom NCCL-from-Triton
  call (not available) or a megakernel with a sync barrier (Triton
  doesn't support cooperative-group sync across SMs).  Three
  kernels with a small inter-kernel sync is the standard pattern.

---

## Phase 48 — Plan-7 close-out

> Hand-off phase.  No new kernels.

### Tasks

1. **`develop/perf/proxy_ep8.md`** — append `P45`, `P46`, `P47`,
   `P48 final` rows.  `P48 final` records the cumulative iter time
   + corrected TFLOP/s/GPU + `vs P28 baseline` (the perpetual
   anchor) + `vs P44 final` (the plan-7 starting point).

2. **`develop/perf/elem_fusion.md`** — append plan-7 rows (one per
   shipped fusion).  Cell format per R2.5: `<ms> ms |
   <effective GB/s>` (plan-7 kernels are memory-bandwidth-bound).

3. **`progress/p45/p45-summary.md` ... `progress/p48/p48-summary.md`** —
   one R2.1 eight-section summary per phase.

4. **`run_deepseek_v4_flash_proxy.sh`** — surface the three new
   env knobs (`PRIMUS_FUSED_ADAM_TRITON`,
   `PRIMUS_FUSED_GRAD_SCALE`, `PRIMUS_FUSED_GRAD_NORM_CLIP`) under
   `${VAR:-1}`-guard.  Header gains a "Plan-7 optimizer-step
   fusion knobs" section mirroring the plan-6 P40 precedent.

5. **Status pinning per R2.4** — every `[x]` row in Phase 45..48
   gets the commit SHA + date pinned.

6. **15-iter clean bake-off** — `progress/p48/run_smoke_p48_bakeoff.sh`
   with all plan-7 default-on knobs.  Steady iter time + TFLOP/s/GPU
   per iter; report in `progress/p48/p48-summary.md` §3.

7. **Plan-7 close-out commit** — final commit message follows the
   plan-6 P40 convention: `docs(deepseek-v4)[plan-7][P48]:
   plan-7 close-out — proxy bake-off + perf docs + status pinning`.
