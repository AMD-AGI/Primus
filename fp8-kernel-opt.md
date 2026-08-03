# Native Dynamic FP8 Kernel Optimization Design for FLUX on ROCm

## 1. Purpose and scope

This document extends the native TorchAO dynamic tensor-wise FP8 work described
in [`flux-fp8.md`](flux-fp8.md). It reviews the current FLUX implementation and
the ROCm kernel surfaces in Primus-Turbo, AITER, and FlyDSL, and defines a
measurement-first plan for replacing only the kernels that can improve the
existing path.

This document now includes the completed MI355X raw-GEMM screen. The screen
compares the existing TorchAO/TorchTitan-style `torch._scaled_mm` path with
Primus-Turbo HipBLASLt, CK, Triton, and FlyDSL while preserving tensor-wise
E4M3/E5M2 scale semantics.

The first optimization target remains unchanged:

- FLUX.1-Schnell on 8 x MI355X (`gfx950`);
- TorchAO dynamic/current tensor-wise scaling;
- E4M3 forward operands and E5M2 gradient output;
- FP8 wgrad for 190 non-QKV block Linears;
- high-precision wgrad for 38 image/text QKV Linears;
- BF16 attention and surrounding model operations;
- full-graph compiled repeated blocks, FSDP2, activation checkpointing, and
  unchanged checkpoint/convergence semantics.

MXFP8, row-wise/block-wise scaling, FP8 attention, and weight-only inference
kernels are not substitutes for this contract and are not part of the first
kernel replacement.

## 2. Executive conclusion

The raw-GEMM screen did not identify a production replacement for
`torch._scaled_mm`.

1. **Keep the TorchAO/TorchTitan-style `_scaled_mm` path.** Across all 18 real
   forward/dgrad/wgrad cases, direct Primus-Turbo HipBLASLt, CK, Triton, and
   FlyDSL were respectively `56.0%`, `1060.7%`, `35.2%`, and `5.5%` slower in
   summed raw latency than the control.
2. **Short FlyDSL wins were not repeatable.** A 10-iteration screen suggested
   gains for QKV forward/dgrad and MLP-up forward, but a 100-iteration repeat
   made every candidate `0.3-8.4%` slower than `_scaled_mm`.
3. **Do not add the proposed raw-GEMM bridge or backend table.** No backend met
   the `3%` affected-Linear threshold, so a private TorchAO hook and an extra
   runtime dependency would add complexity without an end-to-end opportunity.
4. **TorchTitan is a control-plane reference, not a distinct GEMM kernel.** Its
   relevant tensor-wise training path also resolves through TorchAO and
   `torch._scaled_mm`; it is represented by the control in this experiment.
5. **AITER attention is independently qualified and retained.** The earlier
   failure used the wrong backend name. `flash_attn_aiter` later improved
   three-seed median time-to-quality by `9.6%`, but AITER dense A8W8 still does
   not expose the required tensor-wise E4M3/E5M2 training contract.
6. **The next measured target is norm/pointwise and scheduling overhead.** The
   previous profile already showed TorchAO FP8 GEMM plus cast/scale doing less
   work than NeMo in the matched P1 comparison. Kernel specialization should
   resume only when a new FlyDSL/CK implementation demonstrates a stable K0
   win before integration.

## 3. Reviewed source snapshots

The analysis used the current local worktrees at review time:

| Repository | Reviewed revision | Relevant surface |
|---|---|---|
| Primus | current `feat/zirui/flux-mlperf` worktree | FLUX TorchAO conversion and validation record |
| Primus-Turbo | `56c789e5` | dynamic quantization, FP8 GEMM/autograd, CK/Triton/FlyDSL/HipBLASLt dispatch |
| AITER | `d6de77692` | A8W8 CK/ASM/CKTile/FlyDSL GEMMs and tuned dispatch |
| FlyDSL | `90a24275` | gfx950 FP8 MFMA GEMMs and preshuffle kernels |
| TorchAO | local `/shared_nfs/zirui/code/ao` checkout | `Float8Linear`, dynamic casts, `Float8TrainingTensor`, `_scaled_mm` dispatch |

Primus-Turbo's `3rdparty/composable_kernel` submodule was locally dirty. This
review relies on the visible source interfaces, not on an assumption that the
worktree exactly matches a released wheel. Before implementation, the image
must pin the actual Primus-Turbo/AITER/FlyDSL and CK revisions used to build the
runtime binaries.

## 4. Current native FP8 implementation

### 4.1 Integration boundary

The production implementation is in:

```text
primus/backends/diffusion/models/registrations/flux.py
```

It performs the correct high-level sequence:

```text
construct and initialize DiT
-> load optional pretrained weights
-> convert selected nn.Linear modules with TorchAO
-> wrap for training
-> activation checkpointing
-> compile repeated blocks
-> FSDP2
-> optimizer
```

`float8_recipe=null` is a strict BF16 no-op. `float8_recipe=tensorwise` lazily
imports TorchAO and performs two disjoint conversions:

- 190 non-QKV Linears use dynamic tensor-wise FP8 for forward, dgrad, and
  wgrad;
- 38 image/text QKV Linears use dynamic tensor-wise FP8 for forward and dgrad,
  but disable both FP8 wgrad operand casts.

The implementation preserves the original `Parameter` objects and disables
FP8 FSDP all-gather. The existing validation demonstrates full-graph compile,
FSDP2, DTCP resume, finite 500-step behavior, and one full convergence seed.
These semantics are constraints for a kernel optimization, not areas to
redesign.

### 4.2 Runtime operation flow

For a converted Linear, TorchAO's `matmul_with_hp_or_float8_args` executes:

```text
forward:
  BF16 input  -> dynamic amax/scale/cast to E4M3
  BF16 weight -> dynamic amax/scale/cast to E4M3
  torch.mm(Float8TrainingTensor, Float8TrainingTensor)
  -> torch._scaled_mm
  -> BF16 output

backward dgrad:
  BF16 grad_output -> dynamic cast to E5M2
  BF16 weight      -> dynamic cast to E4M3
  scaled FP8 GEMM  -> BF16 grad_input

backward wgrad, non-QKV:
  BF16 grad_output -> dynamic cast to E5M2
  BF16 input       -> dynamic cast to E4M3
  scaled FP8 GEMM  -> high-precision parameter gradient

backward wgrad, QKV:
  BF16 grad_output and input remain high precision
  -> ordinary high-precision torch.mm
```

TorchAO stores the **quantization multiplier** in `Float8TrainingTensor._scale`
and passes its reciprocal to `torch._scaled_mm`. Primus-Turbo GEMM accepts the
inverse/dequantization scale. A bridge can therefore pass
`torchao_scale.reciprocal()` without changing the numerical definition, but
zero/NaN handling and scale shape must still be checked against TorchAO.

### 4.3 Real GEMM shapes

The six primary forward shapes recorded by the current plan are listed as
`(tokens, input_features, output_features)`, equivalent to GEMM `(M,K,N)`:

```text
(M, K, N)
(16384,  3072,  9216)
(16384,  3072,  3072)
(16384,  3072, 12288)
(16384, 12288,  3072)
(32768,  3072, 21504)
(32768, 15360,  3072)
```

For each forward `(M, K, N)`, qualification must also cover:

- dgrad GEMM `(M, N, K)`;
- wgrad GEMM `(N, M, K)` for non-QKV modules;
- the exact physical row/column-major strides produced by TorchAO;
- E4M3 x E4M3 forward and E5M2 x E4M3 hybrid backward operands.

Large token `M` becomes the contraction dimension of wgrad. A kernel that wins
only the forward NT GEMM is insufficient.

### 4.4 Current measured attribution

The latest pre-AITER optimized-candidate profile in `flux-fp8.md` reports the
following Primus-over-NeMo kernel-work deltas:

| Group | Delta |
|---|---:|
| Attention | `+108.4 ms/step` |
| Norm + pointwise | `+77.0 ms/step` |
| FP8 GEMM + cast/scale | `+86.2 ms/step` |
| BF16 GEMM + optimizer | `+46.5 ms/step` |

These are overlapping GPU event sums, not additive wall time. The later AITER
qualification removed attention as the first unaddressed target and reduced
the steady throughput gap to `14.2%`. The K0 results below reject a raw FP8
GEMM swap, leaving norm/pointwise fusion and host/GPU scheduling as the next
measured areas. A new profiler capture with AITER enabled is required before
assigning the residual gap more precisely.

## 5. ROCm kernel library assessment

### 5.1 Primus-Turbo

Relevant files include:

```text
primus_turbo/pytorch/ops/gemm_fp8.py
primus_turbo/pytorch/ops/quantization.py
primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py
primus_turbo/pytorch/kernels/quantization/quantization_impl.py
primus_turbo/pytorch/modules/linear_fp8.py
primus_turbo/flydsl/gemm/gemm_fp8_kernel.py
```

#### Capabilities that match FLUX

- tensor-wise dynamic FP8 quantization with scalar FP32 inverse scales;
- E4M3, E5M2, and hybrid E4M3/E5M2 GEMMs;
- BF16/FP16 output;
- forward, dgrad, and wgrad layouts;
- HipBLASLt, CK, Triton, and FlyDSL tensor-wise backends;
- a fake implementation for `primus_turbo::gemm_fp8_impl`, making the raw op a
  viable `torch.compile` boundary;
- shape/layout/dtype-aware dispatch and an autotune cache;
- gfx950 FlyDSL support for tensor-wise FP8 when `K > 128`, including native
  NT/NN/TN layouts.

All six FLUX dimensions satisfy the obvious gfx950 FlyDSL `K > 128` condition.
That does not establish support for TorchAO's actual strides or performance.

#### Current blockers to using `Float8Linear` directly

- The public `gemm_fp8()` function is decorated with
  `torch._dynamo.disable(recursive=True)` because its autograd path constructs
  `QuantizedTensor` wrappers. It is incompatible with the required
  `torch.compile(fullgraph=True)` block contract.
- `Float8Linear.forward()` calls that disabled function, so replacing TorchAO
  modules with Primus-Turbo `Float8Linear` would introduce a graph break.
- The default `Float8QuantConfig` is E4M3, not TorchAO's full hybrid training
  policy. `Format.HYBRID` must be selected explicitly for matching backward
  behavior.
- The high-level autograd function always computes FP8 wgrad; it does not model
  the production QKV high-precision-wgrad exception.
- It saves/creates transposed quantized operands on gfx950 to favor NT backward
  GEMMs. The copy and memory costs may be appropriate, but differ from the
  compile-proven TorchAO path and must be measured.
- Tensor-wise quantization performs reduction, scale computation, and cast via
  a C++ custom op with temporary `amax`, `scale`, `scale_inv`, and reduction
  workspace allocations. It is not automatically faster than Inductor's
  current implementation.

#### Assessment

**Best first candidate, but only below TorchAO's cast/autograd layer.** The
compile-friendly raw `gemm_fp8_impl` is the useful integration surface. The
high-level `Float8Linear` is not currently a valid drop-in replacement.

### 5.2 AITER

Relevant files include:

```text
aiter/ops/gemm_op_a8w8.py
aiter/configs/a8w8_tuned_gemm.csv
aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv
aiter/configs/a8w8_blockscale_tuned_gemm.csv
aiter/utility/dtypes.py
```

#### Useful capabilities

- CK, CKTile, ASM, Triton, and FlyDSL A8W8 GEMM implementations;
- shape-tuned CSV dispatch and split-K selection;
- weight-preshuffled dense FP8 kernels;
- fake tensor wrappers around several GEMM entry points;
- gfx942/gfx950 dtype selection and broad ROCm deployment experience.

#### Mismatch with the current training contract

- The primary `gemm_a8w8` surface is an inference-style raw GEMM, not a Linear
  training autograd implementation.
- The public dtype alias is architecture-selected E4M3. The inspected dense
  path does not present the same explicit E5M2 x E4M3 hybrid contract needed by
  TorchAO backward.
- Many optimized paths assume per-token/per-channel scales, block scales, or a
  preshuffled weight. Switching from tensor-wise dynamic scaling would be a new
  numerical recipe requiring convergence qualification.
- Weight preshuffle is attractive for inference but training weights change
  every optimizer step. Its conversion/invalidation cost must be included and
  it is especially risky with FSDP2 and activation recompute.
- The earlier FLUX AITER attention failure used the wrong backend name;
  `flash_attn_aiter` is now compile- and convergence-qualified. That does not
  change the dense A8W8 contract mismatch described above.

#### Assessment

**Benchmark-only in the first wave.** AITER may expose a winning CK/ASM kernel
for a particular E4M3 forward shape, but it should not drive production design
until hybrid backward, scale semantics, compile, startup, and all real layouts
are demonstrated.

### 5.3 FlyDSL

Relevant files include:

```text
kernels/gemm/fp8_gemm_4wave.py
kernels/gemm/fp8_gemm_8wave.py
kernels/gemm/fp8_gemm_utils.py
kernels/gemm/preshuffle_gemm.py
```

#### Useful capabilities

- gfx950 native FP8 MFMA kernels;
- 4-wave and 8-wave dense GEMM variants;
- explicit control of tiles, LDS staging, scheduling, and preshuffle layouts;
- E4M3 and E5M2 operand support in the Primus-Turbo wrapper;
- row/column scaling and bias/activation epilogues in the generic preshuffle
  family;
- a path for specializing the exact static FLUX shapes.

#### Integration risks

- Standalone APIs are compile/build-oriented and require explicit kernel
  compilation, stream plumbing, output allocation, and often B preshuffle.
- The generic preshuffle API applies per-row/per-column scales, which is not the
  same ABI as the current scalar tensor-wise recipe without an adapter.
- Direct integration would duplicate backend gating, fake/meta registration,
  fallback, caching, and packaging already present in Primus-Turbo.
- A hand-selected tile can regress dgrad/wgrad even if it wins forward.

#### Assessment

**Promising kernel implementation, wrong direct integration layer.** Consume
FlyDSL through Primus-Turbo's tensor-wise backend first. Add a new FlyDSL shape
specialization only after the existing backend is shown to be the limiting
factor.

## 6. Compatibility matrix

| Requirement | TorchAO current | Primus-Turbo raw op | AITER dense A8W8 | FlyDSL standalone |
|---|---|---|---|---|
| Dynamic tensor-wise scale | Yes | Yes | Not the main tuned contract | Adapter required for many kernels |
| E4M3 forward | Yes | Yes | Yes | Yes on gfx950 |
| E5M2 x E4M3 backward | Yes | Yes | Not established by inspected public dense API | Available through Primus-Turbo wrapper |
| Forward/dgrad/wgrad | Yes | Yes | Raw GEMM only | Raw GEMM only |
| QKV HP wgrad exception | Yes | Must remain in TorchAO policy | Caller-owned | Caller-owned |
| `fullgraph=True` | Validated | Raw custom op has fake impl; must qualify | Wrappers exist; JIT risk remains | Requires wrapper/fake/caching |
| FSDP2/checkpoint behavior | Validated | Unchanged if used below TorchAO | Unqualified | Unqualified |
| Shape-aware backend choice | PyTorch/HipBLASLt | Yes | Yes, tuned CSV | Explicit tile selection |
| Recommended first use | Baseline/control | Production candidate | Benchmark competitor | Via Primus-Turbo |

## 7. Recommended integration design

### 7.1 Preserve the TorchAO control plane

Keep unchanged:

- `convert_to_float8_training()` and current module filters;
- both `Float8LinearConfig` values;
- TorchAO's `matmul_with_hp_or_float8_args` autograd split;
- QKV high-precision wgrad;
- original parameters/state-dict keys;
- BF16 all-gather and FSDP2 policy;
- activation checkpointing and block compile boundaries;
- the public `float8_recipe=tensorwise` numerical meaning.

The optimized path must receive already-quantized FP8 operands plus their
scales. It must not own module conversion, parameters, optimizer hooks, or
checkpoint state.

### 7.2 Add one compile-safe raw GEMM bridge

The bridge contract should be equivalent to:

```text
fp8_gemm(
    a_fp8,
    b_fp8,
    a_inverse_scale,
    b_inverse_scale,
    out_dtype,
    logical layout/transpose metadata,
) -> output
```

Requirements:

- call a registered custom op with a fake/meta implementation;
- accept E4M3/E5M2 combinations and scalar inverse scales;
- map TorchAO's physical strides to NT/NN/TN without materializing an
  unnecessary transpose;
- return the same logical shape and BF16 dtype as `torch._scaled_mm`;
- fall back to `torch._scaled_mm` when dtype, layout, shape, architecture, or
  library availability is unsupported;
- avoid online compilation or tuning while a compiled training graph is
  executing;
- log the selected backend once per unique `(M,N,K,dtypes,layout)` key, not on
  every step.

Primus-Turbo's existing `primus_turbo::gemm_fp8_impl` is close to this ABI. It
should be promoted to a small supported raw API rather than importing private
implementation modules from Primus. The production bridge should not depend on
Primus-Turbo's graph-disabled `gemm_fp8()` wrapper.

### 7.3 Hook location

TorchAO ultimately routes FP8 `aten.mm` through
`torchao.float8.float8_ops.addmm_float8_unwrapped()` and then
`torch._scaled_mm`. The smallest prototype can install a ROCm-only replacement
for that raw function before any block is compiled.

That hook is private TorchAO API, so it is acceptable only as an experiment.
For production, use one of these in priority order:

1. an upstream/public TorchAO scaled-GEMM backend hook;
2. a small patch in the pinned TorchAO image, with a version assertion;
3. a narrowly scoped Primus installation shim with an exact TorchAO commit
   gate and immediate fallback.

Do not copy TorchAO's full autograd function into Primus. That would duplicate
roughly all forward/dgrad/wgrad casting logic and create a second numerical
implementation to maintain.

### 7.4 Backend policy

The initial backend policy should be conservative:

```text
if ROCm gfx950, tensor-wise scalar scales, supported dtypes/layout, known shape:
    use pre-qualified Primus-Turbo backend for the exact pass/shape
else:
    use torch._scaled_mm
```

Candidate ordering is determined by benchmark, not hardcoded reputation:

```text
FlyDSL vs CK vs Triton vs HipBLASLt/TorchAO control
```

Primus-Turbo global environment selection is useful for isolated screening,
but production should use a precomputed shape table or a persisted dispatcher
cache. Runtime autotune is inappropriate inside a full-graph training run and
Primus-Turbo already suppresses autotuning during graph capture.

### 7.5 Do not change quantization in phase 1

Keep TorchAO's existing dynamic cast. Screen the raw GEMM replacement both:

- as GEMM-only latency with pre-quantized operands;
- end-to-end with TorchAO amax/scale/cast included.

Only if GEMM wins but cast/scale remains dominant should a second phase compare:

- TorchAO/Inductor dynamic cast;
- Primus-Turbo tensor-wise quantization;
- a fused reduction + scale + cast kernel;
- optional generation of both physical orientations in one pass;
- reuse of a quantized weight within forward/backward/recompute while the
  high-precision weight version is unchanged.

Weight reuse is potentially valuable because a training weight is unchanged
between forward and its backward, and activation recompute may repeat the
forward. It is also substantially riskier: cache invalidation after optimizer
step, FSDP parameter versioning, checkpointing, memory, and compile capture all
need explicit handling. It is not part of the first kernel bridge.

## 8. Benchmark and qualification plan

### Phase K0: reproducible microbenchmark

Pin the exact training image and record:

```text
PyTorch and ROCm commits
TorchAO commit
Primus-Turbo/AITER/FlyDSL commits
CK and HipBLASLt versions
GPU architecture and clock/power settings
compile mode
```

For all six forward shapes, benchmark forward, dgrad, and applicable wgrad.
Compare:

- current TorchAO `torch._scaled_mm`;
- Primus-Turbo HipBLASLt;
- Primus-Turbo CK;
- Primus-Turbo Triton;
- Primus-Turbo FlyDSL;
- Primus-Turbo autotune after an explicit pre-tuning stage;
- AITER only where the exact dtype/scale/layout contract can be represented.

Report both:

1. pre-quantized GEMM latency and TFLOP/s;
2. complete dynamic cast + scale + GEMM latency.

Warmup must exclude one-time JIT/library initialization. Results must include
median and tail latency, selected kernel name, output SNR/error, and allocation
count. A backend is invalid if it silently changes the scale granularity or FP8
format.

#### Completed K0 result

K0 ran on one MI355X using
`zirui3/mlperf-rocm:v0.1-flydsl-v0.2.3`, FlyDSL `0.2.3`, and Primus-Turbo
`56c789e5` built for `gfx950`. A fresh container successfully loaded the built
Primus-Turbo through a read-only source mount and exposed HipBLASLt, CK, Triton,
and FlyDSL without modifying the image. The benchmark used already-quantized
operands, scalar inverse scales, E4M3 forward operands, E5M2 gradient operands,
and the native NT/NN/TN layouts.

The corrected 10-iteration screen produced these summed raw latencies across
all six shapes and three passes:

| Backend | Sum over 18 cases | Relative to `_scaled_mm` | Decision |
|---|---:|---:|---|
| TorchAO/TorchTitan `_scaled_mm` | `11.886 ms` | control | Keep |
| Primus-Turbo HipBLASLt | `18.539 ms` | `+56.0%` | Reject |
| Primus-Turbo CK | `137.972 ms` | `+1060.7%` | Reject |
| Primus-Turbo Triton | `16.072 ms` | `+35.2%` | Reject |
| Primus-Turbo FlyDSL | `12.539 ms` | `+5.5%` | Reject |

All FlyDSL outputs were finite and matched the `_scaled_mm` control with zero
reported relative L2 error in this screen. The large fused single-stream shapes
were slower with FlyDSL: `(32768,3072,21504)` regressed `14.1%/4.5%/6.9%` for
forward/dgrad/wgrad, and `(32768,15360,3072)` regressed
`0.3%/6.3%/7.6%`.

The only apparent short-screen candidates were retested with 10 warmups and
100 timed iterations. QKV forward/dgrad changed from apparent `11-13%` wins to
`0.6%/0.3%` regressions; MLP-up forward changed from an apparent `11.4%` win to
a `1.7%` regression. No pass retained a stable win.

Artifacts:

```text
/shared_nfs/zirui/runs/primus_turbo_flydsl_build_20260803/
/shared_nfs/zirui/runs/flux_fp8_gemm_flydsl_corrected_full_20260803T053501Z/
/shared_nfs/zirui/runs/flux_fp8_gemm_flydsl_candidates_20260803T054443Z/
```

K1-K3 are therefore not entered for this implementation. They remain the
qualification procedure for a future kernel that first passes K0.

### Phase K1: compile-only integration gate

With the raw bridge enabled:

- compile one real-shape Linear forward/backward with `fullgraph=True`;
- compile one double-stream and one single-stream block;
- verify zero graph breaks and no unexpected recompiles;
- inspect the FX/Inductor graph to confirm the custom op is present;
- confirm the intended CK/FlyDSL kernel in the GPU trace;
- ensure unsupported cases execute the `_scaled_mm` fallback;
- ensure BF16 mode never imports the optional kernel library.

A fake implementation alone does not prove capture safety. The runtime op must
also avoid JIT compilation, Python-side tensor inspection that Dynamo traces,
and synchronization on every invocation.

### Phase K2: numerical gate

For every pass and shape:

- compare against BF16 and the current TorchAO path;
- test zero, near-zero, large finite values, NaN/Inf propagation, and
  non-contiguous/transpose layouts;
- verify E4M3 forward and E5M2 gradient-output encoding on gfx950;
- use SNR/cosine/error gates suitable for FP8, not loose element-wise
  `allclose` alone;
- verify QKV wgrad remains high precision and byte-for-byte follows the control
  operation graph where practical.

Then run the existing tiny optimizer, activation checkpoint, and FSDP2 tests.
The bridge must not add scale/cache tensors to state dicts or DTCP.

### Phase K3: model-level performance funnel

Use the same production settings as the qualified control:

1. 100-step 8-GPU A/B, fixed seed;
2. 500-step A/B with production warmup behavior;
3. five-step exact profiler capture after steady state;
4. one full convergence seed only if the short runs pass;
5. remaining seeds only if time-to-quality improves.

Measure:

- steady-state step time and global samples/s;
- FP8 GEMM and cast/scale time separately;
- attention and norm/pointwise time to detect shifted bottlenecks;
- peak memory and allocation count;
- compile/startup time;
- validation-inclusive time-to-quality.

### Acceptance criteria

A production kernel backend must satisfy all of the following:

- no graph break or additional steady-state recompilation;
- no numerical-policy change and no new non-finite behavior;
- all existing FLUX FP8 unit/FSDP2/DTCP tests pass;
- at least `3%` improvement in aggregate affected Linear forward/backward time;
- at least `1.5%` end-to-end steady-state step-time improvement;
- no material convergence-step or validation-inclusive time-to-quality
  regression;
- deterministic fallback on unsupported library versions or shapes.

A smaller gain should not justify a new runtime dependency or private TorchAO
patch.

## 9. Expected opportunities by priority

| Priority | Opportunity | Expected value | Risk |
|---:|---|---|---|
| P0 | Re-profile the AITER winner to refresh residual attribution | High; current detailed profile predates the attention change | Low |
| P1 | Fuse measured norm/pointwise chains inside compiled blocks | Medium; largest unaddressed pre-AITER GPU delta | Medium |
| Deferred | Shape/pass-specific raw GEMM selection through Primus-Turbo | No current winner; all tested backends regress in stable K0 | Low-medium |
| Deferred | Tune/add gfx950 FlyDSL kernels for the six FLUX shape families | Reconsider only after a new kernel beats K0 by at least 3% | Medium |
| Deferred | Avoid unnecessary layout copies/transposes in dgrad/wgrad | Profile first; raw native-layout kernels did not win | Medium |
| Deferred | Fuse or reduce launches in dynamic tensor-wise quantization | Current matched profile did not identify cast/scale as primary | Medium-high |
| P4 | Reuse quantized weights across forward/backward/recompute | Potentially high | High: invalidation/FSDP/memory/compile |
| P5 | Preshuffled training weights | Unknown | High; optimizer updates every step |
| P6 | Change to row-wise/block-wise/MXFP8 | Separate accuracy project | Very high; changes recipe |

The fused native single-stream shapes `(32768,3072,21504)` and
`(32768,15360,3072)` were the initial specialization candidates, but the
current FlyDSL implementation regressed every pass except a noise-level `0.3%`
forward result. They are no longer an integration target without new kernel
tiling or scheduling work.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Primus-Turbo direct `Float8Linear` breaks fullgraph | Use only the raw custom op below TorchAO; keep the high-level module out of production |
| Scale convention is inverted | Define bridge inputs as inverse/dequant scales and test against TorchAO `_scaled_mm` |
| Physical strides do not match transpose flags | Include stride/layout in the dispatch key and qualification matrix; fallback rather than copy by default |
| Backend wins forward but loses backward | Select by exact pass/shape/layout; accept only aggregate forward+backward gains |
| E5M2 hybrid path falls back or is unsupported | Assert operand dtypes and profiler kernel; retain `_scaled_mm` fallback |
| Runtime autotune disrupts compile/training | Tune offline or during explicit prewarm; persist the selected table |
| Private TorchAO hook drifts | Pin/assert TorchAO commit and prefer an upstream backend hook |
| AITER JIT/import repeats the prior failure | Keep AITER out of production until import and fullgraph smoke pass independently |
| Opaque custom op blocks useful Inductor fusion | Compare complete cast+GEMM, not GEMM alone; keep TorchAO control |
| Weight cache uses stale parameters | Defer cache work; if attempted, key invalidation to optimizer/FSDP parameter version |
| Faster steady state changes convergence outcome | Preserve exact recipe and run the existing fixed-seed/full-convergence funnel |
| Added complexity exceeds the gain | Require the end-to-end threshold and one narrow bridge; do not add a generic FLUX kernel framework |

## 11. Implementation sequence

K0 identified no winning backend, so no production bridge was implemented.
If a future kernel passes K0, use this order:

```text
1. expose/support Primus-Turbo raw tensor-wise FP8 GEMM API
2. add isolated correctness and compile tests for that raw op
3. add one ROCm-only TorchAO raw-GEMM bridge behind an experimental gate
4. populate an offline-qualified shape/pass backend table
5. run block compile and 8-GPU short A/B tests
6. make it the tensorwise default only after fallback and convergence gates
7. investigate quantization/layout reuse only if profiling still justifies it
```

The intended production diff is deliberately small: one optional raw GEMM
bridge and its tests. There should be no new Linear class, generic converter,
checkpoint state, or user-facing scaling recipe in the first optimization.

## 12. Final recommendation

Keep the current TorchAO dynamic tensor-wise implementation and its
`torch._scaled_mm` GEMM. Do not add a Primus-Turbo bridge, direct
`Float8Linear`, AITER A8W8 module, or standalone FlyDSL kernel: none passed the
raw performance gate, and several also retain fullgraph or numerical-contract
risks.

AITER attention is now part of the qualified performance candidate. The next
step is a new profile of that exact winner, followed by narrowly targeted
norm/pointwise or scheduling work. Reopen FP8 GEMM replacement only when a new
HipBLASLt/CK/FlyDSL kernel demonstrates a repeatable microbenchmark win of at
least `3%` on the actual pass/layout mix.
