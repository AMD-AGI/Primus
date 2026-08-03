# Primus Native Dynamic Tensor-wise FP8 FLUX Training Plan

## 1. Purpose

This document defines the implementation and validation plan for native FP8
FLUX training in the Primus diffusion backend.

The first delivery targets the existing FLUX.1-Schnell MLPerf recipe on one
node with eight MI355X GPUs. The goal is to keep the implementation small,
retain the current native PyTorch model and distributed stack, achieve training
quality as close to the validated BF16 baseline as practical, and improve
end-to-end time-to-quality.

The first implementation uses TorchAO dynamic tensor-wise FP8 for forward and
dgrad. Wgrad also uses FP8 except for the image and text QKV projections, which
remain high precision for numerical stability. Exact Transformer Engine (TE)
or NeMo delayed-scaling behavior is not required.

The follow-on ROCm kernel assessment and measurement-first optimization design
are documented in [`fp8-kernel-opt.md`](fp8-kernel-opt.md). That design keeps
this TorchAO numerical and integration policy unchanged and evaluates replacing
only the raw scaled FP8 GEMM with qualified Primus-Turbo CK/FlyDSL kernels.

## 2. Background

The native Primus BF16 path is already a validated baseline:

- ordinary `nn.Linear` and `nn.RMSNorm` modules;
- FSDP2 with FP32 parameter storage, BF16 forward parameters, and FP32
  gradient reduction;
- activation checkpoint ratio `0.25`;
- `reshard_after_forward=false`;
- per-block `torch.compile(fullgraph=True)`;
- `1.6445 s/step` and approximately `314.4 samples/s` on 8 x MI355X;
- peak memory of `206.61 GiB`;
- three independent seeds reaching `validation_loss <= 0.586`.

The full BF16 analysis and evidence are in `primus-nemo-perf-gap.md`. That
baseline remains frozen and is the regression reference for all FP8 work.

The native FP8 work must preserve the main reason the BF16 path is fast:
full-graph compilation of each repeated FLUX transformer block. A numerically
correct FP8 implementation that introduces graph breaks or splits the compiled
regions is not acceptable.

## 3. Scope

### 3.1 First delivery

The first delivery supports:

- FLUX.1-Schnell;
- the current MLPerf preprocessed-data training path;
- TorchAO dynamic/current tensor-wise scaling;
- block Linear FP8 forward and dgrad GEMMs, with FP8 wgrad except for QKV;
- BF16 attention, norms, RoPE, pointwise operations, and FSDP communication;
- FP32 loss, optimizer master state, and gradient reduction;
- the existing selective activation checkpointing, block compile, FSDP2, and
  DTCP paths.

### 3.2 Non-goals for the first delivery

The first delivery does not include:

- TE modules or TE FP8 autocast;
- delayed scaling or persistent amax history;
- exact NeMo step-by-step numerical equivalence;
- FP8 parameter storage;
- FP8 FSDP all-gather;
- BF16 gradient reduction;
- row-wise FP8 as a production option;
- MXFP8;
- FP8 attention;
- changes to SDPA, RMSNorm, RoPE, AdamW, or the FLUX block structure;
- splitting the fused native single-stream Linear modules;
- a generic quantization/converter framework for all Primus models;
- Flux Dev convergence qualification.

These items may be evaluated later, but must not complicate the first
tensor-wise implementation.

## 4. Reference Implementations

### 4.1 TorchAO

The public API used by this plan is:

```python
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
```

TorchAO `Float8LinearConfig()` currently represents dynamic tensor-wise
scaling. Its default hybrid data types are:

| Operand | FP8 format |
|---|---|
| Forward input | E4M3 |
| Forward weight | E4M3 |
| Backward grad output | E5M2 |

The FP8 operands are transient. The trainable parameters and resulting
gradients remain high precision. Primus uses FP8 wgrad for 190 block Linear
modules and disables it for the 38 image/text QKV modules because selective
screening traced the all-FP8 non-finite gradients to QKV wgrad.

Relevant source:

- `/shared_nfs/zirui/code/ao/torchao/float8/config.py`
- `/shared_nfs/zirui/code/ao/torchao/float8/float8_linear.py`
- `/shared_nfs/zirui/code/ao/torchao/float8/float8_linear_utils.py`

### 4.2 Diffusers TorchAO FLUX example

The external `diffusers-torchao` example demonstrates a small integration:

<https://github.com/sayakpaul/diffusers-torchao/tree/main/training>

It converts the transformer with `convert_to_float8_training()`, excludes the
final projection, and leaves the surrounding training precision at BF16.

It is useful as API and ordering evidence, but it is not a full-parameter
training reference. The example is DreamBooth LoRA training with a frozen base
transformer. It does not demonstrate TorchAO FP8 with block compile, FSDP2,
full-model optimizer state, BF16-versus-FP8 convergence, or distributed
throughput.

### 4.3 TorchTitan

TorchTitan provides the closest generic integration reference:

- `torchtitan-main/torchtitan/components/quantization/float8.py`
- `torchtitan-main/torchtitan/models/flux/parallelize.py`

It demonstrates conversion before compile, FSDP, and optimizer construction.
It also handles TorchTitan-specific module protocols and meta-device
initialization. Those abstractions should not be copied into the Primus
diffusion backend.

TorchTitan is a reference and optional diagnostic harness, not an intermediate
implementation target. The production implementation should be built directly
in Primus because the important compatibility risks are specific to the Primus
block layout, activation checkpointing, full-graph compile, FSDP2 policy, and
DTCP lifecycle.

## 5. Relationship to NeMo FP8

The NeMo MLPerf configurations use BF16 mixed precision with selected TE Linear
GEMMs in FP8. Depending on the system configuration, the active recipe is
typically one of:

```yaml
# delayed.yaml
fp8: hybrid
fp8_recipe: delayed
fp8_margin: 0
fp8_amax_history_len: 1024
fp8_amax_compute_algo: max
fp8_params: false
```

```yaml
# delayed_short.yaml
fp8: hybrid
fp8_recipe: delayed
fp8_margin: 0
fp8_amax_history_len: 4
fp8_amax_compute_algo: most_recent
fp8_params: false
```

The proposed native implementation aligns with NeMo in the following ways:

- BF16 is the surrounding model/activation precision;
- forward activations and weights use E4M3 inside FP8 GEMMs;
- backward grad output uses E5M2;
- the major transformer block Linear GEMMs use FP8;
- attention remains BF16;
- input, conditioning, modulation, final projection, and loss stay higher
  precision;
- primary model parameters are not stored in FP8;
- optimizer master state remains FP32.

The proposed implementation intentionally differs in the following ways:

| Area | NeMo/TE | Primus first delivery |
|---|---|---|
| Scaling time | Delayed | Current/dynamic |
| Amax history | 4 or 1024 entries | None |
| Amax synchronization | TE process group reduction | No TE global amax state |
| FP8 control | TE autocast scope | Per-module TorchAO conversion |
| Distributed model | Replicated DDP + distributed optimizer | FSDP2 |
| Gradient reduction | BF16 | FP32 |
| Activation recompute | Disabled in current NeMo recipe | Ratio `0.25` |
| Single block | Four independent TE Linears | Two fused native Linears |
| Wgrad | FP8 | FP8 except image/text QKV |

The native `SingleStreamBlock` fuses QKV with MLP-up in `linear1` and fuses
attention-output with MLP-down in `linear2`. A tensor-wise scale therefore
covers a larger fused tensor than in NeMo. Exact scale-level parity would
require splitting these modules, changing GEMM count, checkpoint mapping,
compile graphs, and the validated BF16 architecture. That is not justified for
the first implementation.

The acceptance target is therefore BF16-like convergence and improved
time-to-quality, not exact NeMo internal state equivalence.

## 6. Precision Policy

The first production policy is:

| Component | Precision |
|---|---|
| Trainable parameter storage before FSDP | FP32 |
| FSDP forward parameter | BF16 |
| Selected Linear forward GEMM | FP8 E4M3 x E4M3 |
| Selected Linear dgrad GEMM | FP8 using E5M2 grad output |
| Non-QKV block Linear wgrad GEMM | FP8 tensor-wise |
| Image/text QKV wgrad GEMM | BF16/high precision |
| Linear output and parameter gradient | BF16/high precision |
| Attention, norms, RoPE, GELU, residuals | BF16/compiled native behavior |
| FSDP all-gather | BF16 |
| Gradient reduction | FP32 |
| MSE loss | FP32 |
| AdamW master state and moments | FP32 |

The policy uses two disjoint conversion calls. Non-QKV block Linears use the
standard dynamic tensor-wise configuration:

```python
Float8LinearConfig(
    pad_inner_dim=False,
    enable_fsdp_float8_all_gather=False,
)
```

QKV Linears explicitly disable both wgrad operand casts rather than relying on
a locally patched default:

```python
from torchao.float8 import CastConfig, ScalingType

Float8LinearConfig(
    cast_config_input_for_grad_weight=CastConfig(
        scaling_type=ScalingType.DISABLED,
    ),
    cast_config_grad_output_for_grad_weight=CastConfig(
        scaling_type=ScalingType.DISABLED,
    ),
    pad_inner_dim=False,
    enable_fsdp_float8_all_gather=False,
)
```

Both configurations keep dynamic tensor-wise FP8 for forward and dgrad. The
first also applies it to wgrad, while the second computes QKV wgrad in high
precision. All selected Schnell block dimensions are divisible by 16, so
inner-dimension padding is unnecessary. Explicitly disabling it also avoids
behavior changes between the upstream TorchAO checkout and Primus images that
patch the default to `True`.

Emulation must be disabled in performance or convergence runs.

## 7. Module Coverage

### 7.1 Included modules

The first policy converts only the repeated token-wise block GEMMs.

Double-stream blocks:

```text
double_blocks.*.img_attn.qkv
double_blocks.*.img_attn.proj
double_blocks.*.txt_attn.qkv
double_blocks.*.txt_attn.proj
double_blocks.*.img_mlp.0
double_blocks.*.img_mlp.2
double_blocks.*.txt_mlp.0
double_blocks.*.txt_mlp.2
```

Single-stream blocks:

```text
single_blocks.*.linear1
single_blocks.*.linear2
```

For the standard Schnell architecture this is:

| Region | Count |
|---|---:|
| 19 double blocks x 8 Linear modules | 152 |
| 38 single blocks x 2 Linear modules | 76 |
| Total | 228 |

The wgrad split is:

| Policy | Modules | Count |
|---|---|---:|
| High-precision wgrad | `double_blocks.*.{img_attn,txt_attn}.qkv` | 38 |
| FP8 wgrad | All other included block Linears | 190 |

The implementation should derive the expected candidate count from the actual
number of double and single blocks, log all converted FQNs at debug level, and
fail if FP8 is requested but no candidate is found. Tests for the standard
Schnell preset should assert the exact count of 228.

### 7.2 Excluded modules

The first policy excludes:

```text
img_in
txt_in
time_in
vector_in
guidance_in
double_blocks.*.img_mod.*
double_blocks.*.txt_mod.*
single_blocks.*.modulation.*
final_layer.*
T5 encoder
CLIP encoder
VAE
```

The modulation and conditioning layers have a small runtime `M`, so dynamic
amax and cast overhead may cost more than the underlying GEMM saves. The final
projection has a small output width and is zero-initialized. Keeping these
paths in BF16 also follows the main NeMo coverage boundary and the external
Diffusers example's decision to exclude `proj_out`.

The first implementation must use a fixed Flux-specific policy. It should not
expose arbitrary include/exclude patterns in the user configuration.

## 8. Integration Design

### 8.1 Configuration

Add one optional model configuration value:

```yaml
model:
  config:
    float8_recipe: null
```

Supported first-delivery values:

```text
null
tensorwise
```

`null` retains the current BF16 path and must not import TorchAO. `tensorwise`
selects the fixed selective-wgrad policy described above.

The local MLPerf launcher may expose an environment override such as
`FLUX_FLOAT8_RECIPE`, but the normalized model configuration remains the source
of truth recorded in run metadata.

Do not introduce a separate backend selector, scaling-history configuration,
or arbitrary module filter in the first delivery.

### 8.2 Conversion boundary

Keep the optional TorchAO integration in the existing Flux model registration:

```text
primus/backends/diffusion/models/registrations/flux.py
```

The conversion block is limited to:

1. validate the requested recipe;
2. lazily import the public TorchAO APIs;
3. identify the fixed block Linear candidates by FQN;
4. create the two explicit tensor-wise `Float8LinearConfig` values;
5. call `convert_to_float8_training()` for the two disjoint module sets;
6. log the recipe, candidate count, Torch/PyTorch/ROCm/TorchAO versions, and
   selected FQNs;
7. fail early when the requested native FP8 path is unavailable.

It must not introduce a converter class or own optimizer hooks, distributed
state, delayed scale state, checkpoint code, or general quantization
registration.

### 8.3 Conversion timing

The required construction order is:

```text
construct and initialize DiT
-> load optional pretrained DiT weights
-> convert selected Linear modules to TorchAO FP8 training
-> construct FluxForTraining
-> freeze non-trainable modules
-> apply activation checkpoint wrappers
-> compile repeated blocks with fullgraph=True
-> apply FSDP2
-> construct optimizer
-> load DTCP training state when requested
```

The conversion call belongs immediately after optional pretrained loading in:

```text
primus/backends/diffusion/models/registrations/flux.py
```

It must run before block compilation, FSDP2 wrapping, and optimizer creation.

With FP8 all-gather disabled, TorchAO reuses the original weight and bias
`Parameter` objects. The expected model state-dict keys and high-precision
checkpoint values therefore remain unchanged. This property must be tested,
not assumed.

### 8.4 Trainer behavior

Dynamic tensor-wise scaling is stateless between iterations. The first
implementation should require no changes to:

- the forward/loss function;
- the training step;
- gradient accumulation;
- AdamW;
- FSDP mixed-precision policy;
- gradient clipping;
- activation checkpoint lifecycle;
- DTCP state format.

Activation recompute repeats the same current-scaling calculation during the
recomputed forward. There is no mutable delayed history to update twice.

## 9. Environment and Dependency Gate

TorchAO APIs and Inductor behavior are version-sensitive. Before model work,
record and pin:

```text
Primus image digest
PyTorch version and commit
ROCm version
TorchAO version and commit
Triton/Inductor version
GPU architecture
```

The repository's installation path currently pins TorchAO and applies ROCm
patches, while `/shared_nfs/zirui/code/ao` may be at a different revision. A
result obtained from one revision must not be treated as evidence for another.

FP8 qualification must fail clearly when:

- TorchAO is absent;
- the required public API is unavailable;
- the device is not a supported ROCm FP8 target;
- the selected FP8 dtype is not the expected MI355X OCP format;
- native scaled FP8 GEMM cannot execute.

The runtime imports TorchAO lazily and reports missing APIs immediately. Device
and kernel capability are qualified by the single-GPU compiled smoke and
profiler gate rather than by adding a probe GEMM to every model startup.

BF16 runs must remain independent of TorchAO availability.

## 10. Implementation Phases

### Phase 0: Freeze the BF16 reference

Use the existing Schnell configuration and three-seed convergence result as
the immutable comparison point. Do not combine FP8 work with attention,
reduction-dtype, optimizer, data-loader, or checkpoint-policy changes.

### Phase 1: MI355X kernel and compiler gate

Before model conversion, test the six primary FLUX GEMM shapes:

```text
(16384, 3072,  9216)
(16384, 3072,  3072)
(16384, 3072, 12288)
(16384,12288,  3072)
(32768, 3072, 21504)
(32768,15360,  3072)
```

For each shape measure:

- forward;
- dgrad;
- wgrad;
- eager execution;
- `torch.compile(fullgraph=True)` execution;
- output and gradient finiteness;
- error relative to BF16;
- complete cast, amax, scale, transpose, and GEMM cost;
- actual profiler kernel names.

The gate fails if execution uses emulation, silently falls back to BF16, cannot
compile without graph breaks, or regresses the relevant compiled BF16 path.

As an experimental accuracy/performance comparison, also probe TorchAO's
`rowwise_with_gw_hp` named recipe. It is not a first-delivery production option.
Current ROCm row-wise support and test coverage must be verified on MI355X
rather than inferred from NVIDIA results.

### Phase 2: Minimal Primus integration

Implement:

- the single `float8_recipe` configuration;
- the Flux registration conversion block;
- conversion after weight loading;
- the fixed 228-module Schnell policy;
- configuration/version logging;
- unit tests for conversion and BF16 no-op behavior.

Do not change the trainer unless a verified incompatibility requires it.

### Phase 3: Compile and FSDP2 qualification

Qualify, in order:

1. one real-shape `Float8Linear` forward/backward;
2. one double block in eager mode;
3. one double block with full-graph compile;
4. one single block in eager mode;
5. one single block with full-graph compile;
6. a tiny FLUX optimizer step;
7. two-GPU FSDP2;
8. eight-GPU Schnell smoke.

The current single-GPU FSDP2 trainer returns before applying block compile.
Single-GPU compile tests must therefore compile the block directly instead of
assuming a one-GPU trainer smoke covers the production graph.

### Phase 4: Checkpoint qualification

Verify:

- identical state-dict keys before and after conversion;
- high-precision stored parameter values;
- full DTCP save and resume;
- optimizer and scheduler restoration;
- RNG and data-consumption restoration;
- next-step loss after resume;
- `dit_only` safetensors export, or explicitly mark it unsupported until fixed.

No dynamic scale or amax tensor should appear in the first-delivery checkpoint.

### Phase 5: Performance and convergence funnel

Use the following funnel to avoid spending full convergence runs on an invalid
recipe:

1. 100-step 8-GPU smoke for finiteness, graph stability, and memory;
2. 500-step fixed-seed BF16/FP8 loss-curve comparison;
3. profiler capture for block kernels and end-to-end step breakdown;
4. one complete FP8 convergence run;
5. three independent FP8 convergence runs for the winning policy;
6. BF16-versus-FP8 time-to-quality comparison.

Only the final winning configuration should receive the three-seed run.

### Phase 6: Performance uplift matrix

Use the same image, seed, MBS 64, checkpoint ratio `0.25`, block compile,
no-reshard policy, and production LR warmup for every row:

| Experiment | Forward/dgrad | Wgrad | FSDP all-gather | Purpose |
|---|---|---|---|---|
| A0 | BF16 | BF16 | BF16 | Frozen baseline |
| A1 | FP8 tensor-wise | High precision | BF16 | Conservative control |
| A1.5 | FP8 tensor-wise | High precision | FP8 | Isolate FP8 all-gather |
| A2 | FP8 tensor-wise | FP8 tensor-wise | BF16 | Isolate FP8 wgrad |
| A3 | FP8 tensor-wise | FP8 tensor-wise | FP8 | Match TorchTitan tensor-wise scope |

Selective-wgrad screening temporarily evaluated fixed module groups. The
winning `all_except_qkv` policy is now the sole `tensorwise` behavior; the
experiment switch and group enumeration have been removed from production
configuration.

Run A2 before adding FP8 all-gather. The earlier all-FP8 NaN result used
`warmup_steps=0`; it does not establish instability under the production
`warmup_steps=1600` schedule. A2 must pass, in order:

1. 100 steps with the shortened matching warmup used by the BF16 control;
2. 500 steps with production warmup behavior;
3. gradient-finiteness logging before clipping;
4. fixed-seed loss comparison against A0 and A1.

Only after A2 is stable should A3 add FP8 FSDP all-gather. A3 must additionally
verify:

- the TorchAO weight tensor subclass survives FSDP2 sharding;
- one batched scale precompute runs after each optimizer step;
- DTCP save/resume preserves high-precision model and optimizer state;
- no extra per-weight scale all-reduces remain on the critical path;
- end-to-end throughput improves beyond A2.

Use the TorchAO Llama3 benchmark as context, not as an acceptance value. Its
H100 `25.03%` tensor-wise result includes all-FP8 wgrad and FP8 all-gather. The
more relevant published AMD MI300X result is `14.68%`; A1 already measured
approximately `11%` throughput improvement with high-precision wgrad and BF16
all-gather.

## 11. Test Plan

### 11.1 CPU/unit tests

- `float8_recipe=null` is a strict no-op.
- BF16 mode does not import TorchAO.
- unsupported recipe values fail with a clear error.
- candidate FQN matching includes only the intended block modules.
- the standard Schnell preset has 228 candidates.
- input, conditioning, modulation, and final modules remain `nn.Linear`.
- conversion happens after pretrained weight loading.
- parameter values and state-dict keys are unchanged by conversion.
- FP8 metadata is recorded in the normalized run configuration.

### 11.2 Single-GPU MI355X tests

- native FP8 forward/dgrad for every real shape;
- native FP8 wgrad for non-QKV shapes and high-precision QKV wgrad;
- zero and near-zero input handling;
- finite output and gradients;
- eager versus compiled numerical comparison;
- double-block full-graph forward/backward;
- single-block full-graph forward/backward;
- tiny FLUX optimizer step;
- activation checkpoint recomputation;
- profiler confirmation of native ROCm FP8 kernels.

### 11.3 Distributed tests

- two-GPU FSDP2 with BF16 all-gather and FP8 compute;
- eight-GPU Schnell smoke;
- checkpoint ratio `0` and `0.25`;
- the production no-reshard policy;
- DTCP save/resume;
- no graph breaks or unexpected recompilation;
- no tensor-subclass leakage into checkpoint or optimizer state.

## 12. Accuracy Strategy

"As close to BF16 as practical" is evaluated against the existing BF16
three-seed envelope, not by requiring bitwise equality.

The first tensor-wise policy is accepted when:

- short fixed-seed training does not show a systematic loss divergence;
- validation behavior remains within normal BF16 run-to-run variation;
- all three full runs reach `validation_loss <= 0.586`;
- convergence steps do not show a material systematic regression;
- FP8 improves time-to-quality after validation cost is included.

Track, but do not overfit to, per-block output error, gradient error, cosine
similarity, and scale distributions. End-to-end validation loss and
time-to-quality are the final criteria.

### 12.1 Accuracy fallback order

If dynamic tensor-wise FP8 does not meet the accuracy target:

1. verify that the issue is not an emulation/fallback kernel, overflow,
   incorrect dtype, compile, or FSDP problem;
2. evaluate TorchAO `rowwise_with_gw_hp` with BF16 communication;
3. use per-module measurements to reduce FP8 coverage while keeping the fixed
   policy simple;
4. evaluate native delayed scaling only after simpler online-scaling options
   have failed.

Do not introduce several fallbacks in the first implementation. Each fallback
must be justified by measured convergence or kernel evidence.

## 13. Performance Strategy

FP8 peak throughput is not the expected end-to-end speedup. FLUX training also
contains attention, normalization, RoPE, pointwise operations, activation
recompute, FSDP communication, optimizer work, data loading, and validation.

Measure:

- median steady-state step time;
- samples per second;
- forward, backward, and optimizer breakdown;
- cast/amax/scale/transpose time;
- recompute overhead;
- peak HBM;
- compile time and cache behavior;
- validation-inclusive time-to-quality.

The existing performance acceptance thresholds remain useful minimum gates:

- at least 3% improvement for an isolated block-level replacement;
- at least 1.5% end-to-end steady-state improvement;
- no loss of convergence or material time-to-quality regression.

Profiler evidence must show that the intended native ROCm FP8 kernels execute.
Configuration flags alone are not evidence.

## 14. Follow-up Work

Follow-up work is considered only after dynamic tensor-wise FP8 is qualified.

### 14.1 FP8 FSDP all-gather

Evaluate separately because it introduces a weight tensor subclass, scale
precomputation, a post-optimizer hook, and additional checkpoint risk. The
current no-reshard Schnell policy may also limit the benefit relative to a
resharding workload.

### 14.2 Delayed tensor-wise scaling

If online scaling cannot meet accuracy and performance requirements, the
Megatron local FP8 implementation provides a reference:

```text
primus/backends/megatron/core/extensions/primus_turbo_float8_local.py
```

It includes dynamic and delayed tensor-wise autograd paths, independent scale
tracks, amax histories, and fused updates. It is coupled to Megatron parallel
Linear modules and must not be imported directly into the native diffusion
backend.

A native delayed implementation would additionally need to define:

- scale update timing under activation recompute;
- data-parallel amax synchronization;
- warmup reset behavior;
- checkpoint/resume state;
- semantics for the fused native single-stream Linears.

### 14.3 MXFP8

MXFP8 is a separate project. The current TorchAO dense MXFP8 training path
hardcodes a CUDA dim-1 cast choice in the inspected revision, so MI355X support
requires a verified ROCm cast/kernel path before FLUX integration. MXFP8 must
not share the first tensor-wise acceptance matrix.

## 15. Risks

| Risk | Mitigation |
|---|---|
| TorchAO/PyTorch/ROCm API drift | Pin the exact image and commits; isolate TorchAO calls in the Flux registration |
| Silent BF16 or emulation fallback | Require profiler kernel evidence and startup capability checks |
| Full-graph compile break | Make compiled block forward/backward an early hard gate |
| FP8 slower after cast/scale overhead | Benchmark full forward/backward for real FLUX shapes |
| Small-`M` Linear regression | Use the fixed block-only policy |
| Loss or convergence drift | Use the short-run funnel before one- and three-seed convergence |
| FSDP/checkpoint incompatibility | Keep BF16 all-gather and test state-dict/DTCP round trips |
| Unintended module conversion | Match explicit FQNs and assert the Schnell candidate count |
| BF16 regression | Keep `null` as a strict no-op without TorchAO imports |
| Too many numerical changes at once | Do not combine FP8 with attention, reduction, optimizer, or all-gather changes |

## 16. Estimated Effort

| Work item | Estimate |
|---|---:|
| Pin/probe TorchAO and six MI355X shapes | 1-2 person-days |
| Config and Flux-specific conversion block | 1-2 person-days |
| Unit and single-GPU compile tests | 2-4 person-days |
| FSDP2 and DTCP qualification | 2-4 person-days |
| Eight-GPU profiling and short numerical comparison | 2-4 person-days |
| Full three-seed convergence | 1-2 calendar weeks with continuous GPU access |

A functional prototype is expected in approximately 3-6 person-days. A
production-ready implementation with compile, FSDP2, checkpoint, profiling,
and short-run numerical evidence is expected in approximately 8-15
person-days. Full MLPerf qualification is dominated by convergence runtime and
hardware availability.

## 17. Implementation Validation

The implemented policy was validated on one node with eight MI355X GPUs using
PyTorch 2.10, ROCm 7.2, TorchAO 0.15, MBS 64, checkpoint ratio `0.25`, block
compile, and no reshard.

- TorchAO converted exactly 228 block Linear modules.
- Compiled double/single block forward and backward completed with finite
  gradients, and profiler output included `aten::_scaled_mm`.
- The all-FP8 forward/dgrad/wgrad variant became non-finite by step 4 and was
  rejected.
- High-precision wgrad completed 100 steps without non-finite loss or gradient
  norm and served as the conservative selective-wgrad control.
- At step 100, BF16 loss was `0.8318` and FP8 loss was `0.8487`.
- BF16 averaged `1.6233 s/step`; FP8 averaged `1.4622 s/step`, a `9.9%`
  step-time reduction in the short run.
- Peak memory was effectively unchanged: `206.61 GiB` for BF16 and
  `206.51 GiB` for FP8.
- A full DTCP checkpoint saved at step 100 and resumed successfully for step
  101.

The performance-uplift matrix was also executed:

| Experiment | Stability | Step time | Peak memory | Decision |
|---|---|---:|---:|---|
| A0 BF16 | Stable | `1.6233 s` | `206.61 GiB` | Baseline |
| A1 FP8 forward/dgrad + HP wgrad | Stable | `1.4622 s` | `206.51 GiB` | Keep |
| A2 all-FP8 wgrad | NaN before step 20 | `1.2900 s` after NaN | `173.36 GiB` | Reject |
| A2 + power-of-two scales | NaN before step 10 | `1.2722 s` after NaN | `173.36 GiB` | Reject |
| A1.5 HP wgrad + FP8 all-gather | Stable | `1.5122 s` | `198.29 GiB` | Reject for performance |

A2 used the same 100-step shortened warmup as A0/A1, so the earlier NaN is not
explained only by the original `warmup_steps=0` smoke. Power-of-two scale
rounding did not fix it. A1.5 saved `8.22 GiB`, but its scale-precompute and
all-gather path regressed step time by `3.4%` relative to A1 under the current
single-node no-reshard policy. None of these experimental switches remain in
the production configuration surface.

Selective-wgrad screening then isolated the instability:

| FP8 wgrad group | Stability | Step time | Peak memory | Decision |
|---|---|---:|---:|---|
| None, simultaneous A1 control | Stable | `1.4944 s` | `206.51 GiB` | Control |
| Image + text QKV + attention projection | NaN before step 10 | N/A | `205.50 GiB` | Reject |
| Image + text QKV | NaN before step 20 | N/A | `205.50 GiB` | Reject |
| Image QKV only | NaN before step 10 | N/A | `206.01 GiB` | Reject |
| Text QKV only | NaN before step 10 | N/A | `206.01 GiB` | Reject |
| Attention projection only | Stable | `1.4478 s` | `206.51 GiB` | Include |
| Double-stream MLP | Stable | `1.4222 s` | `201.48 GiB` | Include |
| Single-stream Linears | Stable | `1.4056 s` | `179.40 GiB` | Include |
| Double-stream MLP + single-stream Linears | Stable | `1.3956 s` | `174.36 GiB` | Include |
| All except image/text QKV | Stable | `1.3867 s` | `174.36 GiB` | Production winner |

The winner reduces 100-step mean step time by `7.2%` versus its simultaneous
A1 control and peak memory by `32.15 GiB`. A 500-step stress run with seed
`10007` and a 100-step warmup remained finite through step 500. Excluding the
first logged warmup point, it averaged `1.3656 s/step` and `46.8818`
samples/GPU/s, with `174.36 GiB` peak memory. At step 500, loss was `0.6365`
and gradient norm was `0.3458`.

After removing the temporary group selector, the fixed production path passed
all 24 Flux backend unit tests in the target image. An eight-GPU production
smoke converted exactly 228 modules with the intended 190 FP8-wgrad / 38
high-precision-QKV split, completed five finite compiled FSDP2 steps, averaged
`1.3700 s/step` after the compile step, and saved a full DTCP checkpoint. A
separate process loaded that checkpoint at step 5, restored the scheduler,
completed finite step 6, and saved a new full checkpoint.

### 17.1 Full convergence: seed 10007

The first complete production-policy run reached the MLPerf target at exactly
the same step as the matched BF16 seed:

| Metric | BF16 | Selective FP8 | Result |
|---|---:|---:|---:|
| Target validation loss | `0.586` | `0.586` | Same |
| Convergence step | `13824` | `13824` | No step regression |
| Final validation loss | `0.585988` | `0.585874` | FP8 lower by `0.000114` |
| MLPerf time-to-quality | `33742.858 s` | `26535.193 s` | FP8 `21.4%` faster |
| Mean step time, including eval/checkpoint effects | `2.4368 s` | `1.9140 s` | FP8 `21.5%` lower |
| Arithmetic mean of logged-window throughput | `34.3468` | `40.0881 samples/GPU/s` | FP8 `16.7%` higher |
| Peak memory | `206.61 GiB` | `174.37 GiB` | FP8 saves `32.24 GiB` |

The run completed with MLPerf `run_stop` status `success`, saved a complete
final DTCP checkpoint, and had no non-finite values, OOM, or runtime error. Its
validation curve tracked the BF16 seed within approximately `0.001` through
step 2048 and reached the target after `7h 22m 15s`, saving `2h 00m 08s` versus
BF16. This establishes one full-seed convergence and time-to-quality result;
seeds `10008` and `10009` are still required for three-seed qualification.

The logged-window throughput row above is an arithmetic mean of instantaneous
window rates, not end-to-end throughput. The corresponding FP8 convergence
rate from samples divided by time-to-quality is `7,077,888 / 26,535.193 =
266.7 global samples/s`; it includes validation and periodic DTCP checkpoint
costs.

### 17.2 NeMo `delayed_short` throughput comparison

On 2026-08-01, three paired 500-step runs compared NeMo FP8 `delayed_short`
against the current native Primus tensor-wise FP8 path. Each pair ran on the
same 8 x MI355X node with seed `10007`, MBS/GBS `64/512`, compile enabled, no
validation, no periodic checkpoint, and no W&B logging. Statistics exclude the
first 100 training steps. NeMo used no activation recompute; Primus P0 retained
the production checkpoint ratio `0.25`, while P1 changed only that ratio to
`0`.

| Node | NeMo N0 global samples/s | Primus P0 global samples/s | P0 gap | Primus P1 no-AC global samples/s | P1 gap |
|---|---:|---:|---:|---:|---:|
| `crsuse2-m2m-118` | `525.505` | `375.855` | `28.48%` | `388.607` | `26.05%` |
| `crsuse2-m2m-119` | `527.308` | `377.931` | `28.33%` | `392.126` | `25.64%` |
| `crsuse2-m2m-234` | `523.667` | `375.530` | `28.29%` | `389.465` | `25.63%` |
| **Median** | **`525.505`** | **`375.855`** | **`28.48%`** | **`389.465`** | **`25.89%`** |

The median step times were `0.9743 s` for NeMo N0, `1.3620 s` for Primus P0,
and `1.3148 s` for Primus P1. Disabling activation checkpointing improved
Primus throughput by `3.62%` but increased peak memory from `174.36 GiB` to
`214.05 GiB`. It closed only about `9%` of the original absolute throughput
gap, so activation recompute is not the primary cause of the remaining
approximately `0.34 s/step` difference. P1 is an attribution experiment, not a
new production policy, and has not received convergence qualification.

The run artifacts are under:

```text
/shared_nfs/zirui/runs/flux_fp8_nemo_throughput_20260801T043638Z/
```

The runs used Primus commit `56a918b8da2323084754f5cdf91864c34465c03f`
with image `rocm/primus:v26.3`, and MLPerf/NeMo commit
`d73e1838c92bb280c8b94f9ef98c22873709ab43` with image digest
`sha256:38cbbb041c232c8a7caa53bb8409b8c2061f1c2682de247861c98d343378b3a2`.
Both worktrees contained the existing uncommitted changes used by the earlier
validation work.

### 17.3 Remaining-gap profiling and deferred scalar synchronization

Five step-exact Torch profiler iterations on rank 0 reproduced the comparison:
Primus P1 averaged `1.3316 s/step` while NeMo N0 averaged `1.0209 s/step` under
profiler overhead. Raw GPU kernel events, without double-counting parent
operators, showed the following useful attribution:

| Kernel group | Primus P1 | NeMo N0 | Primus minus NeMo |
|---|---:|---:|---:|
| FP8 GEMM + cast/scale | `604.0 ms/step` | `642.2 ms/step` | `-38.3 ms` |
| Attention | `152.9 ms/step` | `75.1 ms/step` | `+77.8 ms` |
| Norm + pointwise | `349.0 ms/step` | `292.4 ms/step` | `+56.6 ms` |
| BF16/other GEMM + optimizer | `57.1 ms/step` | `25.3 ms/step` | `+31.8 ms` |
| Communication | `265.0 ms/step` | `273.5 ms/step` | `-8.5 ms` |

These kernel-work sums include stream overlap and therefore are attribution
signals, not additive wall-clock predictions. They show that TorchAO dynamic
cast/scale is not the primary measured regression. The strongest GPU-side
targets are the SDPA attention path and the surrounding norm/pointwise work.

The CPU trace also found two premature Primus synchronizations per step: the
training loss `.item()` waited approximately `250-308 ms` before backward, and
the gradient-norm `.item()` waited approximately `697-707 ms` before the
optimizer. NeMo had one main scalar synchronization of approximately
`500-580 ms` after its combined training step. Primus now keeps detached loss
and gradient norm as tensors until a logging step, after the optimizer. This
does not change gradients, clipping, optimizer math, or checkpoint state.

Two 500-step runs validated the change:

| Configuration | Before | Deferred sync | Throughput change | Peak memory |
|---|---:|---:|---:|---:|
| P0, checkpoint ratio `0.25` | `375.9 global samples/s` median | `385.2 global samples/s` | `+2.5%` | `174.36 GiB` |
| P1, checkpoint ratio `0` | `391.8 global samples/s` on matched node `009` | `404.4 global samples/s` | `+3.2%` | `214.05 GiB` |

The deferred-sync P1 step time was `1.2658 s`, leaving approximately
`0.29 s/step` versus the NeMo median. The production P0 step time was
`1.3293 s`, leaving approximately `0.36 s/step`. This early AITER ablation used
the wrong backend name; the later explicit `flash_attn_aiter` qualification in
Section 17.5 supersedes that failure.

Profiler and follow-up artifacts are under:

```text
/shared_nfs/zirui/runs/flux_fp8_torchprof_20260801T064822Z/
/shared_nfs/zirui/runs/flux_fp8_deferred_sync_20260801T070916Z/
/shared_nfs/zirui/runs/flux_fp8_deferred_sync_p0_20260801T073524Z/
/shared_nfs/zirui/runs/flux_fp8_deferred_sync_aiter_20260801T073300Z/
```

### 17.4 Compile/reduction optimization and full convergence

A second parallel screen kept the production checkpoint ratio `0.25` and
tested attention selection, compile modes, gradient clipping, BF16 reduction,
and FSDP topology. The useful isolated results were:

| Experiment | Step time | Global throughput | Decision |
|---|---:|---:|---|
| Deferred-sync control | `1.3280 s` | `385.7 samples/s` | Baseline |
| SDPA CK preference | `1.3203 s` | `387.9 samples/s` | Too small, `+0.6%` |
| No gradient clipping | `1.3090 s` | `391.1 samples/s` | Reject numerical change |
| `max-autotune-no-cudagraphs` | `1.2937 s` | `395.7 samples/s` | Keep |
| BF16 gradient reduction | `1.2905 s` | `396.6 samples/s` | Keep as qualified candidate |
| `reduce-overhead` | `6.0790 s` | `84.2 samples/s` | Reject |
| HSDP `dp_replicate=2` | `1.4195 s` | `360.5 samples/s` | Reject |
| Root-only FSDP | `1.3003 s` | `393.9 samples/s` | Reject versus block policy |

Combining `max-autotune-no-cudagraphs` with BF16 gradient reduction produced
`1.2508 s/step` and `409.1 global samples/s`. Three nodes reproduced
`407.98-410.29 samples/s`, so the gain is not node noise. This is `6.1%`
faster than the deferred-sync control, but remains `22.1%` below the NeMo
`delayed_short` median of `525.5 samples/s`. Adding no activation checkpointing
reached `1.1883 s/step`, `430.8 samples/s`, and `214.01 GiB`, leaving an
`18.0%` throughput gap.

Node `crsuse2-m2m-234` consistently regressed to `7-8 s/step` for non-default
compile-mode runs even when the same configuration was fast elsewhere. Its
compile-mode results are excluded from candidate selection.

Both seed `10007` full-convergence candidates reached the MLPerf target and
saved complete final DTCP checkpoints:

| Metric | Qualified FP8 baseline | Optimized ratio `0.25` | Optimized no-AC |
|---|---:|---:|---:|
| Convergence step | `13824` | `14336` | `14848` |
| Samples to converge | `7,077,888` | `7,340,032` | `7,602,176` |
| Final validation loss | `0.585874` | `0.585999` | `0.585429` |
| Time-to-quality | `26,535.193 s` | `26,841.915 s` | `26,396.741 s` |
| E2E samples/s | `266.7` | `273.5` | `288.0` |
| Peak memory | `174.37 GiB` | `174.33 GiB` | `214.03 GiB` |

The optimized ratio `0.25` path improved steady throughput but converged one
validation interval later, making time-to-quality `1.2%` slower than the prior
seed `10007` baseline. No-AC converged two intervals later and improved
time-to-quality by only `0.5%`. The production decision therefore remains
checkpoint ratio `0.25`; BF16 reduction and the new compile mode need more
seeds before replacing the qualified convergence baseline.

A new five-step rank-0 profile of the optimized ratio `0.25` candidate measured
`1.2896 s/step` versus NeMo's `1.0209 s/step` under profiler overhead. The
largest kernel-work deltas were attention `+108.4 ms/step`, norm plus pointwise
`+77.0 ms/step`, FP8 GEMM plus cast/scale `+86.2 ms/step`, and BF16 GEMM plus
optimizer `+46.5 ms/step`. Kernel work overlaps across streams, so these values
rank targets rather than sum to wall time. This profile motivated the later
AITER attention and raw-GEMM screens; Sections 17.5 and 17.6 supersede its
candidate ordering.

Artifacts are under:

```text
/shared_nfs/zirui/runs/flux_fp8_opt_wave2_20260801T082156Z/
/shared_nfs/zirui/runs/flux_fp8_opt_wave3_20260801T092129Z/
/shared_nfs/zirui/runs/flux_fp8_opt_wave4_20260801T104350Z/
/shared_nfs/zirui/runs/flux_fp8_winner_convergence_20260801T143043Z/
/shared_nfs/zirui/runs/flux_fp8_winner_torchprof_20260801T220505Z/
```

### 17.5 AITER attention qualification

The earlier AITER failure used the wrong backend name. The explicit
`flash_attn_aiter` backend passed compiled FSDP2 forward/backward and a
500-step screen. FlashAttention 2 also passed, but AITER was faster:

| Attention backend | Step time | Global throughput | Peak memory |
|---|---:|---:|---:|
| SDPA winner | `1.2508 s` | `409.1 samples/s` | `174.33 GiB` |
| FlashAttention 2 | `1.1870 s` | `431.5 samples/s` | `177.19 GiB` |
| AITER, node `009` | `1.1198 s` | `457.2 samples/s` | `185.64 GiB` |
| AITER, node `235` | `1.1527 s` | `444.6 samples/s` | `185.64 GiB` |

The two-node AITER average was `450.9 samples/s`, approximately `10.2%` above
the SDPA winner and `14.2%` below NeMo `delayed_short`.

AITER then completed three independent convergence seeds with complete final
DTCP checkpoints:

| Seed | Convergence step | Samples | Validation loss | Time-to-quality |
|---:|---:|---:|---:|---:|
| `10007` | `14336` | `7,340,032` | `0.585916` | `24,275.36 s` |
| `10008` | `13312` | `6,815,744` | `0.585952` | `22,730.13 s` |
| `10009` | `14336` | `7,340,032` | `0.585817` | `24,293.85 s` |
| **Median** | **`14336`** | **`7,340,032`** | | **`24,275.36 s`** |

The corresponding SDPA compile/reduction candidate had median TTQ
`26,841.92 s`; AITER improves median TTQ by `9.6%` without changing median
convergence step. Median validation-inclusive throughput increases from
approximately `276.5` to `302.1 samples/s`. Peak memory rises by `11.31 GiB`
but remains within MI355X capacity.

The local MLPerf launcher now selects `flash_attn_aiter` by default only when
`FLUX_FLOAT8_RECIPE=tensorwise`; BF16 and other recipes retain the qualified
SDPA default. An explicit `ATTENTION_BACKEND` continues to override this
selection.

Artifacts are under:

```text
/shared_nfs/zirui/runs/flux_fp8_attention_smoke_20260802T031732Z/
/shared_nfs/zirui/runs/flux_fp8_attention_perf_20260802T111746Z/
/shared_nfs/zirui/runs/flux_fp8_attention_perf_repeat_20260802T115356Z/
/shared_nfs/zirui/runs/flux_fp8_aiter_convergence_20260802T115356Z/
/shared_nfs/zirui/runs/flux_fp8_aiter_convergence_seeds_20260802T184439Z/
```

### 17.6 ROCm FP8 GEMM kernel screen

The six FLUX Linear shapes were screened on MI355X with pre-quantized
E4M3/E5M2 operands and native forward/dgrad/wgrad layouts. The control is the
TorchAO/TorchTitan-style `torch._scaled_mm` path; TorchTitan does not provide a
separate dense FP8 kernel for this comparison. Primus-Turbo `56c789e5` was
built for `gfx950` inside the FlyDSL `0.2.3` environment and loaded by mounting
the build into a fresh container.

Summing raw latency over all six shapes and three passes gave:

| Raw GEMM backend | Sum over 18 cases | Versus `_scaled_mm` |
|---|---:|---:|
| TorchAO/TorchTitan `_scaled_mm` | `11.886 ms` | control |
| Primus-Turbo HipBLASLt | `18.539 ms` | `56.0%` slower |
| Primus-Turbo CK | `137.972 ms` | `1060.7%` slower |
| Primus-Turbo Triton | `16.072 ms` | `35.2%` slower |
| Primus-Turbo FlyDSL | `12.539 ms` | `5.5%` slower |

All FlyDSL outputs were finite and matched the control in the reported relative
L2 metric. The fused single-stream shapes did not provide a specialization
opportunity: FlyDSL regressed `(32768,3072,21504)` by
`14.1%/4.5%/6.9%` and `(32768,15360,3072)` by `0.3%/6.3%/7.6%` for
forward/dgrad/wgrad.

A short screen initially suggested FlyDSL wins for QKV forward/dgrad and MLP-up
forward. A 100-iteration repeat rejected all three: QKV became `0.6%/0.3%`
slower and MLP-up forward became `1.7%` slower. No backend met the `3%`
affected-Linear threshold, so no raw-GEMM bridge, private TorchAO hook, or
shape dispatch table was added.

Artifacts are under:

```text
/shared_nfs/zirui/runs/primus_turbo_flydsl_build_20260803/
/shared_nfs/zirui/runs/flux_fp8_gemm_flydsl_corrected_full_20260803T053501Z/
/shared_nfs/zirui/runs/flux_fp8_gemm_flydsl_candidates_20260803T054443Z/
```

Together, the results establish functional, 500-step numerical, compile,
FSDP2, checkpoint, one-seed full-convergence viability, and a reproducible
NeMo throughput gap. They do not replace the remaining three-seed
time-to-quality qualification or the profiler attribution needed to close that
gap. Compile/reduction gains alone were absorbed by convergence variance, but
AITER improved three-seed median time-to-quality by `9.6%`. Generic raw FP8
GEMM replacement did not pass K0, including on the larger fused native shapes.
The next step is to profile the exact AITER winner and then target measured
norm/pointwise fusion or host/GPU scheduling overhead; FP8 GEMM work is deferred
until a new kernel shows a repeatable shape-level win.

## 18. Final Decision

The implementation order is:

```text
freeze the BF16 reference
-> probe TorchAO on the six real MI355X shapes
-> implement the thin Primus Flux conversion
-> qualify block compile and FSDP2
-> qualify DTCP
-> run the short accuracy/performance funnel
-> run one full convergence seed
-> run three independent seeds
-> compare validation-inclusive time-to-quality
```

The first delivery remains intentionally narrow: native TorchAO dynamic
tensor-wise FP8 forward/dgrad for the 228 repeated Schnell block Linears, FP8
wgrad for the 190 non-QKV Linears, and high-precision wgrad for the 38
image/text QKV Linears, with the rest of the validated Primus BF16 training
stack unchanged. For the local MLPerf FP8 recipe, the best qualified
performance combination is deferred scalar synchronization, block compile with
`max-autotune-no-cudagraphs`, BF16 gradient reduction, AITER attention,
checkpoint ratio `0.25`, block FSDP, and no reshard. It averages
`450.9 samples/s`, remains `14.2%` below the NeMo steady throughput reference,
and improves three-seed median TTQ by `9.6%` versus the SDPA candidate. The raw
GEMM remains `_scaled_mm` because every screened ROCm replacement failed the
performance gate.
