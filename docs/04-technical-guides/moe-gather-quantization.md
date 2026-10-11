# MoE gather and MXFP4 quantization fusion

This experimental, opt-in path avoids materializing the BF16 expert-ordered
activation and output-gradient tensors before grouped MXFP4 quantization.
The quantizer reads original token rows through a destination-to-source map.
Required expert output combination and input-gradient reduction remain unchanged.

## Supported configuration

The measured path is GPT-OSS on MI355X/gfx950, with TP=1, EP=1, BF16 activations,
32 experts, top-4 routing, hidden width 2880, and Turbo's fused grouped MXFP4 MLP.
Backward fusion additionally restricts the source to at most 32768 tokens, no
merging probabilities or capacity padding at unpermute, and ordinary first-order
backward outside CUDA graph capture. Unsupported backward consumers materialize
the existing gather. Tensor hooks and aliases share that materialized storage.
Changing routing/source tensors before consumption raises an error.

Both repositories need their paired gather-fusion implementation. Enable in the
training process only when the permutation output goes directly to the supported
Turbo grouped MLP/quantizer (storage-preserving EP1 identity transport is allowed):

```yaml
tensor_model_parallel_size: 1
expert_model_parallel_size: 1
moe_token_dispatcher_type: alltoall
moe_permute_fusion: true
enable_primus_turbo: true
use_turbo_grouped_gemm: true
turbo_fused_grouped_gemm: true
fp4: e2m1
fp4_recipe: mxfp4
moe_skip_identity_sort: true
moe_permute_quant_fusion: true
moe_backward_permute_quant_fusion: true
```

Set `moe_backward_permute_quant_fusion: false` for forward-only fusion. Both fusion arguments are false
by default; they are not a general-purpose replacement for every permutation
consumer. The forward placeholder has no activation values until its registered
quantizer consumes the source and map. Do not apply ordinary value operations to
it. Missing or mismatched handoff metadata raises an error rather than reading
uninitialized activation storage. Primus loads the paired Turbo helpers lazily,
so disabling the arguments keeps compatibility with installations without them.

## Evidence and validation limits

PyTorch profiles of the original implementation show 24 forward and 24
backward gather launches removed per step, with 11.08–11.18 ms of net raw target
kernel savings after accounting for replacement work. The estimated 72.48
GB/rank/step of avoided intermediate reads/writes is logical tensor traffic,
not measured HBM traffic or peak-memory savings.

The original implementation passed six forward and fourteen backward exactness cases,
including stochastic-rounding progression, defined quantized payload/scales,
BF16 main-gradient ownership, hooks, aliases and outstanding backward calls.
They used older pinned source revisions. The latest-main port and its composition
with router-backward optimization need new paired GPU correctness/profiling
runs; the kernel measurements above do not validate that stack.
Full training-to-target/convergence equivalence has not been established.

## Implementation and tests

`primus/backends/transformer_engine/pytorch/permutation.py` manages the opt-in
handoff and deferred unpermute gradient. The Triton helpers build compact routing
metadata and the destination-to-source map without copying the activations.

With the paired Turbo installed on gfx950:

```bash
pytest tests/unit_tests/backends/transformer_engine/test_moe_gather_quantization.py -v
```

This compares unfused, forward-only and bidirectional paths, including every
input/weight/probability gradient and the stochastic-rounding counter. The test
skips when the paired Turbo module or a GPU is unavailable; a skip is not GPU
validation.

## Argument-based integration

Primus validates the topology and Turbo consumer from the training arguments,
then opts in only inside the all-to-all dispatcher’s MXFP4 forward path. Generic
permutation calls remain eager. Direct callers use
`moe_permute_with_probs(..., fuse_permute_quant=True,
fuse_backward_permute_quant=True)` and must provide the supported Turbo consumer.
The backward choice travels with the registered forward handoff; Turbo no longer
reads the training process’s topology or fusion environment variables.

The old `GPTOSS_FUSED_PERMUTE_QUANT`, `GPTOSS_FUSED_BACKWARD_PERMUTE_QUANT`, and
`MOE_SKIP_IDENTITY_SORT` switches must be mapped into the corresponding Primus
YAML arguments by downstream launchers. `PRIMUS_TP` and `PRIMUS_EP` may still be
used by launchers to populate the parallelism arguments; kernels do not read them.
Both repositories must be updated together for the new handoff keyword argument.
