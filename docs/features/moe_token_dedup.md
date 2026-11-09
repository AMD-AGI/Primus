# MoE token dedup (Megatron)

Port of ROCm/Megatron-LM [PR #165](https://github.com/ROCm/Megatron-LM/pull/165)
(`--moe-token-dedup`) as Primus patches over the Megatron-core v0.19 all-to-all
token dispatcher. `third_party/Megatron-LM` is not modified.

## What it does

With top-k routing a token often selects several experts that live on the same
expert-parallel (EP) rank. The stock all-to-all dispatcher still sends one hidden
row per (token, expert) assignment.

`moe_token_dedup: true` sends one All-to-All row per unique (token, destination
EP rank). The row carries the token's local expert ids and routing weights, the
receiver fans it out into expert-major order, and the combine direction is
deduplicated the same way (local expert outputs for one wire row are summed
before the combine All-to-All). Routing and the rows each expert computes are
unchanged. At EP=1 it silently falls back to the stock dispatcher.

Measured on Qwen3-30B-A3B (TP1 EP8, seq 4096, GBS 128, 8x MI300X, resumed from a
2000-iteration checkpoint, ROCm Megatron-LM base): dedup cuts All-to-All rows
per token from 8.0 to 5.3 and step time by 3.6 to 4.0%.

## Configuration

```yaml
moe_token_dispatcher_type: alltoall   # required
use_turbo_deepep: false               # DeepEP switches to the 'flex' dispatcher
turbo_sync_free_moe_stage: 0
moe_token_dedup: true
moe_token_dedup_weighted_combine: false
```

Constraints (validated at `build_args`): all-to-all dispatcher, dropless routing
(no `moe_expert_capacity_factor` / `moe_pad_expert_input_to_capacity` /
`moe_router_padding_for_quantization`), `expert_tensor_parallel_size: 1`, no
CUDA graphs, at most 256 local experts, and neither `use_turbo_mega_moe` nor the
deprecated 20241209 MoE layer. Primus-Turbo attention and the Turbo grouped
GEMM are compatible.

`moe_token_dedup_weighted_combine: true` applies the routing weights in the
dedup combine's reduce kernel instead of inside the experts; the experts then
run with unit weights (Primus keeps the expert MLPs untouched, so the multiply
by ones remains). It is numerically equivalent and off by default.

See `examples/megatron/configs/MI300X/qwen3_30B_A3B-BF16-pretrain-token_dedup.yaml`.

## Implementation

| Patch | Phase | Purpose |
| --- | --- | --- |
| `megatron.moe.token_dedup_args` | `build_args` | Validates the options against the dispatcher and routing settings. |
| `megatron.moe.token_dedup` | `before_train` | Exposes the options on `TransformerConfig`, binds `PrimusDedupAlltoAllTokenDispatcher` in place of `MoEAlltoAllTokenDispatcher`. |
| `megatron.moe.token_dedup_experts_probs` | `before_train` | Lets the expert MLPs accept `permuted_probs=None` (weighted combine only). |

Code: `primus/backends/megatron/core/transformer/moe/token_dedup.py` (Triton
kernels and autograd functions, unchanged from the PR) and
`token_dedup_dispatcher.py` (the dispatcher subclass).
Tests: `tests/unit_tests/megatron/moe/test_token_dedup.py` (CPU kernel tests;
EP parity test on 2 to 8 GPUs).
