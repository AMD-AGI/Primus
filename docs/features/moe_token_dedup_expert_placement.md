# MoE token dedup and co-activation expert placement (Megatron)

Port of ROCm/Megatron-LM [PR #165](https://github.com/ROCm/Megatron-LM/pull/165)
(`--moe-token-dedup`) and [PR #166](https://github.com/ROCm/Megatron-LM/pull/166)
(`--moe-expert-placement`) as Primus patches over the Megatron-core v0.19
all-to-all token dispatcher. `third_party/Megatron-LM` is not modified.

## What it does

With top-k routing a token often selects several experts that live on the same
expert-parallel (EP) rank. The stock all-to-all dispatcher still sends one hidden
row per (token, expert) assignment.

* **Token dedup** (`moe_token_dedup: true`) sends one All-to-All row per unique
  (token, destination EP rank). The row carries the token's local expert ids and
  routing weights, the receiver fans it out into expert-major order, and the
  combine direction is deduplicated the same way (local expert outputs for one
  wire row are summed before the combine All-to-All). Routing and the rows each
  expert computes are unchanged. At EP=1 it silently falls back to the stock
  dispatcher.
* **Co-activation expert placement** (`moe_expert_placement: <json>`) reorders
  which global expert lives on which EP rank (globally or per layer) so experts
  that are often selected together share a rank, which gives dedup fewer wire
  rows per token. Checkpoints stay keyed by global expert id, so any placement
  loads from any `torch_dist` checkpoint.

Measured on Qwen3-30B-A3B (TP1 EP8, seq 4096, GBS 128, 8x MI300X, resumed from a
2000-iteration checkpoint, ROCm Megatron-LM base): dedup alone cuts All-to-All
rows per token from 8.0 to 5.3 and step time by 3.6 to 4.0%; dedup plus the
co-activation placement cuts rows to 4.5 and step time by 4.9 to 5.9%.

## Configuration

```yaml
moe_token_dispatcher_type: alltoall   # required
use_turbo_deepep: false               # DeepEP switches to the 'flex' dispatcher
turbo_sync_free_moe_stage: 0
moe_token_dedup: true
moe_token_dedup_weighted_combine: false
moe_expert_placement: /path/to/placement.json   # optional
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

## Building a placement

1. Capture routes with the stock routing (dedup may be on or off). The capture
   is env-var gated:

   ```bash
   CMOE_CAPTURE=1 CMOE_CAPTURE_DIR=/scratch/routes CMOE_CAPTURE_START=2001 \
   CMOE_CAPTURE_CONS_ITERS=8 ./runner/primus-cli direct -- train pretrain --config <exp.yaml>
   ```

   Each rank writes one `iter_<it>_mb<k>_rank<r>.pt` shard per micro-batch with
   the per-token top-k expert ids and gate weights of every MoE layer.
   `CMOE_CAPTURE_INTERVAL`, `CMOE_CAPTURE_NUM_RANKS`, `CMOE_CAPTURE_MAX_MB` and
   `CMOE_CAPTURE_VERBOSE` are documented in
   `primus/backends/megatron/core/capture.py`.

2. Build the placement (Kernighan-Lin partition of the per-layer co-activation
   matrix; `--nodes` additionally groups ranks per node):

   ```bash
   python tools/moe_expert_placement/build_expert_placement.py /scratch/routes \
       --iters 2001-2008 --num-experts 128 --ep-size 8 --out placement.json
   python tools/moe_expert_placement/inspect_expert_capture.py /scratch/routes   # optional
   ```

3. Train with `moe_expert_placement: placement.json`. A placement names EP ranks,
   so it is specific to the EP size it was built for. Distributed-optimizer state
   is restored by buffer position, not by global expert id: when resuming a
   checkpoint saved under a different placement set `no_load_optim: true`
   (`load_checkpoint` prints a warning otherwise). Legacy `torch`-format
   checkpoints are refused under a different placement.

## Implementation

| Patch | Phase | Purpose |
| --- | --- | --- |
| `megatron.moe.token_dedup_args` | `build_args` | Validates the options and the placement JSON (the table is loaded from the path again at `before_train`). |
| `megatron.moe.token_dedup` | `before_train` | Exposes the options on `TransformerConfig`, binds `PrimusAlltoAllTokenDispatcher` / `PrimusDedupAlltoAllTokenDispatcher` in place of `MoEAlltoAllTokenDispatcher`, resolves the per-layer placement in `BaseMoELayer.__init__` / `set_layer_number`. |
| `megatron.moe.token_dedup_experts_probs` | `before_train` | Lets the expert MLPs accept `permuted_probs=None` (weighted combine only). |
| `megatron.moe.expert_placement_checkpoint` | `before_train` | Keys the distributed-checkpoint expert axis by the placed global id (`TEGroupedMLP`, `SequentialMLP`, `TEGroupedLinear`, hence also `PrimusTurboGroupedLinear`); guards `load_checkpoint`. |
| `megatron.moe.route_capture` | `before_train` | `CMOE_CAPTURE=1`: wraps `GPTModel.forward` and `TopKRouter.forward` to record routes. |

Code: `primus/backends/megatron/core/transformer/moe/token_dedup.py` (Triton
kernels and autograd functions, unchanged from the PR),
`token_dedup_dispatcher.py` (the dispatcher subclasses), `expert_placement.py`
(placement resolution and parsing), `primus/backends/megatron/core/capture.py`.
Tests: `tests/unit_tests/megatron/moe/test_token_dedup.py` (CPU kernel and
placement tests; EP parity tests on 2 to 8 GPUs).
