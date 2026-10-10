# Qwen3-30B-A3B FP8 tensorwise on 8x MI355X: Turbo kernel campaign (handoff)

Task: [prompt.md](prompt.md). Log of every round: [logs/optimize.md](logs/optimize.md).

Status (2026-10-10): the 40,000 tokens/s/GPU target is reached. The baseline was 29,474 tokens/s/GPU and the best stack runs at 40,283 (mean of 4 runs).

## End-to-end

Qwen3-30B-A3B FP8 tensorwise pretrain on 8x MI355X: EP8, MBS 8, GBS 512, seq 4096, even routing, 20 iterations, mean of iterations 11-20. tokens/s/GPU = 262144 / (s/iter). Each row adds one change on top of the row above.

| run | ms/iter | TFLOP/s/GPU | tokens/s/GPU | mem | loss@20 |
|---|---|---|---|---|---|
| baseline: submitted config, Turbo e2d9f1d7 | 8894.0 | 670.6 | 29,474 | 277 GB | 11.33393 |
| + `use_turbo_fused_act_with_probs` | 7915.6 | 753.5 | 33,117 | 223 GB | 11.33435 |
| Turbo 1103b2df (base of the Turbo branches) | 7925.4 | 752.5 | 33,077 | 222 GB | 11.33427 |
| + flat tensorwise FP8 quant kernel | 7870.9 | 757.7 | 33,305 | 222 GB | 11.33415 |
| + per-token `get_dispatch_layout` | 7774.2 | 767.2 | 33,720 | 222 GB | 11.33431 |
| + HIP permute in `DeepEPTokenDispatcher` | 7643.5 | 780.3 | 34,296 | 222 GB | 11.33423 |
| + `turbo_deepep_num_cu 160` | 7353.4 | 811.1 | 35,649 | 223 GB | 11.33456 |
| + fused qk RMSNorm + RoPE, head_dim 128 | 6987.1 | 853.6 | 37,518 | 224 GB | 11.33373 |
| + `turbo_fused_grouped_gemm` (2 runs: 6820.2, 6815.1) | 6817.7 | 874.9 | 38,451 | 224.5 GB | 11.33412 |
| + `turbo_fp8_permute` (4 runs: 6442.4, 6515.0, 6523.8, 6549.1) | 6507.6 | 916.7 | 40,283 | 223.3 GB | 11.33404 |

Run-to-run loss noise is about 4e-4 at iteration 20.

## Settings for the last row

The final configuration is `examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml` plus:
- `--use_turbo_fused_act_with_probs True`
- `--turbo_deepep_num_cu 160`
- `--turbo_fused_grouped_gemm True`
- `--turbo_fp8_permute True`
- env `PRIMUS_FUSED_QK_RMSNORM_ROPE=1`

Code it needs:
- Turbo: all five Turbo branches below. Building on `perf/moe/fp8-permute-tensorwise` also brings in the permute-default change.
- Primus: `perf/qwen3-qk-rmsnorm-rope-hd128`, which sits on `perf/qwen3-30b-a3b-tuning`, plus `perf/megatron/turbo-fp8-permute`.

See [e2e/e2e_run.sh](e2e/e2e_run.sh) and [e2e/env_turbo_opt_qknorm.sh](e2e/env_turbo_opt_qknorm.sh). Their paths are for the old machine.

## Branches and PRs

Turbo (`AMD-AGI/Primus-Turbo`):

| branch | head | PR | notes |
|---|---|---|---|
| `fix/moe/permute-padding-rows` | f1a59fb9 | #553 merged | TURBO permute padding rows with worst-case buffers + pad |
| `perf/moe/permute-default-hip` | 30b6cc7c | #554 open | body: [pr_bodies/perf-moe-permute-default-hip.md](pr_bodies/perf-moe-permute-default-hip.md) |
| `perf/moe/fp8-permute-tensorwise` | f4485b88 | #555 open | on main + #554's commit; body: [pr_bodies/perf-moe-fp8-permute-tensorwise.md](pr_bodies/perf-moe-fp8-permute-tensorwise.md) |
| `perf/quantization/tensorwise-fp8-qwen3` | 6351abf4 | #558 open | independent; body: [pr_bodies/perf-quantization-tensorwise-fp8-qwen3.md](pr_bodies/perf-quantization-tensorwise-fp8-qwen3.md) |
| `perf/deep_ep/dispatch-layout-per-token` | 0931ba1e | #557 open | independent; body: [pr_bodies/perf-deep-ep-dispatch-layout-per-token.md](pr_bodies/perf-deep-ep-dispatch-layout-per-token.md) |
| `perf/flydsl/qk-rmsnorm-rope-hd128` | 1b920afe | #556 open | independent; needed by the Primus qk-norm patch; body: [pr_bodies/perf-flydsl-qk-rmsnorm-rope-hd128.md](pr_bodies/perf-flydsl-qk-rmsnorm-rope-hd128.md) |
| `fix/fp8-attn-strided-layout` | c825b13e | opening | FP8 attention with sbhd/bhsd storage; body: [pr_bodies/fix-fp8-attn-strided-layout.md](pr_bodies/fix-fp8-attn-strided-layout.md) |

`perf/moe/fp8-permute-tensorwise` is main (with #553) plus #554's commit plus the FP8 permute commit, whose message quotes the 4-run mean (-4.55%). The pre-rebase history is kept locally as `backup/fp8-permute-pre-rebase`. Once #554 merges, rebase the branch onto main and drop #554's commit.

Primus (`AMD-AGI/Primus`):

| branch | head | PR | notes |
|---|---|---|---|
| `perf/qwen3-30b-a3b-tuning` | 82ebf666 | not opened | Qwen3 FP8 config tuned to Turbo tensorwise |
| `perf/qwen3-qk-rmsnorm-rope-hd128` | bfa227f2 | not opened | on top of the tuning branch; fused qk-norm patch for head_dim 128 |
| `perf/megatron/turbo-fp8-permute` | 69a32f64 | not opened | on main; `turbo_fp8_permute` flag |

## Rebuilding the stack on a new machine

All of these merges are clean as of 2026-10-10.

Turbo:

```bash
git fetch origin && git checkout -b integ/qwen3-fp8 origin/main
for b in perf/quantization/tensorwise-fp8-qwen3 perf/deep_ep/dispatch-layout-per-token \
         perf/flydsl/qk-rmsnorm-rope-hd128 perf/moe/fp8-permute-tensorwise; do
  git merge --no-edit origin/$b || break
done
GPU_ARCHS=gfx950 MAX_JOBS=64 python setup.py build_ext --inplace
```

Primus:

```bash
git fetch origin && git checkout -b integ/qwen3-fp8 origin/perf/qwen3-qk-rmsnorm-rope-hd128
git merge --no-edit origin/perf/megatron/turbo-fp8-permute
git submodule update --init third_party/Megatron-LM
```

Then edit the paths in `e2e/env_turbo_opt_qknorm.sh` and `e2e/e2e_run.sh` for the new machine, and run, for example:

```bash
e2e/e2e_run.sh g1 --turbo_fused_grouped_gemm True --turbo_fp8_permute True
```

Summarize the runs with `e2e/parse_e2e.py <run names>`, which reads `/tmp/q3_e2e_<name>.log`.

Run Python from the campaign directory with `PYTHONPATH=<turbo checkout>`. Otherwise the pip-installed Turbo is imported.

## Findings and rejected options

- **MegaMoE** (bf16 and mxfp8) runs out of memory at MBS 8. When the worst-case dispatch pool allocation fails, 261 GB (bf16, failing on 3 GiB) or 264 GB (mxfp8, failing on 1.5 GiB) is already allocated. Using it would need a MegaMoE change, a smaller MBS, or recompute.
- **`turbo_deepep_num_cu`**:
  - 192 gives a NaN forward loss on rank 2 at iteration 5, a DeepEP bug at high channel counts.
  - 256 is slower (7028 ms).
  - 160 is best.
- **Sync-free MoE stage 2** costs +300 ms over stage 1. `num_worst_tokens` makes every post-dispatch step scan 262144 receive rows instead of 32768 tokens.
- **`turbo_grouped_gemm_without_padding`** needs `moe_router_padding_for_quantization False`, which sync-free stage 1 forces to True.
- **Attention**: TE CK is the fastest. The Turbo AITER forward is 2x slower (causal kernel without tile skipping), so keep `use_turbo_attention` false.
- **#455** made TRITON the generic `moe_permute` default, and it is slower than HIP at every measured shape. `perf/moe/permute-default-hip` switches the dispatcher back.
- **MXFP4 tests**: 20 MXFP4 quantization tests also fail on the original build (`x.is_contiguous()` assertion in MXFP4 dequant). Unrelated to this work.
- **TURBO permute requires routed tokens to come first.** DeepEP guarantees this. The worst-case + pad padding-row bug is fixed in #553. TURBO unpermute still leaves rows past the routed count unwritten; this is harmless because DeepEP combine reads only received rows.

## Open items

- **Unexplained e2e jitter with FP8 permute.** The first run took 6442 ms; three later runs took 6515-6549 ms with more per-iteration jitter. These causes are ruled out:
  - the Primus change: a same-code rerun gave 6515 ms;
  - the padding fix: permute kernel times are unchanged;
  - the node: the `turbo_fused_grouped_gemm` baseline reran at 6815 ms.
  The cause is not identified.
- **Recompute**: `turbo_fp8_permute` rejects selective recompute of `moe_act`, because the checkpointed fused-MLP path is not validated with `QuantizedTensor` inputs and gradients.
- **Untested on the final build**: the 12 MXFP8 tests in `tests/pytorch/ops/test_grouped_mlp_fp8.py`. The MXFP8 path is untouched, and FlyDSL compilation takes more than 30 minutes.
- **Remaining hot spots** (R3 profile, rank 0, one step, taken before the fused MLP and the FP8 permute):

  | kernel group | ms |
  |---|---|
  | attention | 1583 |
  | grouped GEMM | 1268 |
  | hipBLASLt | 1003 |
  | DeepEP | 876 |
  | quant chain | 739 |
  | permute / unpermute | 346 |
  | SwiGLU | 262 |

## Files

- `bench_*.py`: per-item microbenchmarks at Qwen3 shapes. `bench_fp8_permute.py --time` gives the kernel breakdown of the FP8 permute.
- `trace_*.py`: kernel-time attribution from torch profiler traces.
- `debug_*.py` and `dbg_*.py`: one-off reproducers.
- `quant_variants/`: tensorwise quant kernel variants, source only.
- `e2e/`: run, queue and A/B scripts, the env files, `parse_e2e.py`, and `amend_msg.py`, which writes the cumulative table into a commit message.
- `pr_bodies/`: PR descriptions, ready to paste.
