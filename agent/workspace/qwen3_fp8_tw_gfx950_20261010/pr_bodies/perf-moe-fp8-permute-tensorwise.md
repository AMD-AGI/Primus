# Description

> Stacked on #<permute-default PR> (`perf/moe/permute-default-hip`): the dispatcher must run the TURBO permute. I will rebase onto main once that merges; until then, review the last two commits (`perf(moe): emit tensorwise FP8 ...` and the format fix).

Under FP8 tensorwise, the fused grouped MLP (`grouped_mlp_fp8`) quantizes:
- its input right after `DeepEPTokenDispatcher` permutes it, and
- its output gradient right after `moe_unpermute`'s backward permutes that.

Each time, permute writes bf16 rows, an amax pass re-reads them, and the quant kernel re-reads them and writes FP8. All three run at the permuted size (top-k × the received tokens).

A tensorwise scale commutes with a row copy. So the quantize can run on the received tokens, and the permute can move FP8 bytes instead:
- A new `permute_routed_amax` op computes per-block abs-max partials over exactly the tokens the permute copies. The unrouted tail of a worst-case buffer never reaches the scale.
- `quantize_fp8_tensorwise_pad_impl` takes those partials, so there is no separate amax pass, and pads the columns to 128.
- The TURBO permute copies the FP8 rows; padding rows stay zero.

The result is **bit-identical** to quantizing the bf16 permute output.

**New opt-in API** (off by default; TURBO permute backend only):
- `moe_permute(..., quantize_dtype=<fp8 dtype>)` returns the permuted tokens as a grouped tensorwise `QuantizedTensor` (`group_lens = tokens_per_expert`).
- `moe_unpermute(..., grad_quantize_dtype=<fp8 dtype>)` makes its backward return the gradient as a tensorwise `QuantizedTensor`.
- `DeepEPTokenDispatcher._post_dispatch` / `_pre_combine` pass both options through.
- `grouped_mlp_fp8` backward accepts a `QuantizedTensor` `grad_out`, as its forward already accepted a `QuantizedTensor` input.

Primus turns it on with `turbo_fp8_permute` (companion PR from `AMD-AGI/Primus` branch `perf/megatron/turbo-fp8-permute`).

**Microbenchmark.** Qwen3-30B-A3B EP8 dispatcher call (32768 received tokens, hidden 2048, 16 local experts, top-8, `pad_multiple=16`, about 262k permuted rows), kernel time per call:

| | before | after |
|---|---|---|
| forward | bf16 permute 232 + amax 154 + quant 255 + scale 4 = **645 µs** | routed amax 36 + quant 32 + scale 4 + FP8 permute 95 = **167 µs** |
| backward (gradient permute + quantize) | **687 µs** | **174 µs** |

The FP8 permute and the quant kernel run at about 6.4 and 6.3 TB/s.

**End-to-end.** Qwen3-30B-A3B FP8 tensorwise pretrain on 8× MI355X (EP8, MBS 8, GBS 512, seq 4096, even routing), mean of iterations 11–20 of 20-iteration runs. The base has every earlier change of this series plus Primus `turbo_fused_grouped_gemm`:

| run | runs (ms/iter) | mean ms/iter | tokens/s/GPU | mem | loss@20 |
|---|---|---|---|---|---|
| `turbo_fused_grouped_gemm` | 6820.2, 6815.1 | 6817.7 | 38,451 | 224.5 GB | 11.33412 |
| + `turbo_fp8_permute` | 6442.4, 6515.0, 6523.8, 6549.1 | 6507.6 | 40,283 | 223.3 GB | 11.33404 |

That is −4.55% ms/iter and +4.76% tokens/s/GPU. The later FP8-permute runs show more per-iteration jitter than the first one; I have not identified the cause. Loss tracks the baseline at every iteration, within run-to-run noise (about 4e-4 at iteration 20).

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [ ] Code refactoring

## Changes

- `csrc/kernels/moe_permute/moe_permute.cu`, `csrc/include/primus_turbo/moe_permute.h`: `permute_routed_amax_kernel` / `permute_routed_amax_impl` (bf16 and fp16).
- `csrc/pytorch/moe_permute/moe_permute.cpp`, `extensions.h`, `bindings_pytorch.cpp`: `permute_routed_amax` op with CUDA and Meta implementations.
- `primus_turbo/pytorch/kernels/moe/moe_permute_impl.py`: `moe_permute_routed_amax_impl`.
- `primus_turbo/pytorch/ops/moe/moe_permute.py`:
  - `quantize_dtype` / `grad_quantize_dtype` options, through a shared helper.
  - Either option raises `ValueError` with a non-TURBO backend, and `quantize_dtype` also does with `use_fp8` / `scaling_factor`.
- `primus_turbo/pytorch/ops/grouped_mlp_fp8.py`: tensorwise backward accepts a `QuantizedTensor` `grad_out`, after checking its dtype and 128-column padding.
- `primus_turbo/pytorch/modules/moe/token_dispatcher.py`: `_post_dispatch(quantize_dtype=...)`, `_pre_combine(grad_quantize_dtype=...)`.
- Tests, described below.

## Tests

- `tests/pytorch/ops/test_moe_permute.py`: 144 passed, 19 skipped.
  - New `test_moe_permute_quantize_tensorwise` and `test_moe_unpermute_grad_quantize_tensorwise`, over e4m3/e5m2, hidden 2048 and 1000, and `pad_multiple` 0/16. Outliers sit in the unrouted tail, and the results are bit-identical to quantizing the bf16 result.
  - New `test_moe_permute_quantize_requires_turbo` and `test_moe_permute_quantize_empty_input`.
- `tests/pytorch/ops/test_grouped_mlp_fp8.py`: new `test_grouped_mlp_fp8_prequantized_input_and_grad`, over E4M3 and HYBRID. The output and every gradient are bit-identical to the bf16-input path. All tensorwise tests pass (14 passed).
- `tests/pytorch/modules/test_token_dispatcher.py` (2 GPUs): 25 passed. New `test_fp8_permute` covers `pad_multiple` 0/16 × `num_worst_tokens` 0/32768 and passed 5/5 repeated runs.

The MXFP8 grouped-MLP path (`FP8GroupedMLPMXFunc`) is untouched.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
