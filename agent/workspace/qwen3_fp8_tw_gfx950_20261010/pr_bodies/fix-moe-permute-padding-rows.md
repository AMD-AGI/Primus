# Description

The TURBO (HIP) `moe_permute` leaves each expert's padding rows uninitialized when `pad_multiple > 0` and the dispatch buffer ends in unrouted rows. The padding rows then hold whatever `torch.empty` returned instead of zeros.

**Root cause.** `permute_preprocessing` writes the `pad_multiple` padding rows of `row_id_map` **after all `max_num_dispatched_tokens` token rows**. `permute_kernel`, however, looked up padding token `i` at row `num_dispatched_tokens + i`, where `num_dispatched_tokens` is the *routed* count. The two positions coincide only when every buffer row is routed. With an unrouted tail, the kernel read the empty rows of unrouted tokens (`n_routed = 0`), so no padding row was ever written.

**Who is affected.** Anyone combining `pad_multiple > 0` with a worst-case receive buffer:
- `DeepEPTokenDispatcher` with `deepep_num_worst_tokens > 0` and `pad_multiple > 0`.
- In Primus, sync-free MoE stage 2 and up (which set `num_worst_tokens`) together with padding for quantization.

The grouped GEMM consumes the garbage rows. They enter both expert weight gradients and, under tensorwise FP8, the amax of the permuted activations, so they can change the scale. The gradient permute in `moe_unpermute`'s backward runs the same kernel and had the same bug. Paths where the routed count equals the buffer size (the default dispatch path, DeepEP sync-free stage 1) are not affected. Before this fix they behaved correctly, and they behave identically after it.

**Repro** (op level, before the fix): a 32768-row buffer with 4000 routed tokens (hidden 4096, 32 local experts, top-2, `pad_multiple=16`) had about 220 of its 224 padding rows nonzero. After the fix, all are zero, and the FP8 and bf16 permute outputs are bit-identical.

**Performance.** The fix only changes which `row_id_map` row a padding token reads. `permute_kernel` time is unchanged at the Qwen3-30B-A3B EP8 shape (32768 tokens, hidden 2048, 16 local experts, top-8, `pad_multiple=16`): bf16 232.3 µs (232 before), FP8 97.3 µs (95 before).

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [ ] Code refactoring

## Changes

- `csrc/kernels/moe_permute/moe_permute.cu`:
  - `permute_kernel` takes `max_num_dispatched_tokens` (= `row_id_map.size(0) - pad_multiple`).
  - It reads padding token `i` from row `max_num_dispatched_tokens + i`; the row offset is now computed in `int64`.
  - When every buffer row is routed, the row index is unchanged.
- `tests/pytorch/ops/test_moe_permute.py`: new `test_moe_permute_padding_rows_zero_with_unrouted_tail` (`pad_multiple` 8 and 16).
  - It routes only the first 300 of 1024 tokens.
  - It hands the output allocation NaN-filled memory, so an unwritten row cannot pass as zero.
  - It checks that the padding rows are zero and the real rows match the source tokens.

Not changed: TURBO `unpermute` still leaves output rows past the routed count unwritten. DeepEP combine reads only the received rows, so those rows are never consumed.

## Tests

- The new test **fails on main (1103b2df) and passes with this fix** (2/2).
- `tests/pytorch/ops/test_moe_permute.py`: 144 passed, 19 skipped (MI355X / gfx950).
- `tests/pytorch/modules/test_token_dispatcher.py` (2 GPUs): 25 passed. This includes a worst-token + `pad_multiple=16` case that failed intermittently before the fix, and passed 5/5 repeated runs with it.

The last two suites ran on a build that also contains the follow-up branch `perf/moe/fp8-permute-tensorwise`, which builds on this fix.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
