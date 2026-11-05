# Description

The FlyDSL fused split + Q/K RMSNorm + RoPE kernels (forward, backward, gamma-grad fold) only accepted head_dim 64 (GPT-OSS). Qwen3 uses head_dim 128 with `qk_layernorm`, so on Qwen3 Primus's `PRIMUS_FUSED_QK_RMSNORM_ROPE` patch always fell back to split + TE RMSNorm + TE fused RoPE. This PR adds head_dim 128.

**How.** head_dim is now a compile-time parameter taken from the k width:
- Each lane keeps its 8-element (dwordx4) chunk per half.
- A D=128 row therefore spans 8 lanes, and a wave carries 8 rows instead of 16.
- Row reductions, backward slot counts and the gamma fold loop follow from that.

**gfx950 hazard.** At D=128 a row's hi half sits 128 bytes past its lo half, which no longer fits an inline `soffset` constant. On gfx950, a dwordx4 buffer store with an SGPR `soffset` followed by a VALU write of its data VGPRs is a hazard that LLVM does not pad: the last lanes of each 16-lane group stored clobbered data. D=128 therefore folds the hi-half offset into the VGPR offset instead. D=64 code generation is unchanged (identical ISA).

**API.** `QK_RMSNORM_ROPE_HEAD_DIM` (an int) becomes `QK_RMSNORM_ROPE_HEAD_DIMS = (64, 128)`. Nothing else in Turbo or in Primus main imports the old name. The companion Primus patch reads the tuple and falls back to `(64,)` on older Turbo.

**Microbenchmark.** Qwen3-30B-A3B attention shape on MI355X (S=4096, B=8, 4 KV groups × 8 Q heads per group, D=128), compared with split + TE RMSNorm + TE fused RoPE:

| | unfused | fused | |
|---|---|---|---|
| forward | 519.5 µs | 126.8 µs | 5.29 TB/s |
| forward + backward | 1243.6 µs | 369.9 µs | 3.36× |

The maximum difference vs the unfused path is 1 bf16 ulp.

**End-to-end.** Qwen3-30B-A3B FP8 tensorwise pretrain on 8× MI355X (EP8, MBS 8, GBS 512, seq 4096, even routing), mean of iterations 11–20 of 20-iteration runs. The patch is enabled with `PRIMUS_FUSED_QK_RMSNORM_ROPE=1` and the companion Primus change (`AMD-AGI/Primus` branch `perf/qwen3-qk-rmsnorm-rope-hd128`):

| run | ms/iter | TFLOP/s/GPU | tokens/s/GPU | loss@20 |
|---|---|---|---|---|
| before (unfused) | 7353.4 | 811.1 | 35,649 | 11.33456 |
| fused qk RMSNorm + RoPE, head_dim 128 | 6987.1 | 853.6 | 37,518 | 11.33373 |

That is −4.98% ms/iter and +5.24% tokens/s/GPU. The loss difference is within run-to-run noise (about 4e-4 at iteration 20). Both runs use Turbo 1103b2df plus the flat tensorwise quant kernel, per-token `get_dispatch_layout` and the HIP permute in the dispatcher, and Primus `use_turbo_fused_act_with_probs` and `turbo_deepep_num_cu 160`.

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [ ] Code refactoring

## Changes

- `primus_turbo/flydsl/rope/qk_rmsnorm_rope_kernel.py`:
  - The forward, backward and gamma-fold kernels are parametric in head_dim.
  - New `QK_RMSNORM_ROPE_HEAD_DIMS`; `_check_row_tileable` takes D.
  - For D=128, the hi-half store offset goes in the VGPR offset.
- `primus_turbo/pytorch/kernels/rope/qk_rmsnorm_rope_impl.py`: the shape check takes D from the k width and accepts any head_dim in `QK_RMSNORM_ROPE_HEAD_DIMS`.
- `tests/pytorch/ops/test_qk_rmsnorm_rope.py`: D=128 cases.

## Tests

- `tests/pytorch/ops/test_qk_rmsnorm_rope.py` adds D=128 forward / backward cases, including an unaligned sequence length (S=129) and NG=1 / NPG=1. The unsupported-head-dim and untileable-shape errors are checked for both head dims.
- `test_qk_rmsnorm_rope.py` + `test_rope.py`: 40 passed (MI355X / gfx950).

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
