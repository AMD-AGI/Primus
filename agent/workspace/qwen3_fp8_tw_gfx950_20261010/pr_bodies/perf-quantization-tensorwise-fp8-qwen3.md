# Description

This PR adds a flat tensorwise FP8 quant kernel for unpadded rows shorter than 4 KB. At the Qwen3 MoE fc2 width (K=768) the quant runs 39% faster, and the output is byte-identical.

**Why.** The row-per-block tensorwise quant kernel gives each row one 256-lane block, with a 16-byte pack per lane. A row shorter than 4 KB therefore leaves lanes idle: only 96 of 256 lanes work at K=768 bf16. When nothing is padded (K == Kp, no N pad), the output keeps the input's flat layout, so a flat kernel with a fixed 16-element pack per thread keeps every lane busy.

**When it applies.** Only for rows shorter than 4 KB. Longer rows keep the row kernel, which is as fast or faster there. Misaligned pointers fall back as before.

**Microbenchmark.** Qwen3-30B-A3B MoE shapes on MI355X, quant only, bf16 → e4m3:

| shape | before | after |
|---|---|---|
| 262144 × 768 | 156.5 µs | 95.3 µs (−39%, 6.34 TB/s) |
| 262144 × 1536 | 221 µs | 207.5 µs (−6%) |
| 262144 × 2048, 32768 × {2048, 4096, 5120} | unchanged (row kernel) | |

The full `quantize_fp8` at 262144 × 768 goes from 217 µs to 156 µs. The output is byte-identical to the previous kernel on 146 shape/dtype cases.

**End-to-end.** Qwen3-30B-A3B FP8 tensorwise pretrain on 8× MI355X (EP8, MBS 8, GBS 512, seq 4096, even routing), mean of iterations 11–20 of 20-iteration runs:

| run | ms/iter | TFLOP/s/GPU | tokens/s/GPU | loss@20 |
|---|---|---|---|---|
| Turbo 1103b2df | 7925.4 | 752.5 | 33,077 | 11.33427 |
| flat tensorwise FP8 quant kernel | 7870.9 | 757.7 | 33,305 | 11.33415 |

That is −0.69% ms/iter and +0.69% tokens/s/GPU. The loss difference is within run-to-run noise (about 4e-4 at iteration 20). Both runs use Primus `use_turbo_fused_act_with_probs`.

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [ ] Code refactoring

## Changes

- `csrc/kernels/quantization/quantization_tensorwise.cu`: flat no-pad kernel, selected when nothing is padded and rows are shorter than 4 KB.
- `tests/pytorch/ops/test_quantization.py`: new `test_quantize_fp8_tensorwise_no_pad_rows`.

## Tests

- New `test_quantize_fp8_tensorwise_no_pad_rows`:
  - covers bf16 / fp16 / fp32 × e4m3 / e5m2, unaligned row counts, and 1-row and odd-K shapes;
  - checks against the reference, and byte equality with the misaligned fallback path.
- `tests/pytorch/ops/test_quantization.py -k tensorwise`: 116 passed (MI355X / gfx950).
- Tensorwise FP8 GEMM / grouped GEMM tests (which quantize through this path): about 1400 passed.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
