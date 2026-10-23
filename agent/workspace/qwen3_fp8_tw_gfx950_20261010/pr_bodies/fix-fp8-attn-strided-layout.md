# Description

`flash_attn_fp8_func` and `flash_attn_fp8_usp_func` mishandle q/k/v whose memory layout is sbhd or bhsd, which is what Megatron passes. This PR fixes both.

**Bug.**
- Both functions take q/k/v whose **logical** shape is always `[b, s, h, d]`. `_infer_qkv_format` only reports how that shape is laid out in memory.
- Both functions nevertheless permuted the tensors as if the inferred format were the logical one.
- So a `[b, s, h, d]` view of sbhd storage became `[s, b, h, d]`.
- Block scaling then failed with `shape '[4096, 32, 0, 64, 128]' is invalid`, or, when the shapes happened to fit, silently attended over the wrong axis.
- bhsd storage hit the same bug.

**Fix.** Make q/k/v bshd-contiguous instead, and hand the output back in the caller's storage layout with its logical `[b, s, h, d]` shape, as the BF16 path already does.

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [ ] Code refactoring

## Changes

- `primus_turbo/pytorch/ops/attention/attention_utils.py`: new `_restore_qkv_storage(o, qkv_format)`, which gives the bshd-contiguous output the caller's storage layout and keeps its logical `[b, s, h, d]` shape.
- `primus_turbo/pytorch/ops/attention/flash_attn_interface.py`, `flash_attn_usp_interface.py`:
  - Make q/k/v `.contiguous()` instead of permuting them by the inferred format.
  - Restore the output with `_restore_qkv_storage`.
- `tests/pytorch/ops/test_attention.py`: new `test_attention_fp8_strided_layout`.

## Tests

- New `test_attention_fp8_strided_layout` covers bshd / sbhd / bhsd storage × causal / non-causal (GQA 16/4 heads, head_dim 128, seq 1024). It checks:
  - the output shape;
  - that the output keeps the caller's storage layout;
  - output, dQ, dK and dV SNR > 20 dB vs the PyTorch reference.
- `tests/pytorch/ops/test_attention.py -k test_attention_fp8`: 78 passed, including the 6 new cases (MI355X / gfx950).

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
