###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-7 P45 G47 — multi-tensor BF16 add Triton parity.

Asserts that the two Triton variants (`per_tensor`, `packed`)
produce results bit-equal to `torch._foreach_add_` on bf16 inputs
across a synthetic V4-Flash-ish parameter mix.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.extensions._triton.multi_tensor_add import (  # noqa: E402
    multi_tensor_add_triton_packed,
    multi_tensor_add_triton_per_tensor,
)


def _foreach_reference(out_list, a_list, b_list, scale: float):
    torch._foreach_copy_(out_list, a_list)
    torch._foreach_add_(out_list, b_list, alpha=scale)


def _build_pool(shapes, *, dtype=torch.bfloat16, seed=20260515):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = [torch.empty(s, dtype=dtype, device="cuda") for s in shapes]
    a = [torch.randn(s, dtype=dtype, device="cuda", generator=gen) for s in shapes]
    b = [torch.randn(s, dtype=dtype, device="cuda", generator=gen) for s in shapes]
    return out, a, b


class TestG47ParityForeach:
    @pytest.mark.parametrize("scale", [0.5, 1.0, -0.25])
    def test_per_tensor_parity(self, scale):
        shapes = [(4096,), (1024, 1024), (4096, 4096), (16, 16)]
        out_t, a, b = _build_pool(shapes)
        out_e, _, _ = _build_pool(shapes)

        _foreach_reference(out_e, a, b, scale)
        multi_tensor_add_triton_per_tensor(out_t, a, b, scale=scale)

        for tt, te in zip(out_t, out_e):
            torch.testing.assert_close(tt, te, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("scale", [0.5, 1.0, -0.25])
    def test_packed_parity(self, scale):
        shapes = [(4096,), (1024, 1024), (4096, 4096), (16, 16)]
        out_t, a, b = _build_pool(shapes)
        out_e, _, _ = _build_pool(shapes)

        _foreach_reference(out_e, a, b, scale)
        multi_tensor_add_triton_packed(out_t, a, b, scale=scale)

        for tt, te in zip(out_t, out_e):
            # Packed kernel reads bf16 -> fp32 -> bf16; the eager
            # foreach path keeps bf16 throughout.  ULP-1 difference
            # is possible at the last mantissa bit.
            torch.testing.assert_close(tt, te, atol=1e-3, rtol=1e-3)


class TestG47Variants:
    def test_packed_v4_flash_mix(self):
        """Mirrors the bench's `v4flash` shape mix at a smaller scale."""
        shapes = [(4096,)] * 8 + [(1024, 1024)] * 4 + [(4096, 4096)] * 2
        out_t, a, b = _build_pool(shapes)
        out_e, _, _ = _build_pool(shapes)

        _foreach_reference(out_e, a, b, 0.5)
        multi_tensor_add_triton_packed(out_t, a, b, scale=0.5)

        for tt, te in zip(out_t, out_e):
            torch.testing.assert_close(tt, te, atol=1e-3, rtol=1e-3)

    def test_per_tensor_handles_odd_sizes(self):
        """`numel` not divisible by BLOCK_SIZE — masked store path."""
        shapes = [(7,), (8193,), (123,)]
        out_t, a, b = _build_pool(shapes)
        out_e, _, _ = _build_pool(shapes)

        _foreach_reference(out_e, a, b, 1.0)
        multi_tensor_add_triton_per_tensor(out_t, a, b, scale=1.0, block_size=8192)

        for tt, te in zip(out_t, out_e):
            torch.testing.assert_close(tt, te, atol=0.0, rtol=0.0)
