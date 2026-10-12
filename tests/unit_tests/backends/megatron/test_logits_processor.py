###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests for the public fused_softcap wrapper (PRPUNDIT-5).

The kernel used to evaluate tanh via exp(2x) in the working dtype. For a finite
saturation-range logit such as 2000 with softcapping 30, 2x overflowed FP32 and
the in-place store became NaN. These tests reach fused_softcap_kernel through
fused_softcap and compare against a float64 scaled-tanh reference, including a
length that is not a multiple of BLOCK_SIZE=1024.
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.extensions.logits_processor import fused_softcap

_SOFTCAP = 30.0
# 5 special values + 1026 extras = 1031, so the last Triton block is a masked tail.
_N_ELEMENTS = 1031
_SPECIAL = (-2000.0, -30.0, 0.0, 30.0, 2000.0)
_FP32_ATOL = 2e-5
_FP32_RTOL = 2e-5
# BF16 rounding around the ±30 saturation plateaus.
_BF16_ATOL = 5e-2
_BF16_RTOL = 2e-2


def _logits(dtype: torch.dtype) -> torch.Tensor:
    special = torch.tensor(_SPECIAL, dtype=dtype, device="cuda")
    tail = torch.linspace(-10.0, 10.0, _N_ELEMENTS - len(_SPECIAL), dtype=dtype, device="cuda")
    return torch.cat([special, tail], dim=0)


def _reference(original: torch.Tensor, softcap: float) -> torch.Tensor:
    orig_f64 = original.detach().to(dtype=torch.float64, device="cpu")
    ref = softcap * torch.tanh(orig_f64 / softcap)
    return ref.to(dtype=original.dtype, device=original.device)


def _assert_matches_scaled_tanh(out: torch.Tensor, original: torch.Tensor) -> None:
    assert out.shape == original.shape
    assert out.dtype == original.dtype
    assert torch.isfinite(out).all(), "fused_softcap produced non-finite values for finite inputs"
    ref = _reference(original, _SOFTCAP)
    atol, rtol = (_BF16_ATOL, _BF16_RTOL) if original.dtype == torch.bfloat16 else (_FP32_ATOL, _FP32_RTOL)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol, equal_nan=False)
    # Saturation-range specials sit at the front of the buffer.
    torch.testing.assert_close(out[0], ref[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(out[4], ref[4], atol=atol, rtol=rtol)
    assert out[0].item() < 0.0
    assert out[4].item() > 0.0
    assert abs(out[0].item() + _SOFTCAP) < 1.0
    assert abs(out[4].item() - _SOFTCAP) < 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=("fp32", "bf16"))
def test_fused_softcap_matches_float64_scaled_tanh(dtype):
    original = _logits(dtype)
    logits = original.clone()
    result = fused_softcap(logits, _SOFTCAP)
    assert result is logits
    _assert_matches_scaled_tanh(logits, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=("fp32", "bf16"))
def test_fused_softcap_oracle_rejects_zero_output_fault(dtype):
    original = _logits(dtype)
    logits = original.clone()
    fused_softcap(logits, _SOFTCAP)
    logits.zero_()
    with pytest.raises(AssertionError):
        _assert_matches_scaled_tanh(logits, original)
