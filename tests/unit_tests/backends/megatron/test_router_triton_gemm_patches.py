###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU numerical tests for the Triton FP32 router GEMM patch."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton

from primus.core.patches import PatchContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Load just this patch so the test does not import unrelated GPU patch modules.
_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/moe_patches/router_triton_gemm_patches.py"
)
_spec = importlib.util.spec_from_file_location("router_triton_gemm_under_test", _PATCH_PATH)
router_triton = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router_triton)


def _patch_context(**kwargs):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**kwargs))},
    )


def _inputs(tokens, hidden, experts, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, hidden, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(experts, hidden, device="cuda", generator=g) * hidden**-0.5).to(torch.bfloat16)
    dy = torch.randn(tokens, experts, device="cuda", generator=g)
    return x, w, dy


@pytest.mark.parametrize("experts", [32, 128, 256])
@pytest.mark.parametrize("tokens", [8192, 1000])
def test_kernels_match_fp64(tokens, experts):
    hidden = 7168
    x, w, dy = _inputs(tokens, hidden, experts)
    assert router_triton._supported(x, w, None, torch.float32)
    x64, w64, dy64 = x.double(), w.double(), dy.double()

    def rel_err(a, ref):
        return ((a.double() - ref).abs().max() / ref.abs().max()).item()

    # Each result must be about as accurate as Megatron's FP32 path (torch FP32 GEMMs here).
    out = router_triton._fwd(x, w)
    ref = x64 @ w64.t()
    assert rel_err(out, ref) <= max(2 * rel_err(x.float() @ w.float().t(), ref), 1e-6)

    # dX is returned in BF16: within one BF16 rounding of the FP64 result
    dx = router_triton._dx(dy, w)
    torch.testing.assert_close(dx, (dy64 @ w64).to(torch.bfloat16), rtol=1.6e-2, atol=1e-2)

    # dW: dlogits enter as a BF16 hi + lo pair (about 16 mantissa bits), so the result stays close to
    # FP32 and far from what rounding dlogits to BF16 (moe_router_bwd_bf16) gives
    dw = router_triton._dw(dy, x)
    ref_dw = dy64.t() @ x64
    err = rel_err(dw, ref_dw)
    err_bf16 = rel_err(dy.to(torch.bfloat16).float().t() @ x.float(), ref_dw)
    assert err < 2e-5 and err < 0.05 * err_bf16, f"dW relative error {err:.2e} (BF16 dlogits {err_bf16:.2e})"


@pytest.mark.parametrize("experts", [32, 128, 256])
@pytest.mark.parametrize("tokens", [8192, 1000])
def test_gfx1250_launch_configs_are_bit_identical(tokens, experts):
    hidden = 7168
    x, w, dy = _inputs(tokens, hidden, experts)
    if not router_triton._gfx1250(x):
        pytest.skip("the gfx1250 launch configs")
    # the shipped configs, launched directly
    out = torch.empty(tokens, experts, dtype=torch.float32, device="cuda")
    router_triton._router_fwd_kernel[(triton.cdiv(tokens, 32),)](
        x,
        w,
        out,
        tokens,
        x.stride(0),
        w.stride(0),
        out.stride(0),
        K=hidden,
        N=experts,
        BLOCK_M=32,
        BLOCK_K=64,
        num_warps=4,
    )
    assert torch.equal(router_triton._fwd(x, w), out)
    dx = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    router_triton._router_dx_kernel[(triton.cdiv(tokens, 128), hidden // 128)](
        dy,
        w,
        dx,
        tokens,
        dy.stride(0),
        w.stride(0),
        dx.stride(0),
        N=experts,
        BLOCK_M=128,
        BLOCK_H=128,
        num_warps=4,
    )
    assert torch.equal(router_triton._dx(dy, w), dx)


def test_unsupported_shapes_are_rejected():
    x, w, _ = _inputs(64, 7168, 96)
    assert not router_triton._supported(x, w, None, torch.float32)  # not a power of two
    x, w, _ = _inputs(64, 7168, 32)
    assert not router_triton._supported(x, w, None, torch.bfloat16)  # not an FP32 router
    assert not router_triton._supported(x.float(), w, None, torch.float32)  # FP32 input
    bias = torch.zeros(32, device="cuda", dtype=torch.bfloat16)
    assert not router_triton._supported(x, w, bias, torch.float32)


def test_enabled_reads_the_argument():
    assert router_triton._enabled(_patch_context(moe_router_triton_gemm=True))
    assert not router_triton._enabled(_patch_context(moe_router_triton_gemm=False))
    assert not router_triton._enabled(_patch_context())
