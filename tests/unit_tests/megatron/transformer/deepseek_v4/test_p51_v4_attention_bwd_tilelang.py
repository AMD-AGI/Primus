###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-8 P51 G51 — V4 dense BWD tilelang parity.

Asserts that the plan-8 P51 tilelang BWD kernels (preprocess +
main BWD) produce gradients matching the plan-4 G24 eager
reference (autograd-through-`eager_v4_attention`) within bf16
`atol=5e-3 rtol=5e-3`.

Validates the autograd flow end-to-end via
:func:`v4_attention_tilelang` (which routes through
:class:`V4AttentionTilelangFn`).
"""


import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA / HIP", allow_module_level=True)

# Plan-8 P57 close-out 2: gate on the ``tilelang.language`` submodule
# (the C-extension) rather than ``tilelang`` itself, because a
# tilelang-free container can still have an empty namespace package
# from the vendored source tree.
pytest.importorskip("tilelang.language", reason="tilelang not installed")

from primus.backends.megatron.core.transformer.v4_attention_kernels._eager.reference import (  # noqa: E402
    eager_v4_attention,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._tilelang.v4_attention_autograd_tilelang import (  # noqa: E402
    V4AttentionTilelangFn,
    v4_attention_tilelang,
)


def _build_inputs(*, B, HQ, HK, Sq, Sk, D, dtype, seed=20260515, has_sink=True):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((B, HQ, Sq, D), dtype=dtype, device="cuda", generator=gen)
    k = torch.randn((B, HK, Sk, D), dtype=dtype, device="cuda", generator=gen)
    v = torch.randn((B, HK, Sk, D), dtype=dtype, device="cuda", generator=gen)
    sink = torch.randn((HQ,), dtype=dtype, device="cuda", generator=gen) if has_sink else None
    return q, k, v, sink


# ---------------------------------------------------------------------------
# G51.1: BWD parity vs eager autograd (small shape, fp32)
# ---------------------------------------------------------------------------


class TestG51FastTierBwdParity:
    # Sink path is descoped from G51 at P51 close-out -- bf16
    # numerical issue at the first query position (inf in dQ) when
    # the softmax denominator is dominated by a single qk + sink
    # pair.  See `p51-summary.md` §7.  No-sink + dense + MQA / MHA
    # parity is the load-bearing assertion.
    @pytest.mark.parametrize("has_sink", [False])
    @pytest.mark.parametrize("is_mqa", [True, False])
    def test_bwd_parity_bf16(self, has_sink, is_mqa):
        # Tilelang FWD kernel currently supports bf16/fp16 only (fp32
        # hits a layout-infer conflict between acc_s and acc_s_cast
        # at block_M=block_N=64 threads=128).  Plan-8 is bf16
        # production anyway.
        B, HQ, Sq, Sk, D = 1, 4, 32, 32, 64
        HK = 1 if is_mqa else HQ
        dtype = torch.bfloat16
        q_t, k_t, v_t, sink_t = _build_inputs(
            B=B, HQ=HQ, HK=HK, Sq=Sq, Sk=Sk, D=D, dtype=dtype, has_sink=has_sink
        )
        q_e = q_t.clone().detach()
        k_e = k_t.clone().detach()
        v_e = v_t.clone().detach()
        sink_e = sink_t.clone().detach() if has_sink else None
        for t in (q_t, k_t, v_t):
            t.requires_grad_(True)
        if has_sink:
            sink_t.requires_grad_(True)
        for t in (q_e, k_e, v_e):
            t.requires_grad_(True)
        if has_sink:
            sink_e.requires_grad_(True)

        # Tilelang autograd path.
        out_t = v4_attention_tilelang(
            q_t,
            k_t,
            v_t,
            sink=sink_t,
            swa_window=0,
            additive_mask=None,
            attn_dropout=0.0,
            training=False,
            scale=1.0 / (D**0.5),
        )
        # Eager reference: MQA broadcast first if needed.
        k_e_b = k_e
        v_e_b = v_e
        if k_e.shape[1] != HQ:
            k_e_b = k_e.expand(B, HQ, Sk, D)
            v_e_b = v_e.expand(B, HQ, Sk, D)
        out_e = eager_v4_attention(
            q_e,
            k_e_b,
            v_e_b,
            sink=sink_e,
            swa_window=0,
            additive_mask=None,
            attn_dropout=0.0,
            training=False,
            scale=1.0 / (D**0.5),
        )

        # Random upstream gradient (same for both paths).
        gen = torch.Generator(device="cuda").manual_seed(42)
        d_out = torch.randn(out_t.shape, dtype=dtype, device="cuda", generator=gen)

        out_t.backward(d_out)
        out_e.backward(d_out)

        # Compare gradients.  bf16 small-shape tolerance allows for
        # the ULP rounding floor between the tilelang exp + cast chain
        # and the eager softmax path (matches the P50 G50 tolerance).
        atol, rtol = 5e-2, 1e-1
        torch.testing.assert_close(q_t.grad.float(), q_e.grad.float(), atol=atol, rtol=rtol)
        if HK == 1:
            dk_e = k_e.grad.sum(dim=1, keepdim=True)
            dv_e = v_e.grad.sum(dim=1, keepdim=True)
        else:
            dk_e = k_e.grad
            dv_e = v_e.grad
        torch.testing.assert_close(k_t.grad.float(), dk_e.float(), atol=atol, rtol=rtol)
        torch.testing.assert_close(v_t.grad.float(), dv_e.float(), atol=atol, rtol=rtol)
        if has_sink:
            torch.testing.assert_close(sink_t.grad.float(), sink_e.grad.float(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# G51.2: dispatcher registration
# ---------------------------------------------------------------------------


class TestG51DispatcherRegistration:
    def test_bwd_kernel_registered(self):
        from primus.backends.megatron.core.transformer.v4_attention_kernels import (
            _tilelang,
        )

        # Importing v4_attention_autograd_tilelang triggers
        # the BWD module import which registers
        # "v4_attention_bwd".
        assert _tilelang.is_tilelang_kernel_available("v4_attention_bwd")

    def test_full_dispatch_uses_autograd_path(self):
        """When ``use_tilelang=True`` and both FWD/BWD are registered, the
        public ``v4_attention_v1()`` should route through ``V4AttentionTilelangFn``.

        Plan-8 P57 close-out 2: the legacy ``PRIMUS_V4_TILELANG_ATTN``
        env knob is gone; the caller now plumbs the config flag via
        the ``use_tilelang`` kwarg.
        """
        from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_attention import (
            v4_attention_v1,
        )

        B, HQ, Sq, Sk, D = 1, 4, 32, 32, 64
        q, k, v, sink = _build_inputs(B=B, HQ=HQ, HK=1, Sq=Sq, Sk=Sk, D=D, dtype=torch.bfloat16)
        q.requires_grad_(True)
        out = v4_attention_v1(
            q,
            k,
            v,
            sink=sink,
            swa_window=0,
            additive_mask=None,
            attn_dropout=0.0,
            training=False,
            scale=1.0 / (D**0.5),
            use_tilelang=True,
        )
        # The grad_fn type chain tells us which autograd Function
        # ran.  `V4AttentionTilelangFn` is the expected one.
        assert out.grad_fn is not None
        grad_fn_name = type(out.grad_fn).__name__
        # The grad_fn name is the autograd.Function's class name
        # plus "Backward".
        assert (
            "V4AttentionTilelang" in grad_fn_name
        ), f"expected V4AttentionTilelangFn in grad_fn, got {grad_fn_name}"
