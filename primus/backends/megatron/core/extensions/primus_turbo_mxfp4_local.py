# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Compile-friendly MXFP4 linear layers for Megatron local spec.

Self-contained autograd Functions that call Primus Turbo's low-level
quantize_mxfp4_dual / quantize_mxfp4 C++ ops and gemm_fp4_impl directly,
bypassing the higher-level wrappers that construct ScalingRecipe objects
and call check_mxfp4_support() global state.

Key properties:
- Uses setup_context pattern with primitive-only args so torch.compile
  can trace through without graph breaks.
- Two backward modes: pure MXFP4 or hybrid (FP4 fwd / FP8 bwd).
- gemm_fp4_impl and gemm_fp8_impl are torch.library.custom_op with register_fake.
- Zero TransformerEngine dependencies.
- Requires tensor_model_parallel_size=1, no GAF, no sequence_parallel.
"""

from typing import Tuple

import torch
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    check_mxfp4_support,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

from .primus_turbo_float8_local import _quantize_fp8_tw

# The AITER MXFP4 preshuffle fast path used to be implemented here as a
# monkey patch on GEMMFP4AITERBackend.execute. It now lives in Primus-Turbo
# (commit 683a7de, "feat(gemm): add AITER MXFP4 preshuffle fast path") and
# is opted into per-call via the gemm_fp4_impl(preshuffled=...) kwarg.
#
# Primus-Turbo PR #383 ("refactor preshuffle ...") removed the public
# enable_preshuffle() helper and moved per-call preshuffle control onto
# Float4QuantConfig.use_preshuffle. MXFP4LinearFunction passes a plain bool
# into its custom ops (not a Float4QuantConfig), so we keep the original
# runtime probe here as a module-local helper, _enable_preshuffle(), which
# reproduces the removed upstream logic verbatim.
#
# We resolve _enable_preshuffle() once per MXFP4 linear at __init__ time and
# cache the result on self._preshuffle. If the dispatcher state changes
# after model construction (env unset, set_gemm_backend(None), autotune
# flipped on) the cached flag is stale and the forward path will
# mis-dispatch -- the same caching pattern as before, just now made loud by
# the module-init contract check below (_assert_preshuffle_contract, which
# raises RuntimeError). Resolving it at __init__ (not inside forward)
# also keeps it out of the torch.compile traced region.


def _enable_preshuffle() -> bool:
    """True iff the AITER FP4 preshuffle fast path is safe.

    Requires: (1) the FP4 GEMM backend is explicitly pinned to AITER (the only
    backend that understands the shuffled layout), and (2) autotune is disabled
    (AITER opts out of tuning, so the tuner cannot select a backend for
    preshuffled inputs). Reproduces the removed Primus-Turbo enable_preshuffle()
    (pre Primus-Turbo PR #383) so MXFP4LinearFunction can keep passing a bool
    into its custom ops.
    """
    return (
        GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) == BackendType.AITER
        and not GlobalBackendManager.auto_tune_enabled()
    )


def _assert_preshuffle_contract(config, preshuffle: bool) -> None:
    """Fail loudly when MXFP4 was explicitly requested but the upstream
    AITER preshuffle fast path is unavailable.

    Pre-cleanup, the local monkey patch silently consumed preshuffled
    inputs on the AITER backend regardless of dispatcher state, masking
    misconfigurations as either a ~5% step-time regression (preshuffle
    inputs landed on AITER but every GEMM paid the shuffle cost) or, on
    dispatch paths 2/3/4, silent numerical corruption (HipBLASLt received
    preshuffled bytes and ran with them). Post-cleanup, both modes become
    a hard init-time failure with an actionable message.

    Gated on ``config.fp4 == "mxfp4"`` so any speculative-instantiation
    flow that constructs the class without actually opting into MXFP4 is
    unaffected (mirrors the precedent of the other ``__init__`` asserts
    in this file).
    """
    if getattr(config, "fp4", None) != "mxfp4":
        return
    if not preshuffle:
        raise RuntimeError(
            "MXFP4 linear requested (config.fp4='mxfp4') but the AITER "
            "preshuffle fast path is unavailable. Set "
            "PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER (or call "
            "GlobalBackendManager.set_gemm_backend(BackendType.AITER, "
            "PrecisionType.FP4)) and leave PRIMUS_TURBO_AUTO_TUNE unset. "
            "Without this, every FP4 GEMM pays ~3 extra shuffle kernel "
            "launches (~5% Flux 12B step-time regression vs the tuned "
            "baseline) -- and on non-AITER dispatch the call now raises."
        )


# ---------------------------------------------------------------------------
# torch.compile-friendly wrappers for MXFP4 C++ quantization ops.
#
# The raw C++ ops lack an Autograd dispatch key. Wrapping them with
# @torch.library.custom_op provides Autograd dispatch and register_fake
# for shape inference during tracing.
# ---------------------------------------------------------------------------

_custom_op = torch.library.custom_op

MXFP4_BLOCK_SIZE = 32
# Bumped from 16 -> 128 to match Primus-Turbo PR #335 ("feat: add quantized
# tensor support"), which made `padding_align_size` an explicit positional arg
# of quantize_mxfp4{_dual} and asserts it equals MXFP4_PADDING_ALIGN_SIZE (=128)
# in csrc/include/primus_turbo/quantization.h.
MXFP4_PADDING_ALIGN_SIZE = 128


def _cdiv(a, b):
    return (a + b - 1) // b


@_custom_op("primus::quantize_mxfp4_dual", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_dual_op(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    padding_align_size: int,
    rowwise_use_2d_block: bool,
    rowwise_use_sr: bool,
    rowwise_use_rht: bool,
    colwise_use_2d_block: bool,
    colwise_use_sr: bool,
    colwise_use_rht: bool,
    shuffle_rowwise_scale: bool,
    shuffle_rowwise: bool,
    shuffle_colwise_scale: bool,
    shuffle_colwise: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
        x,
        out_dtype,
        padding_align_size,
        rowwise_use_2d_block,
        rowwise_use_sr,
        rowwise_use_rht,
        colwise_use_2d_block,
        colwise_use_sr,
        colwise_use_rht,
        shuffle_rowwise_scale,
        shuffle_rowwise,
        shuffle_colwise_scale,
        shuffle_colwise,
    )


@_quantize_mxfp4_dual_op.register_fake
def _quantize_mxfp4_dual_fake(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    padding_align_size: int,
    rowwise_use_2d_block: bool,
    rowwise_use_sr: bool,
    rowwise_use_rht: bool,
    colwise_use_2d_block: bool,
    colwise_use_sr: bool,
    colwise_use_rht: bool,
    shuffle_rowwise_scale: bool,
    shuffle_rowwise: bool,
    shuffle_colwise_scale: bool,
    shuffle_colwise: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape
    M_pad = _cdiv(M, MXFP4_PADDING_ALIGN_SIZE) * MXFP4_PADDING_ALIGN_SIZE
    N_pad = _cdiv(N, MXFP4_PADDING_ALIGN_SIZE) * MXFP4_PADDING_ALIGN_SIZE

    if shuffle_rowwise_scale:
        rs_M = _cdiv(M, 256) * 256
        rs_N = _cdiv(_cdiv(N_pad, MXFP4_BLOCK_SIZE), 8) * 8
        rowwise_scale = torch.empty(rs_M, rs_N, dtype=torch.uint8, device=x.device)
    else:
        rowwise_scale = torch.empty(M, _cdiv(N_pad, MXFP4_BLOCK_SIZE), dtype=torch.uint8, device=x.device)

    rowwise_output = torch.empty(M, N_pad // 2, dtype=torch.uint8, device=x.device)

    if shuffle_colwise_scale:
        cs_M = _cdiv(N, 256) * 256
        cs_N = _cdiv(_cdiv(M_pad, MXFP4_BLOCK_SIZE), 8) * 8
        colwise_scale = torch.empty(cs_M, cs_N, dtype=torch.uint8, device=x.device)
    else:
        colwise_scale = torch.empty(N, _cdiv(M_pad, MXFP4_BLOCK_SIZE), dtype=torch.uint8, device=x.device)

    colwise_output = torch.empty(N, M_pad // 2, dtype=torch.uint8, device=x.device)

    return (
        rowwise_output.view(torch.float4_e2m1fn_x2),
        rowwise_scale.view(torch.float8_e8m0fnu),
        colwise_output.view(torch.float4_e2m1fn_x2),
        colwise_scale.view(torch.float8_e8m0fnu),
    )


def _quantize_mxfp4_dual_setup_context(ctx, inputs, output):
    pass


def _quantize_mxfp4_dual_backward(ctx, *grad_outputs):
    return (None,) * 13


_quantize_mxfp4_dual_op.register_autograd(
    _quantize_mxfp4_dual_backward,
    setup_context=_quantize_mxfp4_dual_setup_context,
)


# ---------------------------------------------------------------------------
# MXFP4 Autograd Function with setup_context (torch.compile friendly)
# ---------------------------------------------------------------------------


_FP4_DTYPE = torch.float4_e2m1fn_x2
_GRAN_VALUE = ScalingGranularity.MX_BLOCKWISE.value
_DEFAULT_BACKEND = BackendType.HIPBLASLT.value


def _quantize_input_dual(input_2d, preshuffle):
    """Quantize input (activation) with dual rowwise + colwise."""
    return _quantize_mxfp4_dual_op(
        input_2d,
        _FP4_DTYPE,
        MXFP4_PADDING_ALIGN_SIZE,
        False,
        False,
        False,  # rowwise: no 2d_block, no sr, no rht
        False,
        False,
        True,  # colwise: no 2d_block, no sr, yes rht
        preshuffle,
        False,  # shuffle_rowwise_scale, shuffle_rowwise
        preshuffle,
        preshuffle,  # shuffle_colwise_scale, shuffle_colwise
    )


def _quantize_weight_dual(weight, preshuffle):
    """Quantize weight with dual rowwise + colwise."""
    return _quantize_mxfp4_dual_op(
        weight,
        _FP4_DTYPE,
        MXFP4_PADDING_ALIGN_SIZE,
        True,
        False,
        False,  # rowwise: 2d_block, no sr, no rht
        True,
        False,
        False,  # colwise: 2d_block, no sr, no rht
        preshuffle,
        preshuffle,  # shuffle_rowwise_scale, shuffle_rowwise
        preshuffle,
        preshuffle,  # shuffle_colwise_scale, shuffle_colwise
    )


def _quantize_grad_dual(grad_2d, preshuffle, use_sr=True):
    """Quantize gradient (activation recipe) with dual rowwise + colwise."""
    return _quantize_mxfp4_dual_op(
        grad_2d,
        _FP4_DTYPE,
        MXFP4_PADDING_ALIGN_SIZE,
        False,
        use_sr,
        False,  # rowwise: no 2d_block, SR configurable, no rht
        False,
        use_sr,
        True,  # colwise: no 2d_block, SR configurable, yes rht
        preshuffle,
        False,  # shuffle_rowwise_scale, shuffle_rowwise
        preshuffle,
        False,  # shuffle_colwise_scale, shuffle_colwise
    )


class MXFP4LinearFunction(torch.autograd.Function):
    """MXFP4 linear (Y = X @ W^T) with MX block-of-32 scaling.

    Two modes via backward_is_fp8 bool primitive:
    - Pure MXFP4: forward + backward both use FP4 quantization + gemm_fp4_impl
    - Hybrid: forward uses FP4, backward re-quantizes saved BF16 to FP8 tensorwise

    Uses setup_context pattern with primitive-only args for torch.compile.
    """

    @staticmethod
    def forward(
        input,
        weight,
        preshuffle,
        backward_is_fp8,
        fp8_bwd_dtype,
        fp8_gran_value,
        fp8_backend_value,
        use_gradient_sr,
    ):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        a_fp4, a_scale, a_t_fp4, a_t_scale = _quantize_input_dual(input_2d, preshuffle)
        b_fp4, b_scale, b_t_fp4, b_t_scale = _quantize_weight_dual(weight, preshuffle)

        output = gemm_fp4_impl(
            a_fp4,
            a_scale,
            False,
            b_fp4,
            b_scale,
            True,
            out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_DEFAULT_BACKEND,
            preshuffled=preshuffle,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        if backward_is_fp8:
            return output, input_2d.view_as(input_2d), weight.view_as(weight)
        else:
            # Return FP4/FP8 tensors as uint8 views to avoid
            # "fill_cuda not implemented for Float4_e2m1fn_x2" when
            # the autograd engine creates zero gradients for
            # non-differentiable outputs.
            return (
                output,
                a_t_fp4.view(torch.uint8),
                a_t_scale.view(torch.uint8),
                b_t_fp4.view(torch.uint8),
                b_t_scale.view(torch.uint8),
            )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            _,
            _,
            preshuffle,
            backward_is_fp8,
            fp8_bwd_dtype,
            fp8_gran_value,
            fp8_backend_value,
            use_gradient_sr,
        ) = inputs

        ctx.preshuffle = preshuffle
        ctx.backward_is_fp8 = backward_is_fp8
        ctx.use_gradient_sr = use_gradient_sr
        ctx.out_dtype = inputs[0].dtype
        ctx.orig_shape = inputs[0].shape

        if backward_is_fp8:
            _, input_2d_saved, weight_saved = output
            ctx.save_for_backward(input_2d_saved, weight_saved)
            ctx.fp8_bwd_dtype = fp8_bwd_dtype
            ctx.fp8_gran_value = fp8_gran_value
            ctx.fp8_backend_value = fp8_backend_value
        else:
            _, a_t_u8, as_u8, b_t_u8, bs_u8 = output
            ctx.save_for_backward(
                a_t_u8.view(_FP4_DTYPE),
                as_u8.view(torch.float8_e8m0fnu),
                b_t_u8.view(_FP4_DTYPE),
                bs_u8.view(torch.float8_e8m0fnu),
            )
            ctx.mark_non_differentiable(a_t_u8, as_u8, b_t_u8, bs_u8)

    @staticmethod
    def backward(ctx, grad_output, *_):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.backward_is_fp8:
            input_2d, weight = ctx.saved_tensors

            grad_fp8, grad_scale_inv = _quantize_fp8_tw(grad_2d, ctx.fp8_bwd_dtype)
            a_fp8, a_scale_inv = _quantize_fp8_tw(input_2d, ctx.fp8_bwd_dtype)
            b_fp8, b_scale_inv = _quantize_fp8_tw(weight, ctx.fp8_bwd_dtype)

            grad_input = gemm_fp8_impl(
                grad_fp8,
                grad_scale_inv,
                False,
                b_fp8,
                b_scale_inv,
                False,
                ctx.out_dtype,
                False,
                granularity=ctx.fp8_gran_value,
                default_backend=ctx.fp8_backend_value,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            grad_weight = gemm_fp8_impl(
                a_fp8,
                a_scale_inv,
                True,
                grad_fp8,
                grad_scale_inv,
                False,
                ctx.out_dtype,
                True,
                granularity=ctx.fp8_gran_value,
                default_backend=ctx.fp8_backend_value,
            )
        else:
            a_t_fp4, a_t_scale, b_t_fp4, b_t_scale = ctx.saved_tensors
            preshuffle = ctx.preshuffle

            g_fp4, g_scale, g_t_fp4, g_t_scale = _quantize_grad_dual(
                grad_2d, preshuffle, use_sr=ctx.use_gradient_sr
            )

            grad_input = gemm_fp4_impl(
                g_fp4,
                g_scale,
                False,
                b_t_fp4,
                b_t_scale,
                True,
                ctx.out_dtype,
                False,
                granularity=_GRAN_VALUE,
                default_backend=_DEFAULT_BACKEND,
                preshuffled=preshuffle,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            grad_weight = gemm_fp4_impl(
                g_t_fp4,
                g_t_scale,
                False,
                a_t_fp4,
                a_t_scale,
                True,
                ctx.out_dtype,
                False,
                granularity=_GRAN_VALUE,
                default_backend=_DEFAULT_BACKEND,
                preshuffled=preshuffle,
            )

        return grad_input, grad_weight, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# MXFP4-aware parallel linear layers
# ---------------------------------------------------------------------------


class MXFP4ColumnParallelLinear(ColumnParallelLinear):
    """ColumnParallelLinear with per-module MXFP4. torch.compile friendly.

    Requires: tensor_model_parallel_size=1, gradient_accumulation_fusion=False,
    sequence_parallel=False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.tensor_model_parallel_size != 1:
            raise ValueError(
                "MXFP4ColumnParallelLinear requires tensor_model_parallel_size=1. "
                f"Got {self.config.tensor_model_parallel_size}."
            )
        if self.gradient_accumulation_fusion:
            raise ValueError("MXFP4ColumnParallelLinear requires gradient_accumulation_fusion=False.")
        if self.sequence_parallel:
            raise ValueError("MXFP4ColumnParallelLinear requires sequence_parallel=False.")

        supported, reason = check_mxfp4_support()
        if not supported:
            raise RuntimeError(f"MXFP4 not supported on this device: {reason}")

        self._preshuffle = _enable_preshuffle()
        _assert_preshuffle_contract(self.config, self._preshuffle)
        self._backward_is_fp8 = getattr(self.config, "mxfp4_backward_precision", "mxfp4") == "fp8"
        self._use_gradient_sr = getattr(self.config, "mxfp4_gradient_stochastic_rounding", False)

        if self._backward_is_fp8:
            from primus_turbo.pytorch.core.low_precision import float8_e5m2

            self._fp8_bwd_dtype = float8_e5m2
            self._fp8_gran_value = ScalingGranularity.TENSORWISE.value
            self._fp8_backend_value = BackendType.HIPBLASLT.value
        else:
            self._fp8_bwd_dtype = None
            self._fp8_gran_value = 0
            self._fp8_backend_value = 0

    def _forward_impl(self, input, weight, *args, **kwargs):
        bias = kwargs.get("bias", None)

        result = MXFP4LinearFunction.apply(
            input,
            weight,
            self._preshuffle,
            self._backward_is_fp8,
            self._fp8_bwd_dtype,
            self._fp8_gran_value,
            self._fp8_backend_value,
            self._use_gradient_sr,
        )
        output = result[0]

        if bias is not None:
            output = output + bias
        return output


class MXFP4RowParallelLinear(RowParallelLinear):
    """RowParallelLinear with per-module MXFP4. torch.compile friendly.

    Requires: tensor_model_parallel_size=1, gradient_accumulation_fusion=False,
    sequence_parallel=False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.tensor_model_parallel_size != 1:
            raise ValueError(
                "MXFP4RowParallelLinear requires tensor_model_parallel_size=1. "
                f"Got {self.config.tensor_model_parallel_size}."
            )
        if self.gradient_accumulation_fusion:
            raise ValueError("MXFP4RowParallelLinear requires gradient_accumulation_fusion=False.")
        if self.sequence_parallel:
            raise ValueError("MXFP4RowParallelLinear requires sequence_parallel=False.")

        supported, reason = check_mxfp4_support()
        if not supported:
            raise RuntimeError(f"MXFP4 not supported on this device: {reason}")

        self._preshuffle = _enable_preshuffle()
        _assert_preshuffle_contract(self.config, self._preshuffle)
        self._backward_is_fp8 = getattr(self.config, "mxfp4_backward_precision", "mxfp4") == "fp8"
        self._use_gradient_sr = getattr(self.config, "mxfp4_gradient_stochastic_rounding", False)

        if self._backward_is_fp8:
            from primus_turbo.pytorch.core.low_precision import float8_e5m2

            self._fp8_bwd_dtype = float8_e5m2
            self._fp8_gran_value = ScalingGranularity.TENSORWISE.value
            self._fp8_backend_value = BackendType.HIPBLASLT.value
        else:
            self._fp8_bwd_dtype = None
            self._fp8_gran_value = 0
            self._fp8_backend_value = 0

    def _forward_impl(self, input, weight, *args, **kwargs):
        bias = kwargs.get("bias", None)

        result = MXFP4LinearFunction.apply(
            input,
            weight,
            self._preshuffle,
            self._backward_is_fp8,
            self._fp8_bwd_dtype,
            self._fp8_gran_value,
            self._fp8_backend_value,
            self._use_gradient_sr,
        )
        output = result[0]

        if bias is not None:
            output = output + bias
        return output
