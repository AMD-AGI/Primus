# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Compile-friendly MXFP6 (E2M3) linear layers for Megatron local spec.

The MXFP4 sibling of this module (``primus_turbo_mxfp4_local``) is the template, and
this one is deliberately smaller, because MXFP6 removes most of MXFP4's configuration
surface rather than because anything is left unfinished:

- **No preshuffle contract.** The A6W6 kernels read AITER's packed C0/C1 tile blob
  directly, so there is no unshuffled layout and no fast path to opt into. MXFP4's
  ``_enable_preshuffle`` / ``_assert_preshuffle_contract`` dance, and the whole class of
  misconfiguration it guards against, simply does not exist here.
- **No ScalingRecipe flags.** The 32-point Hadamard rotation is mandatory and fused into
  the packer (the GEMM depends on it cancelling between the two operands), scaling is
  strictly per-1x32 along the contraction axis so ``use_2d_block`` is meaningless, and
  stochastic rounding is not implemented. MXFP4 threads twelve booleans through its
  quantize op; MXFP6 has none to thread.
- **No local custom-op registration.** Primus-Turbo already exposes
  ``primus_turbo::quantize_mxfp6_dual_impl`` as a ``torch.library.custom_op`` with a
  correct fake, so unlike MXFP4 there is nothing to re-wrap in order to bypass
  recipe construction.

Retained from the MXFP4 design: the ``setup_context`` pattern with primitive-only
arguments so ``torch.compile`` traces without graph breaks, the two backward modes
(pure MXFP6, or hybrid MXFP6-forward / FP8-backward), and zero TransformerEngine
dependencies.

Shape constraint worth knowing: MXFP6 needs the linear's M, N **and** K to be multiples
of 256. K is included because the backward GEMMs use it as an output dimension. This is
enforced inside Primus-Turbo's ``gemm_fp6``; here it means a hidden size or sequence
length that is only 128-aligned will be rejected at the first forward.
"""

import torch
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import check_mxfp6_support

from .primus_turbo_float8_local import _quantize_fp8_tw

_GRAN_VALUE = ScalingGranularity.MX_BLOCKWISE.value

# Registered by Primus-Turbo with a pure-arithmetic fake, so it is safe to trace.
_quantize_mxfp6_dual = torch.ops.primus_turbo.quantize_mxfp6_dual_impl


class MXFP6LinearFunction(torch.autograd.Function):
    """MXFP6 linear (Y = X @ W^T) with MX block-of-32 scaling along the contraction axis.

    Two modes via the ``backward_is_fp8`` bool primitive:

    - Pure MXFP6: forward and backward both quantize to MXFP6 and call gemm_fp6_impl.
    - Hybrid: forward is MXFP6, backward re-quantizes the saved BF16 to tensorwise FP8.

    In the pure path the forward returns the column-direction blobs as extra outputs so
    ``setup_context`` can save them; they are already uint8, so unlike MXFP4 there is no
    dtype-view juggling needed to keep the autograd engine from trying to allocate zero
    gradients in an unsupported dtype.
    """

    @staticmethod
    def forward(
        input,
        weight,
        backward_is_fp8,
        fp8_bwd_dtype,
        fp8_gran_value,
        fp8_backend_value,
    ):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        m, k = input_2d.shape
        n = weight.shape[0]

        a_row, a_row_scale, a_col, a_col_scale = _quantize_mxfp6_dual(input_2d)
        b_row, b_row_scale, b_col, b_col_scale = _quantize_mxfp6_dual(weight)

        output = gemm_fp6_impl(
            a_row,
            a_row_scale,
            b_row,
            b_row_scale,
            m,
            n,
            k,
            out_dtype,
            _GRAN_VALUE,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        if backward_is_fp8:
            return output, input_2d.view_as(input_2d), weight.view_as(weight)
        return output, a_col, a_col_scale, b_col, b_col_scale

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            input,
            weight,
            backward_is_fp8,
            fp8_bwd_dtype,
            fp8_gran_value,
            fp8_backend_value,
        ) = inputs

        ctx.backward_is_fp8 = backward_is_fp8
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        # The packed blobs carry no shape, so the logical dims have to be saved too.
        ctx.m = input.numel() // input.shape[-1]
        ctx.k = input.shape[-1]
        ctx.n = weight.shape[0]

        if backward_is_fp8:
            _, input_2d_saved, weight_saved = output
            ctx.save_for_backward(input_2d_saved, weight_saved)
            ctx.fp8_bwd_dtype = fp8_bwd_dtype
            ctx.fp8_gran_value = fp8_gran_value
            ctx.fp8_backend_value = fp8_backend_value
        else:
            _, a_col, a_col_scale, b_col, b_col_scale = output
            ctx.save_for_backward(a_col, a_col_scale, b_col, b_col_scale)
            ctx.mark_non_differentiable(a_col, a_col_scale, b_col, b_col_scale)

    @staticmethod
    def backward(ctx, grad_output, *_):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        m, n, k = ctx.m, ctx.n, ctx.k

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
            a_col, a_col_scale, b_col, b_col_scale = ctx.saved_tensors
            g_row, g_row_scale, g_col, g_col_scale = _quantize_mxfp6_dual(grad_2d)

            # grad_input[M, K] = grad[M, N] @ weight[N, K], contracting N. b_col is the
            # weight packed along N, i.e. logically [K, N] contracting N.
            grad_input = gemm_fp6_impl(
                g_row,
                g_row_scale,
                b_col,
                b_col_scale,
                m,
                k,
                n,
                ctx.out_dtype,
                _GRAN_VALUE,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            # grad_weight[N, K] = grad.T[N, M] @ input[M, K], contracting M.
            grad_weight = gemm_fp6_impl(
                g_col,
                g_col_scale,
                a_col,
                a_col_scale,
                n,
                k,
                m,
                ctx.out_dtype,
                _GRAN_VALUE,
            )

        return grad_input, grad_weight, None, None, None, None


def _init_mxfp6_linear(module) -> None:
    """Shared __init__ tail for both MXFP6 parallel linears.

    MXFP4 duplicates this block between its column and row classes; there is no reason
    for the MXFP6 copy to inherit the duplication.
    """
    name = type(module).__name__

    if module.config.tensor_model_parallel_size != 1:
        raise ValueError(
            f"{name} requires tensor_model_parallel_size=1. "
            f"Got {module.config.tensor_model_parallel_size}."
        )
    if module.gradient_accumulation_fusion:
        # The A6W6 entry point has no beta=1 accumulate epilogue, so wgrad cannot write
        # main_grad in place.
        raise ValueError(f"{name} requires gradient_accumulation_fusion=False.")
    if module.sequence_parallel:
        raise ValueError(f"{name} requires sequence_parallel=False.")

    supported, reason = check_mxfp6_support()
    if not supported:
        raise RuntimeError(f"MXFP6 not supported on this device: {reason}")

    module._backward_is_fp8 = getattr(module.config, "mxfp6_backward_precision", "mxfp6") == "fp8"

    if module._backward_is_fp8:
        from primus_turbo.pytorch.core.low_precision import float8_e5m2

        module._fp8_bwd_dtype = float8_e5m2
        module._fp8_gran_value = ScalingGranularity.TENSORWISE.value
        module._fp8_backend_value = BackendType.HIPBLASLT.value
    else:
        module._fp8_bwd_dtype = None
        module._fp8_gran_value = 0
        module._fp8_backend_value = 0


def _mxfp6_forward_impl(module, input, weight, **kwargs):
    bias = kwargs.get("bias", None)

    result = MXFP6LinearFunction.apply(
        input,
        weight,
        module._backward_is_fp8,
        module._fp8_bwd_dtype,
        module._fp8_gran_value,
        module._fp8_backend_value,
    )
    output = result[0]

    # Bias is added outside the GEMM: the A6W6 entry point has no bias epilogue.
    if bias is not None:
        output = output + bias
    return output


class MXFP6ColumnParallelLinear(ColumnParallelLinear):
    """ColumnParallelLinear with per-module MXFP6. torch.compile friendly.

    Requires: tensor_model_parallel_size=1, gradient_accumulation_fusion=False,
    sequence_parallel=False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _init_mxfp6_linear(self)

    def _forward_impl(self, input, weight, *args, **kwargs):
        return _mxfp6_forward_impl(self, input, weight, **kwargs)


class MXFP6RowParallelLinear(RowParallelLinear):
    """RowParallelLinear with per-module MXFP6. torch.compile friendly.

    Requires: tensor_model_parallel_size=1, gradient_accumulation_fusion=False,
    sequence_parallel=False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _init_mxfp6_linear(self)

    def _forward_impl(self, input, weight, *args, **kwargs):
        return _mxfp6_forward_impl(self, input, weight, **kwargs)
