# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Compile-friendly Primus-Turbo MXFP4 linear kernels for FLUX."""

import os
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._prims_common import suggest_memory_format
import primus_turbo.pytorch.core.quantized_tensor  # Registers transpose_2d.
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.ops.quantization import dequantize_fp8

_MXFP4_BLOCK_SIZE = 32
_MXFP4_PADDING_ALIGN_SIZE = 128
_FP4_DTYPE = torch.float4_e2m1fn_x2
_MXFP8_DTYPE = torch.float8_e4m3fn
_GRANULARITY = ScalingGranularity.MX_BLOCKWISE.value
_FP4_BACKEND = BackendType.HIPBLASLT.value


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@torch.library.custom_op("primus::quantize_mxfp4_dual", mutates_args=(), device_types="cuda")
def _quantize_mxfp4_dual(
    x: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
        x,
        _FP4_DTYPE,
        _MXFP4_PADDING_ALIGN_SIZE,
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


@_quantize_mxfp4_dual.register_fake
def _quantize_mxfp4_dual_fake(
    x: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        rowwise_use_2d_block,
        rowwise_use_sr,
        rowwise_use_rht,
        colwise_use_2d_block,
        colwise_use_sr,
        colwise_use_rht,
        shuffle_rowwise,
        shuffle_colwise,
    )
    rows, columns = x.shape
    rows_padded = _cdiv(rows, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    columns_padded = _cdiv(columns, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    rowwise_scale_rows = _cdiv(rows, 256) * 256 if shuffle_rowwise_scale else rows
    rowwise_scale_columns = _cdiv(_cdiv(columns_padded, _MXFP4_BLOCK_SIZE), 8) * 8
    if not shuffle_rowwise_scale:
        rowwise_scale_columns = _cdiv(columns_padded, _MXFP4_BLOCK_SIZE)
    colwise_scale_rows = _cdiv(columns, 256) * 256 if shuffle_colwise_scale else columns
    colwise_scale_columns = _cdiv(_cdiv(rows_padded, _MXFP4_BLOCK_SIZE), 8) * 8
    if not shuffle_colwise_scale:
        colwise_scale_columns = _cdiv(rows_padded, _MXFP4_BLOCK_SIZE)
    return (
        torch.empty(rows, columns_padded // 2, dtype=torch.uint8, device=x.device).view(_FP4_DTYPE),
        torch.empty(
            rowwise_scale_rows,
            rowwise_scale_columns,
            dtype=torch.uint8,
            device=x.device,
        ).view(torch.float8_e8m0fnu),
        torch.empty(columns, rows_padded // 2, dtype=torch.uint8, device=x.device).view(_FP4_DTYPE),
        torch.empty(
            colwise_scale_rows,
            colwise_scale_columns,
            dtype=torch.uint8,
            device=x.device,
        ).view(torch.float8_e8m0fnu),
    )


_quantize_mxfp4_dual.register_autograd(
    lambda ctx, *grads: (None,) * 11,
    setup_context=lambda ctx, inputs, output: None,
)


@torch.library.custom_op(
    "primus::quantize_mxfp4_colwise", mutates_args=(), device_types="cuda"
)
def _quantize_mxfp4_colwise(
    x: torch.Tensor,
    use_2d_block: bool,
    use_rht: bool,
    shuffle_scale: bool,
    shuffle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp4(
        x,
        _FP4_DTYPE,
        0,
        _MXFP4_PADDING_ALIGN_SIZE,
        use_2d_block,
        False,
        use_rht,
        shuffle_scale,
        shuffle,
    )


@_quantize_mxfp4_colwise.register_fake
def _quantize_mxfp4_colwise_fake(
    x: torch.Tensor,
    use_2d_block: bool,
    use_rht: bool,
    shuffle_scale: bool,
    shuffle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del use_2d_block, use_rht, shuffle
    rows, columns = x.shape
    rows_padded = _cdiv(rows, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    scale_rows = _cdiv(columns, 256) * 256 if shuffle_scale else columns
    scale_columns = _cdiv(rows_padded, _MXFP4_BLOCK_SIZE)
    if shuffle_scale:
        scale_columns = _cdiv(scale_columns, 8) * 8
    return (
        torch.empty(columns, rows_padded // 2, dtype=torch.uint8, device=x.device).view(_FP4_DTYPE),
        torch.empty(scale_rows, scale_columns, dtype=torch.uint8, device=x.device).view(
            torch.float8_e8m0fnu
        ),
    )


_quantize_mxfp4_colwise.register_autograd(
    lambda ctx, *grads: (None,) * 5,
    setup_context=lambda ctx, inputs, output: None,
)


@torch.library.custom_op(
    "primus::quantize_mxfp8_rowwise_mxfp4_colwise",
    mutates_args=(),
    device_types="cuda",
)
def _quantize_mxfp8_rowwise_mxfp4_colwise(
    x: torch.Tensor,
    rowwise_fp8_use_2d_block: bool,
    colwise_fp4_use_2d_block: bool,
    colwise_fp4_use_rht: bool,
    shuffle_colwise_scale: bool,
    shuffle_colwise: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8_rowwise_mxfp4_colwise(
        x,
        rowwise_fp8_use_2d_block,
        colwise_fp4_use_2d_block,
        colwise_fp4_use_rht,
        shuffle_colwise_scale,
        shuffle_colwise,
    )


@_quantize_mxfp8_rowwise_mxfp4_colwise.register_fake
def _quantize_mxfp8_rowwise_mxfp4_colwise_fake(
    x: torch.Tensor,
    rowwise_fp8_use_2d_block: bool,
    colwise_fp4_use_2d_block: bool,
    colwise_fp4_use_rht: bool,
    shuffle_colwise_scale: bool,
    shuffle_colwise: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        rowwise_fp8_use_2d_block,
        colwise_fp4_use_2d_block,
        colwise_fp4_use_rht,
        shuffle_colwise,
    )
    rows, columns = x.shape
    rows_padded = _cdiv(rows, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    columns_padded = _cdiv(columns, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    colwise_scale_rows = _cdiv(columns, 256) * 256 if shuffle_colwise_scale else columns
    colwise_scale_columns = _cdiv(rows_padded, _MXFP4_BLOCK_SIZE)
    if shuffle_colwise_scale:
        colwise_scale_columns = _cdiv(colwise_scale_columns, 8) * 8
    return (
        torch.empty(rows, columns_padded, dtype=_MXFP8_DTYPE, device=x.device),
        torch.empty(
            rows,
            _cdiv(columns_padded, _MXFP4_BLOCK_SIZE),
            dtype=torch.float8_e8m0fnu,
            device=x.device,
        ),
        torch.empty(columns, rows_padded // 2, dtype=torch.uint8, device=x.device).view(_FP4_DTYPE),
        torch.empty(
            colwise_scale_rows,
            colwise_scale_columns,
            dtype=torch.uint8,
            device=x.device,
        ).view(torch.float8_e8m0fnu),
    )


_quantize_mxfp8_rowwise_mxfp4_colwise.register_autograd(
    lambda ctx, *grads: (None,) * 6,
    setup_context=lambda ctx, inputs, output: None,
)


@torch.library.custom_op("primus::quantize_mxfp8_dual", mutates_args=(), device_types="cuda")
def _quantize_mxfp8_dual(
    x: torch.Tensor, rowwise_use_2d_block: bool, colwise_use_2d_block: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8_dual(
        x,
        _MXFP8_DTYPE,
        _MXFP4_PADDING_ALIGN_SIZE,
        rowwise_use_2d_block,
        colwise_use_2d_block,
        False,
        False,
        False,
        False,
    )


@_quantize_mxfp8_dual.register_fake
def _quantize_mxfp8_dual_fake(
    x: torch.Tensor, rowwise_use_2d_block: bool, colwise_use_2d_block: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del rowwise_use_2d_block, colwise_use_2d_block
    rows, columns = x.shape
    rows_padded = _cdiv(rows, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    columns_padded = _cdiv(columns, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    return (
        torch.empty(rows, columns_padded, dtype=_MXFP8_DTYPE, device=x.device),
        torch.empty(rows, _cdiv(columns_padded, _MXFP4_BLOCK_SIZE), dtype=torch.float8_e8m0fnu, device=x.device),
        torch.empty(columns, rows_padded, dtype=_MXFP8_DTYPE, device=x.device),
        torch.empty(columns, _cdiv(rows_padded, _MXFP4_BLOCK_SIZE), dtype=torch.float8_e8m0fnu, device=x.device),
    )


_quantize_mxfp8_dual.register_autograd(
    lambda ctx, *grads: (None,) * 3,
    setup_context=lambda ctx, inputs, output: None,
)


@torch.library.custom_op("primus::quantize_mxfp8_rowwise", mutates_args=(), device_types="cuda")
def _quantize_mxfp8_rowwise(
    x: torch.Tensor, use_2d_block: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_mxfp8(
        x, _MXFP8_DTYPE, 1, _MXFP4_PADDING_ALIGN_SIZE, use_2d_block, False, False
    )


@_quantize_mxfp8_rowwise.register_fake
def _quantize_mxfp8_rowwise_fake(
    x: torch.Tensor, use_2d_block: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    del use_2d_block
    rows, columns = x.shape
    columns_padded = _cdiv(columns, _MXFP4_PADDING_ALIGN_SIZE) * _MXFP4_PADDING_ALIGN_SIZE
    return (
        torch.empty(rows, columns_padded, dtype=_MXFP8_DTYPE, device=x.device),
        torch.empty(
            rows,
            _cdiv(columns_padded, _MXFP4_BLOCK_SIZE),
            dtype=torch.float8_e8m0fnu,
            device=x.device,
        ),
    )


_quantize_mxfp8_rowwise.register_autograd(
    lambda ctx, *grads: (None, None),
    setup_context=lambda ctx, inputs, output: None,
)


@torch.library.custom_op("primus::dequantize_mxfp8_rowwise", mutates_args=(), device_types="cuda")
def _dequantize_mxfp8_rowwise(
    x: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    return dequantize_fp8(
        x,
        out_dtype,
        ScalingGranularity.MX_BLOCKWISE,
        block_size=_MXFP4_BLOCK_SIZE,
        axis=1,
        scale_inv=scale,
    )


@_dequantize_mxfp8_rowwise.register_fake
def _dequantize_mxfp8_rowwise_fake(
    x: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    del scale
    return torch.empty(x.shape, dtype=out_dtype, device=x.device)


_dequantize_mxfp8_rowwise.register_autograd(
    lambda ctx, *grads: (None, None, None),
    setup_context=lambda ctx, inputs, output: None,
)


@torch.library.custom_op("primus::dequantize_mxfp4_rowwise", mutates_args=(), device_types="cuda")
def _dequantize_mxfp4_rowwise(
    x: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    return torch.ops.primus_turbo_cpp_extension.dequantize_mxfp4(
        x, scale, 1, _MXFP4_BLOCK_SIZE, out_dtype
    )


@_dequantize_mxfp4_rowwise.register_fake
def _dequantize_mxfp4_rowwise_fake(
    x: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    del scale
    return torch.empty(x.shape[0], x.shape[1] * 2, dtype=out_dtype, device=x.device)


_dequantize_mxfp4_rowwise.register_autograd(
    lambda ctx, *grads: (None, None, None),
    setup_context=lambda ctx, inputs, output: None,
)


def _quantize_input(
    input_2d: torch.Tensor, preshuffle: bool, forward_rht: bool = False
):
    return _quantize_mxfp4_dual(
        input_2d,
        False,
        False,
        forward_rht,
        False,
        False,
        True,
        preshuffle,
        False,
        preshuffle,
        preshuffle,
    )


def _quantize_weight(
    weight: torch.Tensor, preshuffle: bool, forward_rht: bool = False
):
    return _quantize_mxfp4_dual(
        weight,
        True,
        False,
        forward_rht,
        True,
        False,
        False,
        preshuffle,
        preshuffle,
        preshuffle,
        preshuffle,
    )


def _quantize_gradient(gradient_2d: torch.Tensor, preshuffle: bool, use_sr: bool):
    return _quantize_mxfp4_dual(
        gradient_2d,
        False,
        use_sr,
        False,
        False,
        use_sr,
        True,
        preshuffle,
        False,
        preshuffle,
        False,
    )


def _gemm_fp4(a, a_scale, trans_a, b, b_scale, trans_b, out_dtype, preshuffle):
    return gemm_fp4_impl(
        a,
        a_scale,
        trans_a,
        b,
        b_scale,
        trans_b,
        out_dtype,
        False,
        granularity=_GRANULARITY,
        default_backend=_FP4_BACKEND,
        preshuffled=preshuffle,
    )


class _MXFP4Forward(torch.autograd.Function):
    @staticmethod
    def forward(
        input,
        weight,
        preshuffle,
        use_gradient_sr,
        use_forward_rht,
        activation_residual_mode,
    ):
        input_2d = input.reshape(-1, input.shape[-1])
        a, a_scale, a_t, a_t_scale = _quantize_input(
            input_2d, preshuffle, use_forward_rht
        )
        b, b_scale, b_t, b_t_scale = _quantize_weight(
            weight, preshuffle, use_forward_rht
        )
        output = _gemm_fp4(a, a_scale, False, b, b_scale, True, input.dtype, preshuffle)
        if activation_residual_mode:
            q_input, q_scale, _, _ = _quantize_input(input_2d, False)
            q_input = _dequantize_mxfp4_rowwise(q_input, q_scale, input.dtype)
            residual = input_2d - q_input
            if activation_residual_mode == 1:
                correction = torch.nn.functional.linear(residual, weight)
            elif activation_residual_mode == 2:
                r, r_scale = _quantize_mxfp8_rowwise(residual, False)
                w, w_scale = _quantize_mxfp8_rowwise(weight, True)
                correction = gemm_fp8_impl(
                    r,
                    r_scale,
                    False,
                    w,
                    w_scale,
                    True,
                    input.dtype,
                    False,
                    granularity=_GRANULARITY,
                    default_backend=BackendType.FLYDSL.value,
                )
            else:
                r, r_scale, _, _ = _quantize_input(residual, preshuffle)
                correction = _gemm_fp4(
                    r, r_scale, False, b, b_scale, True, input.dtype, preshuffle
                )
            output = output + correction
        return (
            output,
            a_t.view(torch.uint8),
            a_t_scale.view(torch.uint8),
            b_t.view(torch.uint8),
            b_t_scale.view(torch.uint8),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, _, preshuffle, use_gradient_sr, _, _ = inputs
        ctx.preshuffle = preshuffle
        ctx.use_gradient_sr = use_gradient_sr
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        _, a_t, a_t_scale, b_t, b_t_scale = output
        ctx.save_for_backward(
            a_t.view(_FP4_DTYPE),
            a_t_scale.view(torch.float8_e8m0fnu),
            b_t.view(_FP4_DTYPE),
            b_t_scale.view(torch.float8_e8m0fnu),
        )
        ctx.mark_non_differentiable(a_t, a_t_scale, b_t, b_t_scale)

    @staticmethod
    def backward(ctx, grad_output, *_):
        grad_input, grad_weight = _mxfp4_backward(
            ctx, grad_output, *ctx.saved_tensors
        )
        return grad_input, grad_weight, None, None, None, None


def _mxfp4_backward(ctx, grad_output, a_t, a_t_scale, b_t, b_t_scale):
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    gradient_2d = grad_output.reshape(-1, grad_output.shape[-1])
    g, g_scale, g_t, g_t_scale = _quantize_gradient(
        gradient_2d, ctx.preshuffle, ctx.use_gradient_sr
    )
    grad_input = _gemm_fp4(
        g, g_scale, False, b_t, b_t_scale, True, ctx.out_dtype, ctx.preshuffle
    ).reshape(ctx.orig_shape)
    grad_weight = _gemm_fp4(
        g_t, g_t_scale, False, a_t, a_t_scale, True, ctx.out_dtype, ctx.preshuffle
    )
    return grad_input, grad_weight


class _MXFP8Forward(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, preshuffle, use_gradient_sr):
        input_2d = input.reshape(-1, input.shape[-1])
        a, a_scale, a_t, a_t_scale = _quantize_mxfp8_rowwise_mxfp4_colwise(
            input_2d, False, False, True, preshuffle, preshuffle
        )
        b, b_scale, b_t, b_t_scale = _quantize_mxfp8_rowwise_mxfp4_colwise(
            weight, True, True, False, preshuffle, preshuffle
        )
        output = gemm_fp8_impl(
            a,
            a_scale,
            False,
            b,
            b_scale,
            True,
            input.dtype,
            False,
            granularity=_GRANULARITY,
            default_backend=BackendType.FLYDSL.value,
        )
        return (
            output,
            a_t.view(torch.uint8),
            a_t_scale.view(torch.uint8),
            b_t.view(torch.uint8),
            b_t_scale.view(torch.uint8),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, _, preshuffle, use_gradient_sr = inputs
        ctx.preshuffle = preshuffle
        ctx.use_gradient_sr = use_gradient_sr
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        _, a_t, a_t_scale, b_t, b_t_scale = output
        ctx.save_for_backward(
            a_t.view(_FP4_DTYPE),
            a_t_scale.view(torch.float8_e8m0fnu),
            b_t.view(_FP4_DTYPE),
            b_t_scale.view(torch.float8_e8m0fnu),
        )
        ctx.mark_non_differentiable(a_t, a_t_scale, b_t, b_t_scale)

    @staticmethod
    def backward(ctx, grad_output, *_):
        grad_input, grad_weight = _mxfp4_backward(
            ctx, grad_output, *ctx.saved_tensors
        )
        return grad_input, grad_weight, None, None


class _MXFP8Pure(torch.autograd.Function):
    @staticmethod
    def forward(input, weight):
        input_2d = input.reshape(-1, input.shape[-1])
        a, a_scale, a_t, a_t_scale = _quantize_mxfp8_dual(input_2d, False, False)
        if isinstance(weight, MXFP8GatheredWeight):
            b, b_scale = weight.qdata, weight.scale
            b_t, b_t_scale = weight.qdata_t, weight.scale_t
        else:
            b, b_scale, b_t, b_t_scale = _quantize_mxfp8_dual(
                weight, True, True
            )
        output = gemm_fp8_impl(
            a,
            a_scale,
            False,
            b,
            b_scale,
            True,
            input.dtype,
            False,
            granularity=_GRANULARITY,
            default_backend=BackendType.FLYDSL.value,
        )
        return output, a_t, a_t_scale, b_t, b_t_scale

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, _ = inputs
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        _, a_t, a_t_scale, b_t, b_t_scale = output
        ctx.save_for_backward(a_t, a_t_scale, b_t, b_t_scale)
        ctx.mark_non_differentiable(a_t, a_t_scale, b_t, b_t_scale)

    @staticmethod
    def backward(ctx, grad_output, *_):
        a_t, a_t_scale, b_t, b_t_scale = ctx.saved_tensors
        gradient_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        g, g_scale, g_t, g_t_scale = _quantize_mxfp8_dual(
            gradient_2d, False, False
        )
        grad_input = gemm_fp8_impl(
            g,
            g_scale,
            False,
            b_t,
            b_t_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=_GRANULARITY,
            default_backend=BackendType.FLYDSL.value,
        ).reshape(ctx.orig_shape)
        grad_weight = gemm_fp8_impl(
            g_t,
            g_t_scale,
            False,
            a_t,
            a_t_scale,
            True,
            ctx.out_dtype,
            False,
            granularity=_GRANULARITY,
            default_backend=BackendType.FLYDSL.value,
        )
        return grad_input, grad_weight


class _BF16Forward(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, preshuffle, use_gradient_sr):
        return torch.nn.functional.linear(input.reshape(-1, input.shape[-1]), weight)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, preshuffle, use_gradient_sr = inputs
        ctx.preshuffle = preshuffle
        ctx.use_gradient_sr = use_gradient_sr
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        ctx.save_for_backward(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_2d = input.reshape(-1, input.shape[-1])
        a_t, a_t_scale = _quantize_mxfp4_colwise(
            input_2d, False, True, ctx.preshuffle, ctx.preshuffle
        )
        b_t, b_t_scale = _quantize_mxfp4_colwise(
            weight, True, False, ctx.preshuffle, ctx.preshuffle
        )
        grad_input, grad_weight = _mxfp4_backward(
            ctx, grad_output, a_t, a_t_scale, b_t, b_t_scale
        )
        return grad_input, grad_weight, None, None


_FORWARD_FUNCTIONS = {
    "bf16": _BF16Forward,
    "mxfp4": _MXFP4Forward,
    "mxfp8": _MXFP8Forward,
}

_MXFP4_CAPTURE_STEP = -1
_MXFP4_CAPTURED: set[tuple[int, str]] = set()


def set_mxfp4_capture_step(step: int) -> None:
    """Set the optimizer step associated with subsequent forward captures."""
    global _MXFP4_CAPTURE_STEP
    _MXFP4_CAPTURE_STEP = step


def _save_mxfp4_capture(input: torch.Tensor, weight: torch.Tensor, module_name: str) -> None:
    steps = {
        int(value)
        for value in os.getenv("FLUX_MXFP4_CAPTURE_STEPS", "").split(",")
        if value
    }
    key = (_MXFP4_CAPTURE_STEP, module_name)
    if _MXFP4_CAPTURE_STEP not in steps or key in _MXFP4_CAPTURED:
        return
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        output_dir = os.environ["FLUX_MXFP4_CAPTURE_DIR"]
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "step": _MXFP4_CAPTURE_STEP,
                "module": module_name,
                "input": input.reshape(-1, input.shape[-1])[:1024].detach().cpu(),
                "weight": weight.detach().cpu(),
            },
            os.path.join(
                output_dir,
                f"step{_MXFP4_CAPTURE_STEP}_{module_name.replace('.', '_')}.pt",
            ),
        )
    _MXFP4_CAPTURED.add(key)


@torch.library.custom_op("primus::capture_mxfp4_linear", mutates_args=(), device_types="cuda")
def _capture_mxfp4_linear(
    input: torch.Tensor, weight: torch.Tensor, module_name: str
) -> torch.Tensor:
    _save_mxfp4_capture(input, weight, module_name)
    return input.clone()


@_capture_mxfp4_linear.register_fake
def _capture_mxfp4_linear_fake(
    input: torch.Tensor, weight: torch.Tensor, module_name: str
) -> torch.Tensor:
    del weight, module_name
    return torch.empty_like(input)


_capture_mxfp4_linear.register_autograd(
    lambda ctx, grad: (grad, None, None),
    setup_context=lambda ctx, inputs, output: None,
)


_MXFP8_FSDP_PRESERVE_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


class MXFP8GatheredWeight(torch.Tensor):
    """Logical BF16 weight backed by gathered MXFP8 data in both orientations."""

    @staticmethod
    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        qdata_t: torch.Tensor,
        scale_t: torch.Tensor,
        shape: tuple[int, int],
        dtype: torch.dtype,
        requires_grad: bool = False,
        *,
        _aliases: Optional[list["MXFP8GatheredWeight"]] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=(shape[1], 1),
            dtype=dtype,
            device=qdata.device,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        qdata_t: torch.Tensor,
        scale_t: torch.Tensor,
        shape: tuple[int, int],
        dtype: torch.dtype,
        requires_grad: bool = False,
        *,
        _aliases: Optional[list["MXFP8GatheredWeight"]] = None,
    ):
        del requires_grad
        self.qdata = qdata
        self.scale = scale
        self.qdata_t = qdata_t
        self.scale_t = scale_t
        self._aliases = _aliases if _aliases is not None else []
        self._aliases.append(self)
        self.logical_shape = shape
        self.logical_dtype = dtype

    def __repr__(self):
        return (
            f"MXFP8GatheredWeight(shape={self.logical_shape}, "
            f"dtype={self.logical_dtype}, device={self.device})"
        )

    def __tensor_flatten__(self):
        return ["qdata", "scale", "qdata_t", "scale_t"], {
            "shape": self.logical_shape,
            "dtype": self.logical_dtype,
            "requires_grad": self.requires_grad,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return MXFP8GatheredWeight(
            inner_tensors["qdata"],
            inner_tensors["scale"],
            inner_tensors["qdata_t"],
            inner_tensors["scale_t"],
            metadata["shape"],
            metadata["dtype"],
            metadata["requires_grad"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        tensor = args[0]
        if func == torch.ops.aten.detach.default:
            return MXFP8GatheredWeight(
                tensor.qdata,
                tensor.scale,
                tensor.qdata_t,
                tensor.scale_t,
                tensor.logical_shape,
                tensor.logical_dtype,
                False,
                _aliases=tensor._aliases,
            )
        if func == torch.ops.aten.clone.default:
            return MXFP8GatheredWeight(
                tensor.qdata.clone(),
                tensor.scale.clone(),
                tensor.qdata_t.clone(),
                tensor.scale_t.clone(),
                tensor.logical_shape,
                tensor.logical_dtype,
                tensor.requires_grad,
                _aliases=tensor._aliases,
            )
        if func in {torch.ops.aten.view.default, torch.ops.aten.as_strided.default}:
            if tensor.qdata.numel() != tensor.numel():
                raise RuntimeError("MXFP8 gathered weight views require an unpadded K dimension")
            qdata = func(tensor.qdata, *args[1:], **(kwargs or {}))
            aliases = tensor._aliases if qdata.shape == tensor.qdata.shape else None
            return MXFP8GatheredWeight(
                qdata,
                tensor.scale,
                tensor.qdata_t,
                tensor.scale_t,
                tuple(args[1]),
                tensor.logical_dtype,
                tensor.requires_grad,
                _aliases=aliases,
            )
        if func == torch.ops.aten.select.int:
            data = _dequantize_mxfp8_rowwise(
                tensor.qdata, tensor.scale, tensor.logical_dtype
            )[:, : tensor.logical_shape[-1]]
            return func(data, *args[1:], **(kwargs or {}))
        raise NotImplementedError(f"MXFP8 gathered weight does not support {func}")


class MXFP8FSDPWeight(torch.Tensor):
    """FP32 master weight with MXFP8-only FSDP AllGather transport."""

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        columns: int,
        gathered_aliases: Optional[list[MXFP8GatheredWeight]] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        columns: int,
        gathered_aliases: Optional[list[MXFP8GatheredWeight]] = None,
    ):
        self._tensor = tensor
        self._columns = columns
        self._gathered_aliases = (
            gathered_aliases if gathered_aliases is not None else []
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        columns: Optional[int] = None
        gathered_aliases: Optional[list[MXFP8GatheredWeight]] = None

        def unwrap(tensor):
            nonlocal columns, gathered_aliases
            if columns is None:
                columns = tensor._columns
                gathered_aliases = tensor._gathered_aliases
            return tensor._tensor

        args, kwargs = pytree.tree_map_only(
            MXFP8FSDPWeight, unwrap, (args, kwargs or {})
        )
        output = func(*args, **kwargs)
        if func not in _MXFP8_FSDP_PRESERVE_OPS and func != torch.ops.aten.detach.default:
            return output
        return pytree.tree_map_only(
            torch.Tensor,
            lambda tensor: MXFP8FSDPWeight(tensor, columns, gathered_aliases),
            output,
        )

    def __tensor_flatten__(self):
        return ["_tensor"], {"columns": self._columns}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return MXFP8FSDPWeight(inner_tensors["_tensor"], metadata["columns"])

    def fsdp_pre_all_gather(self, mesh):
        local_weight = self._tensor.reshape(-1, self._columns).to(torch.bfloat16)
        if local_weight.shape[0] % _MXFP4_PADDING_ALIGN_SIZE:
            raise ValueError("MXFP8 AllGather requires row shards aligned to 128")
        qdata, scale = _quantize_mxfp8_rowwise(local_weight, True)
        metadata = (self._columns, qdata.shape[1], scale.shape[1])
        return (qdata.view(torch.uint8), scale.view(torch.uint8)), metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        qdata_bytes, scale_bytes = all_gather_outputs
        columns, padded_columns, scale_columns = metadata
        qdata = qdata_bytes.view(_MXFP8_DTYPE).reshape(-1, padded_columns)
        scale = scale_bytes.view(torch.float8_e8m0fnu).reshape(-1, scale_columns)
        qdata_t = torch.ops.primus_turbo_cpp_extension.transpose_2d(qdata, -2, -1)
        scale_t = scale[::_MXFP4_BLOCK_SIZE].T.repeat_interleave(
            _MXFP4_BLOCK_SIZE, dim=0
        ).contiguous()
        shape = (qdata.shape[0], columns)
        if out is not None:
            from torch.distributed._tensor import DTensor

            target = out._local_tensor if isinstance(out, DTensor) else out
            if not isinstance(target, MXFP8GatheredWeight):
                raise RuntimeError(
                    f"Expected MXFP8GatheredWeight AllGather output, got {type(target)}"
                )
            for alias in target._aliases:
                if not any(alias is existing for existing in self._gathered_aliases):
                    self._gathered_aliases.append(alias)
            target._aliases = self._gathered_aliases
            for alias in self._gathered_aliases:
                alias.qdata = qdata
                alias.scale = scale
                alias.qdata_t = qdata_t
                alias.scale_t = scale_t
                alias.logical_shape = shape
                alias.logical_dtype = param_dtype
            return
        output = MXFP8GatheredWeight(
            qdata,
            scale,
            qdata_t,
            scale_t,
            shape,
            param_dtype,
            _aliases=self._gathered_aliases,
        )
        return output, (qdata_bytes, scale_bytes)


torch.serialization.add_safe_globals([MXFP8FSDPWeight, MXFP8GatheredWeight])


class MXFP4Linear(torch.nn.Module):
    """Linear with recipe-selected forward and backward precision."""

    def __init__(
        self,
        linear: torch.nn.Linear,
        forward_precision: str,
        *,
        backward_precision: str = "mxfp4",
        mxfp8_all_gather: bool = False,
        forward_hadamard: bool = False,
        activation_residual_mode: int = 0,
        eval_bf16: bool = False,
        gradient_stochastic_rounding: bool = False,
        fqn: str = "",
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if mxfp8_all_gather:
            if forward_precision != "mxfp8" or backward_precision != "mxfp8":
                raise ValueError("MXFP8 AllGather requires pure MXFP8 forward and backward")
            self.weight = torch.nn.Parameter(
                MXFP8FSDPWeight(linear.weight, linear.in_features),
                requires_grad=linear.weight.requires_grad,
            )
        else:
            self.weight = linear.weight
        self.bias = linear.bias
        self.forward_precision = forward_precision
        if backward_precision not in {"mxfp4", "mxfp8"}:
            raise ValueError(f"Unsupported backward precision: {backward_precision}")
        if backward_precision == "mxfp8" and forward_precision != "mxfp8":
            raise ValueError("MXFP8 backward requires MXFP8 forward")
        self.backward_precision = backward_precision
        self.forward_hadamard = forward_hadamard
        self.activation_residual_mode = activation_residual_mode
        self.eval_bf16 = eval_bf16
        self.gradient_stochastic_rounding = gradient_stochastic_rounding
        capture_modules = set(
            filter(None, os.getenv("FLUX_MXFP4_CAPTURE_MODULES", "").split(","))
        )
        self._capture_name = fqn if fqn in capture_modules else ""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.eval_bf16 and not self.training:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        shape = input.shape[:-1]
        if self._capture_name:
            input = _capture_mxfp4_linear(input, self.weight, self._capture_name)
        if self.backward_precision == "mxfp8":
            output = _MXFP8Pure.apply(input, self.weight)[0]
        elif self.forward_precision == "mxfp4":
            function = _FORWARD_FUNCTIONS[self.forward_precision]
            result = function.apply(
                input,
                self.weight,
                True,
                self.gradient_stochastic_rounding,
                self.forward_hadamard,
                self.activation_residual_mode,
            )
            output = result[0]
        else:
            function = _FORWARD_FUNCTIONS[self.forward_precision]
            result = function.apply(
                input, self.weight, True, self.gradient_stochastic_rounding
            )
            output = result[0] if self.forward_precision == "mxfp8" else result
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape, self.out_features)

    def switch_forward_to_bf16(self) -> bool:
        """Heal an MXFP4 forward while preserving its MXFP4 backward."""
        if self.forward_precision != "mxfp4":
            return False
        self.forward_precision = "bf16"
        self.forward_hadamard = False
        self.activation_residual_mode = 0
        return True
