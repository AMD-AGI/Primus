# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Compile-friendly FP8 linear layers for Megatron local spec.

Self-contained autograd Functions that call Primus Turbo's low-level building
blocks (quantize_fp8 + gemm_fp8_impl) directly, bypassing the higher-level
pt.ops.gemm_fp8 wrappers entirely.

Key properties:
- Tensorwise uses the setup_context pattern with primitive-only args so
  torch.compile can trace through without graph breaks. FP8 weight data is
  extracted by the caller (Float8*ParallelLinear._forward_impl) to avoid
  tensor subclass tracing inside the autograd.Function.
- Rowwise and blockwise still use @allow_in_graph (separate feasibility work).
- gemm_fp8_impl is already a torch.library.custom_op with register_fake
- Zero TransformerEngine dependencies
- Builds against stock public Primus-Turbo main: the only kernels not on main
  (the fused FP8 cast / cast+transpose Triton ops) are vendored locally in
  fp8_cast_kernels_triton.py; everything else still comes from Turbo main.
- Requires tensor_model_parallel_size=1, no GAF, no sequence_parallel
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from megatron.core.enums import Fp8Recipe
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

from primus.backends.megatron.core.fp8_utils import (
    MXFP8_SCALING_BLOCK_SIZE,
    SCALING_BLOCK_SIZE,
)

# ---------------------------------------------------------------------------
# torch.compile-friendly wrappers for C++ quantization ops.
#
# The raw C++ ops (primus_turbo_cpp_extension::quantize_fp8_tensorwise etc.)
# lack an Autograd dispatch key, which triggers warnings and can break
# torch.compile graph tracing. Wrapping them with @torch.library.custom_op
# (the same pattern used by gemm_fp8_impl) automatically handles Autograd
# dispatch and provides register_fake for shape inference during tracing.
# ---------------------------------------------------------------------------

_custom_op = torch.library.custom_op


@_custom_op("primus::quantize_fp8_tensorwise", mutates_args=(), device_types="cuda")
def _quantize_fp8_tensorwise_op(
    x: torch.Tensor, out_dtype: torch.dtype, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, out_dtype, scale)


@_quantize_fp8_tensorwise_op.register_fake
def _quantize_fp8_tensorwise_fake(
    x: torch.Tensor, out_dtype: torch.dtype, scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_fp8 = torch.empty_like(x, dtype=out_dtype)
    scale_inv = torch.empty((), dtype=torch.float32, device=x.device)
    return x_fp8, scale_inv


def _quantize_fp8_tensorwise_setup_context(ctx, inputs, output):
    pass


def _quantize_fp8_tensorwise_backward(ctx, grad_x_fp8, grad_scale_inv):
    return None, None, None


_quantize_fp8_tensorwise_op.register_autograd(
    _quantize_fp8_tensorwise_backward,
    setup_context=_quantize_fp8_tensorwise_setup_context,
)


def _cast_transpose_fp8_fused_with_amax(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    scale: torch.Tensor,
    amax_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused FP8 quantize + transpose + optional amax via the Triton @triton_op.

    Returns (fp8_out, fp8_transpose, scale_inv). Inductor-transparent.

    Uses the cast+transpose Triton kernel vendored in Primus
    (fp8_cast_kernels_triton), so this arm builds against stock Turbo main.
    """
    from primus.backends.megatron.core.extensions.fp8_cast_kernels_triton import (
        cast_transpose_fp8_triton,
    )

    return cast_transpose_fp8_triton(x, out_dtype, scale, amax_out)


def _cast_fp8_fused_with_amax(
    x: torch.Tensor,
    out_dtype: torch.dtype,
    scale: torch.Tensor,
    amax_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Native-layout fused FP8 quantize (no transpose) + optional amax.

    Analog of _cast_transpose_fp8_fused_with_amax for the natural NN/TN arm:
    applies the delayed scale and captures the current abs-max in a single HBM
    pass, with no [N, M] transpose write. Returns (fp8_out, scale_inv).

    Uses the no-transpose cast Triton kernel vendored in Primus
    (fp8_cast_kernels_triton), so this arm builds against stock Turbo main.
    """
    from primus.backends.megatron.core.extensions.fp8_cast_kernels_triton import (
        cast_fp8_triton,
    )

    return cast_fp8_triton(x, out_dtype, scale, amax_out)


def _quantize_fp8_tw(
    x: torch.Tensor,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch tensorwise FP8 quantize via the standard C++ @custom_op."""
    return _quantize_fp8_tensorwise_op(x, out_dtype)


def _get_fp8_dtype(format: Format, is_fwd: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _build_fp8_config(config):
    """Build Float8QuantConfig from TransformerConfig without TE dependency."""
    FORMAT_MAP = {"e4m3": Format.E4M3, "hybrid": Format.HYBRID}
    fmt = FORMAT_MAP[config.fp8]

    if config.fp8_recipe in (Fp8Recipe.tensorwise, Fp8Recipe.delayed):
        # Fp8Recipe.delayed is accepted as a TE-compatible alias: it implies
        # tensorwise granularity (the only granularity that makes sense with
        # delayed/amax-history scaling).  The actual delayed-vs-dynamic
        # strategy is resolved separately via fp8_scaling_strategy or the
        # recipe itself (see Float8*ParallelLinear._use_delayed_scaling).
        return Float8QuantConfig(format=fmt, granularity=ScalingGranularity.TENSORWISE)
    elif config.fp8_recipe == Fp8Recipe.blockwise:
        return Float8QuantConfig(
            format=fmt,
            granularity=ScalingGranularity.BLOCKWISE,
            block_size=SCALING_BLOCK_SIZE,
        )
    elif config.fp8_recipe == Fp8Recipe.mxfp8:
        return Float8QuantConfig(
            format=fmt,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP8_SCALING_BLOCK_SIZE,
            scale_dtype=ScaleDtype.E8M0,
        )
    else:
        raise ValueError(
            f"Float8 local spec does not support fp8_recipe={config.fp8_recipe}. "
            f"Supported: tensorwise, delayed, blockwise, mxfp8."
        )


# ---------------------------------------------------------------------------
# Decomposed FP8 quantize — native aten ops for Inductor fusion
# ---------------------------------------------------------------------------


def _quantize_fp8_tensorwise(x, fp8_dtype, fp8_max):
    """Tensorwise FP8 quantize using native aten ops (traceable by Inductor).

    Scale is computed in FP32 for precision, then narrowed to x.dtype (BF16) so
    the pointwise chain (mul, clamp, to_fp8) stays in BF16 registers inside the
    fused Triton kernel.  scale_inv is derived from the FP32 scale before the
    narrowing to preserve hipBLASLt's float32 precision requirement.
    """
    amax = x.abs().amax().float()
    scale_f32 = fp8_max / amax.clamp(min=1e-12)
    scale_inv = 1.0 / scale_f32
    scale = scale_f32.clamp(max=torch.finfo(x.dtype).max).to(x.dtype)
    x_fp8 = (x * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale_inv


# ---------------------------------------------------------------------------
# Autograd Functions — one per FP8 scaling granularity.
# ---------------------------------------------------------------------------


class DecomposedFP8LinearTensorwiseFunction(torch.autograd.Function):
    """FP8 linear (Y = X @ W^T) with tensorwise scaling, using decomposed
    quantize for forward-pass Inductor fusion.

    Experimental: designed to enable Inductor fusion of quantize with
    surrounding kernels. Currently slower than OpaqueFP8LinearTensorwiseFunction
    due to Inductor's internal FP32 promotion and amax reduction barriers.
    Kept for future work (delayed scaling, Compiled Autograd).

    Uses setup_context pattern with primitive-only arguments (no config objects)
    for clean Dynamo tracing.
    """

    @staticmethod
    def forward(
        input, weight, fp8_fwd_dtype, fp8_bwd_dtype, fp8_fwd_max, fp8_bwd_max, gran_value, backend_value
    ):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        a_fp8, a_scale_inv = _quantize_fp8_tensorwise(input_2d, fp8_fwd_dtype, fp8_fwd_max)
        b_fp8, b_scale_inv = _quantize_fp8_tensorwise(weight, fp8_fwd_dtype, fp8_fwd_max)

        output = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=gran_value,
            default_backend=backend_value,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        return output, a_fp8, a_scale_inv, b_fp8, b_scale_inv

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, _, fp8_fwd_dtype, fp8_bwd_dtype, fp8_fwd_max, fp8_bwd_max, gran_value, backend_value = inputs
        output_val, a_fp8, a_scale_inv, b_fp8, b_scale_inv = output

        ctx.save_for_backward(a_fp8, a_scale_inv, b_fp8, b_scale_inv)
        ctx.mark_non_differentiable(a_fp8, a_scale_inv, b_fp8, b_scale_inv)
        ctx.out_dtype = inputs[0].dtype
        ctx.orig_shape = inputs[0].shape
        ctx.fp8_bwd_dtype = fp8_bwd_dtype
        ctx.fp8_bwd_max = fp8_bwd_max
        ctx.gran_value = gran_value
        ctx.backend_value = backend_value

    @staticmethod
    def backward(ctx, grad_output, *_):
        a_fp8, a_scale_inv, b_fp8, b_scale_inv = ctx.saved_tensors

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        if not grad_2d.is_contiguous():
            grad_2d = grad_2d.contiguous()

        grad_fp8, grad_scale_inv = _quantize_fp8_tensorwise(grad_2d, ctx.fp8_bwd_dtype, ctx.fp8_bwd_max)

        grad_input = gemm_fp8_impl(
            grad_fp8,
            grad_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            False,
            ctx.out_dtype,
            False,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
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
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )

        return grad_input, grad_weight, None, None, None, None, None, None


class OpaqueFP8LinearTensorwiseFunction(torch.autograd.Function):
    """FP8 linear (Y = X @ W^T) with tensorwise scaling using Primus Turbo's
    opaque C++ quantize_fp8 kernel.

    Uses the setup_context pattern with primitive-only saved state so that
    torch.compile / TorchDynamo can trace through without graph breaks.
    The caller must pre-extract FP8 weight data (b_fp8, b_scale_inv) from
    FP8UnshardedWeightTensor *before* calling .apply().

    Default path for tensorwise -- faster than the decomposed variant due to
    optimized C++ quantize kernels.

    Backward GEMMs are normalized to NT layout (transA=F, transB=T, transC=F)
    so that hipBLASLt always selects the fast TN Tensile kernel on MI355X.
    Pre-transposed FP8 copies of input and weight are saved for this purpose.
    """

    @staticmethod
    def forward(
        input,
        weight,
        weight_fp8,
        weight_scale_inv,
        fp8_fwd_dtype,
        fp8_bwd_dtype,
        gran_value,
        backend_value,
        force_nt=True,
    ):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        a_fp8, a_scale_inv = _quantize_fp8_tw(input_2d, fp8_fwd_dtype)

        output = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            weight_fp8,
            weight_scale_inv,
            True,
            out_dtype,
            False,
            granularity=gran_value,
            default_backend=backend_value,
        )

        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        if force_nt:
            # Pre-transpose so backward dgrad/wgrad both run as NT GEMMs.
            a_t_fp8 = a_fp8.t().contiguous()
            w_t_fp8 = weight_fp8.t().contiguous()
            return (output, a_t_fp8, a_scale_inv, w_t_fp8)

        # Native: keep operands in their natural layout (no transpose). weight_fp8
        # is not returned -- setup_context saves it straight from inputs.
        return (output, a_fp8, a_scale_inv)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Tolerant unpack: legacy 8-arg callers default to forced NT; only the
        # 9th positional (force_nt) toggles the native arm.
        force_nt = inputs[8] if len(inputs) > 8 else True
        (
            input,
            weight,
            weight_fp8,
            weight_scale_inv,
            fp8_fwd_dtype,
            fp8_bwd_dtype,
            gran_value,
            backend_value,
        ) = inputs[:8]

        if force_nt:
            output_val, a_t_fp8, a_scale_inv, w_t_fp8 = output
            ctx.save_for_backward(a_t_fp8, a_scale_inv, w_t_fp8, weight_scale_inv)
            ctx.mark_non_differentiable(a_t_fp8, a_scale_inv, w_t_fp8)
        else:
            output_val, a_fp8, a_scale_inv = output
            ctx.save_for_backward(a_fp8, a_scale_inv, weight_fp8, weight_scale_inv)
            ctx.mark_non_differentiable(a_fp8, a_scale_inv)

        ctx.force_nt = force_nt
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        ctx.fp8_bwd_dtype = fp8_bwd_dtype
        ctx.gran_value = gran_value
        ctx.backend_value = backend_value

    @staticmethod
    def backward(ctx, grad_output, *_):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_fp8, grad_scale_inv = _quantize_fp8_tw(grad_2d, ctx.fp8_bwd_dtype)

        if ctx.force_nt:
            a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv = ctx.saved_tensors

            grad_input = gemm_fp8_impl(
                grad_fp8,
                grad_scale_inv,
                False,
                w_t_fp8,
                w_scale_inv,
                True,
                ctx.out_dtype,
                False,
                granularity=ctx.gran_value,
                default_backend=ctx.backend_value,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            grad_t_fp8 = grad_fp8.t().contiguous()
            grad_weight = gemm_fp8_impl(
                grad_t_fp8,
                grad_scale_inv,
                False,
                a_t_fp8,
                a_scale_inv,
                True,
                ctx.out_dtype,
                False,
                granularity=ctx.gran_value,
                default_backend=ctx.backend_value,
            )
        else:
            a_fp8, a_scale_inv, w_fp8, w_scale_inv = ctx.saved_tensors

            # dgrad NN: grad @ weight  (transA=F, transB=F)
            grad_input = gemm_fp8_impl(
                grad_fp8,
                grad_scale_inv,
                False,
                w_fp8,
                w_scale_inv,
                False,
                ctx.out_dtype,
                False,
                granularity=ctx.gran_value,
                default_backend=ctx.backend_value,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            # wgrad TN: grad^T @ a  (transA=T, transB=F)
            grad_weight = gemm_fp8_impl(
                grad_fp8,
                grad_scale_inv,
                True,
                a_fp8,
                a_scale_inv,
                False,
                ctx.out_dtype,
                False,
                granularity=ctx.gran_value,
                default_backend=ctx.backend_value,
            )

        return grad_input, grad_weight, None, None, None, None, None, None, None


class DelayedFP8LinearTensorwiseFunction(torch.autograd.Function):
    """FP8 linear with delayed tensorwise scaling -- 3 independent scale tracks.

    Weight is quantized inline during the forward pass (inside the compiled
    graph where Inductor eliminates CPU dispatcher overhead).  Input and
    gradient quantization also happen inline with fused amax capture.
    Backward GEMM layout follows ``force_nt`` (set from ``fp8_force_nt_layout``):
    native by default (dgrad=NN, wgrad=TN, no transpose), or forced-NT (operands
    pre-transposed so both backward GEMMs run as NT) when opted in.
    """

    @staticmethod
    def forward(
        input,
        weight,
        scale_input,
        scale_weight,
        scale_grad,
        staged_input_amax,
        staged_weight_amax,
        staged_grad_amax,
        fp8_fwd_dtype,
        fp8_bwd_dtype,
        gran_value,
        backend_value,
        force_nt=True,
    ):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        if force_nt:
            a_fp8, a_t_fp8, a_scale_inv = _cast_transpose_fp8_fused_with_amax(
                input_2d, fp8_fwd_dtype, scale_input, staged_input_amax
            )
            torch._assert(a_t_fp8.is_contiguous(), "cast_transpose must return contiguous transpose")

            w_fp8, w_t_fp8, w_scale_inv = _cast_transpose_fp8_fused_with_amax(
                weight, fp8_fwd_dtype, scale_weight, staged_weight_amax
            )
            torch._assert(w_t_fp8.is_contiguous(), "cast_transpose must return contiguous transpose")

            output = gemm_fp8_impl(
                a_fp8,
                a_scale_inv,
                False,
                w_fp8,
                w_scale_inv,
                True,
                out_dtype,
                False,
                granularity=gran_value,
                default_backend=backend_value,
            )
            output = output.reshape(*orig_shape[:-1], output.shape[-1])

            return (output, a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv)

        # Native: apply the delayed scale and capture the current amax in a
        # single fused pass via the no-transpose Triton kernel. No transpose;
        # backward runs dgrad=NN / wgrad=TN (see backward).
        a_fp8, a_scale_inv = _cast_fp8_fused_with_amax(
            input_2d, fp8_fwd_dtype, scale_input, staged_input_amax
        )

        w_fp8, w_scale_inv = _cast_fp8_fused_with_amax(
            weight, fp8_fwd_dtype, scale_weight, staged_weight_amax
        )

        output = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            w_fp8,
            w_scale_inv,
            True,
            out_dtype,
            False,
            granularity=gran_value,
            default_backend=backend_value,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        return (output, a_fp8, a_scale_inv, w_fp8, w_scale_inv)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Tolerant unpack: legacy 12-arg callers default to forced NT; only the
        # 13th positional (force_nt) toggles the native arm.
        force_nt = inputs[12] if len(inputs) > 12 else True
        (
            input,
            weight,
            scale_input,
            scale_weight,
            scale_grad,
            staged_input_amax,
            staged_weight_amax,
            staged_grad_amax,
            fp8_fwd_dtype,
            fp8_bwd_dtype,
            gran_value,
            backend_value,
        ) = inputs[:12]

        if force_nt:
            (output_val, a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv) = output
            ctx.save_for_backward(a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv, scale_grad, staged_grad_amax)
            ctx.mark_non_differentiable(a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv)
        else:
            (output_val, a_fp8, a_scale_inv, w_fp8, w_scale_inv) = output
            ctx.save_for_backward(a_fp8, a_scale_inv, w_fp8, w_scale_inv, scale_grad, staged_grad_amax)
            ctx.mark_non_differentiable(a_fp8, a_scale_inv, w_fp8, w_scale_inv)

        ctx.force_nt = force_nt
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        ctx.fp8_bwd_dtype = fp8_bwd_dtype
        ctx.gran_value = gran_value
        ctx.backend_value = backend_value

    @staticmethod
    def backward(ctx, grad_output, *_):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        op_a, a_scale_inv, op_w, w_scale_inv, scale_grad, staged_grad_amax = ctx.saved_tensors

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.force_nt:
            # op_a = a_t_fp8, op_w = w_t_fp8
            grad_fp8, grad_t_fp8, grad_scale_inv = _cast_transpose_fp8_fused_with_amax(
                grad_2d, ctx.fp8_bwd_dtype, scale_grad, staged_grad_amax
            )
            torch._assert(grad_t_fp8.is_contiguous(), "cast_transpose must return contiguous transpose")

            grad_input = gemm_fp8_impl(
                grad_fp8,
                grad_scale_inv,
                False,
                op_w,
                w_scale_inv,
                True,
                ctx.out_dtype,
                False,
                granularity=ctx.gran_value,
                default_backend=ctx.backend_value,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            grad_weight = gemm_fp8_impl(
                grad_t_fp8,
                grad_scale_inv,
                False,
                op_a,
                a_scale_inv,
                True,
                ctx.out_dtype,
                False,
                granularity=ctx.gran_value,
                default_backend=ctx.backend_value,
            )
        else:
            # Native: op_a = a_fp8, op_w = w_fp8. Fused quantize + amax, no transpose.
            grad_fp8, grad_scale_inv = _cast_fp8_fused_with_amax(
                grad_2d, ctx.fp8_bwd_dtype, scale_grad, staged_grad_amax
            )

            # dgrad NN: grad @ weight  (transA=F, transB=F)
            grad_input = gemm_fp8_impl(
                grad_fp8,
                grad_scale_inv,
                False,
                op_w,
                w_scale_inv,
                False,
                ctx.out_dtype,
                False,
                granularity=ctx.gran_value,
                default_backend=ctx.backend_value,
            )
            grad_input = grad_input.reshape(ctx.orig_shape)

            # wgrad TN: grad^T @ a  (transA=T, transB=F)
            grad_weight = gemm_fp8_impl(
                grad_fp8,
                grad_scale_inv,
                True,
                op_a,
                a_scale_inv,
                False,
                ctx.out_dtype,
                False,
                granularity=ctx.gran_value,
                default_backend=ctx.backend_value,
            )

        # 13 inputs to forward -> 13 grads returned (2 real + 11 None).
        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DualFP8LinearTensorwiseFunction(torch.autograd.Function):
    """Two independent FP8 linears in a single autograd node.

    Y_a = X_a @ W_a^T,  Y_b = X_b @ W_b^T

    Uses the setup_context pattern with primitive-only saved state so that
    torch.compile can trace through without graph breaks. The caller must
    pre-extract FP8 weight data before calling .apply().

    Backward GEMMs are normalized to NT layout (transA=F, transB=T, transC=F)
    so that hipBLASLt always selects the fast TN Tensile kernel on MI355X.
    """

    @staticmethod
    def forward(
        input_a,
        weight_a,
        weight_fp8_a,
        weight_scale_a,
        input_b,
        weight_b,
        weight_fp8_b,
        weight_scale_b,
        fp8_fwd_dtype,
        fp8_bwd_dtype,
        gran_value,
        backend_value,
    ):
        out_dtype = input_a.dtype

        orig_shape_a = input_a.shape
        orig_shape_b = input_b.shape
        input_a_2d = input_a.reshape(-1, input_a.shape[-1])
        input_b_2d = input_b.reshape(-1, input_b.shape[-1])

        a_fp8_a, a_scale_a = _quantize_fp8_tw(input_a_2d, fp8_fwd_dtype)
        a_fp8_b, a_scale_b = _quantize_fp8_tw(input_b_2d, fp8_fwd_dtype)

        output_a = gemm_fp8_impl(
            a_fp8_a,
            a_scale_a,
            False,
            weight_fp8_a,
            weight_scale_a,
            True,
            out_dtype,
            False,
            granularity=gran_value,
            default_backend=backend_value,
        )
        output_b = gemm_fp8_impl(
            a_fp8_b,
            a_scale_b,
            False,
            weight_fp8_b,
            weight_scale_b,
            True,
            out_dtype,
            False,
            granularity=gran_value,
            default_backend=backend_value,
        )

        output_a = output_a.reshape(*orig_shape_a[:-1], output_a.shape[-1])
        output_b = output_b.reshape(*orig_shape_b[:-1], output_b.shape[-1])

        a_t_fp8_a = a_fp8_a.t().contiguous()
        w_t_fp8_a = weight_fp8_a.t().contiguous()
        a_t_fp8_b = a_fp8_b.t().contiguous()
        w_t_fp8_b = weight_fp8_b.t().contiguous()

        return (output_a, output_b, a_t_fp8_a, a_scale_a, w_t_fp8_a, a_t_fp8_b, a_scale_b, w_t_fp8_b)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            input_a,
            weight_a,
            weight_fp8_a,
            weight_scale_a,
            input_b,
            weight_b,
            weight_fp8_b,
            weight_scale_b,
            fp8_fwd_dtype,
            fp8_bwd_dtype,
            gran_value,
            backend_value,
        ) = inputs
        (output_a, output_b, a_t_fp8_a, a_scale_a, w_t_fp8_a, a_t_fp8_b, a_scale_b, w_t_fp8_b) = output

        ctx.save_for_backward(
            a_t_fp8_a,
            a_scale_a,
            w_t_fp8_a,
            weight_scale_a,
            a_t_fp8_b,
            a_scale_b,
            w_t_fp8_b,
            weight_scale_b,
        )
        ctx.mark_non_differentiable(
            a_t_fp8_a,
            a_scale_a,
            w_t_fp8_a,
            a_t_fp8_b,
            a_scale_b,
            w_t_fp8_b,
        )
        ctx.out_dtype = input_a.dtype
        ctx.orig_shape_a = input_a.shape
        ctx.orig_shape_b = input_b.shape
        ctx.fp8_bwd_dtype = fp8_bwd_dtype
        ctx.gran_value = gran_value
        ctx.backend_value = backend_value

    @staticmethod
    def backward(ctx, grad_output_a, grad_output_b, *_):
        if not grad_output_a.is_contiguous():
            grad_output_a = grad_output_a.contiguous()
        if not grad_output_b.is_contiguous():
            grad_output_b = grad_output_b.contiguous()

        (a_t_fp8_a, a_scale_a, w_t_fp8_a, w_scale_a, a_t_fp8_b, a_scale_b, w_t_fp8_b, w_scale_b) = (
            ctx.saved_tensors
        )

        # --- Stream A ---
        grad_a_2d = grad_output_a.reshape(-1, grad_output_a.shape[-1])
        grad_fp8_a, grad_scale_a = _quantize_fp8_tw(grad_a_2d, ctx.fp8_bwd_dtype)

        grad_input_a = gemm_fp8_impl(
            grad_fp8_a,
            grad_scale_a,
            False,
            w_t_fp8_a,
            w_scale_a,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )
        grad_input_a = grad_input_a.reshape(ctx.orig_shape_a)

        grad_t_fp8_a = grad_fp8_a.t().contiguous()
        grad_weight_a = gemm_fp8_impl(
            grad_t_fp8_a,
            grad_scale_a,
            False,
            a_t_fp8_a,
            a_scale_a,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )

        # --- Stream B ---
        grad_b_2d = grad_output_b.reshape(-1, grad_output_b.shape[-1])
        grad_fp8_b, grad_scale_b = _quantize_fp8_tw(grad_b_2d, ctx.fp8_bwd_dtype)

        grad_input_b = gemm_fp8_impl(
            grad_fp8_b,
            grad_scale_b,
            False,
            w_t_fp8_b,
            w_scale_b,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )
        grad_input_b = grad_input_b.reshape(ctx.orig_shape_b)

        grad_t_fp8_b = grad_fp8_b.t().contiguous()
        grad_weight_b = gemm_fp8_impl(
            grad_t_fp8_b,
            grad_scale_b,
            False,
            a_t_fp8_b,
            a_scale_b,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )

        return (
            grad_input_a,
            grad_weight_a,
            None,
            None,
            grad_input_b,
            grad_weight_b,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch._dynamo.allow_in_graph
class FP8LinearRowwiseFunction(torch.autograd.Function):
    """FP8 linear (Y = X @ W^T) with rowwise scaling. Compile-friendly."""

    @staticmethod
    def forward(ctx, input, weight, config):
        a_dtype = _get_fp8_dtype(config.format, is_fwd=True)
        b_dtype = _get_fp8_dtype(config.format, is_fwd=True)
        out_dtype = input.dtype
        gran = config.granularity

        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        a_fp8_row, a_scale_inv_row = quantize_fp8(input_2d, a_dtype, gran, axis=-1)
        b_fp8_row, b_scale_inv_row = quantize_fp8(weight, b_dtype, gran, axis=-1)

        output = gemm_fp8_impl(
            a_fp8_row,
            a_scale_inv_row,
            False,
            b_fp8_row,
            b_scale_inv_row,
            True,
            out_dtype,
            False,
            granularity=gran.value,
            default_backend=BackendType.CK.value,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        a_fp8_col, a_scale_inv_col = quantize_fp8(input_2d, a_dtype, gran, axis=-2)
        b_fp8_col, b_scale_inv_col = quantize_fp8(weight, b_dtype, gran, axis=-2)

        ctx.save_for_backward(a_fp8_col, a_scale_inv_col, b_fp8_col, b_scale_inv_col)
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.orig_shape = orig_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        a_fp8_col, a_scale_inv_col, b_fp8_col, b_scale_inv_col = ctx.saved_tensors
        grad_dtype = _get_fp8_dtype(ctx.config.format, is_fwd=False)
        gran = ctx.config.granularity

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

        grad_fp8_row, grad_scale_inv_row = quantize_fp8(grad_2d, grad_dtype, gran, axis=-1)

        grad_input = gemm_fp8_impl(
            grad_fp8_row,
            grad_scale_inv_row,
            False,
            b_fp8_col,
            b_scale_inv_col,
            False,
            ctx.out_dtype,
            False,
            granularity=gran.value,
            default_backend=BackendType.CK.value,
        )
        grad_input = grad_input.reshape(ctx.orig_shape)

        grad_fp8_col, grad_scale_inv_col = quantize_fp8(grad_2d, grad_dtype, gran, axis=-2)

        grad_weight = gemm_fp8_impl(
            a_fp8_col,
            a_scale_inv_col,
            True,
            grad_fp8_col,
            grad_scale_inv_col,
            False,
            ctx.out_dtype,
            True,
            granularity=gran.value,
            default_backend=BackendType.CK.value,
        )

        return grad_input, grad_weight, None


@torch._dynamo.allow_in_graph
class FP8LinearBlockwiseFunction(torch.autograd.Function):
    """FP8 linear (Y = X @ W^T) with blockwise scaling. Compile-friendly.

    Note: saves BF16 input AND BF16 weight for backward (must re-quantize
    with different axis/dtype), so no activation memory savings compared to
    BF16 baseline.

    Forward uses CK backend (NT layout). Backward uses Triton backend for
    NN (grad_input) and TN (grad_weight) layouts because CK's blockwise
    GEMM produces NaN on these layouts.  To satisfy Triton's same-dtype
    constraint, the weight is re-quantized to the backward dtype (E5M2 for
    hybrid format) from the saved BF16 copy.
    """

    @staticmethod
    def forward(ctx, input, weight, config):
        a_dtype = _get_fp8_dtype(config.format, is_fwd=True)
        b_dtype = _get_fp8_dtype(config.format, is_fwd=True)
        out_dtype = input.dtype
        gran = config.granularity
        bs = config.block_size

        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(input_2d, a_dtype, axis=1, block_size=bs)
        b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(weight, b_dtype, block_size=bs)

        output = gemm_fp8_impl(
            a_fp8_row,
            a_scale_inv_row,
            False,
            b_fp8,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=gran.value,
            default_backend=BackendType.CK.value,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        ctx.save_for_backward(input_2d, weight)
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.orig_shape = orig_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input_saved, weight_saved = ctx.saved_tensors
        bwd_dtype = _get_fp8_dtype(ctx.config.format, is_fwd=False)
        gran = ctx.config.granularity
        bs = ctx.config.block_size

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

        grad_fp8_row, grad_scale_inv_row = quant_fp8_blockwise_impl(
            grad_2d, bwd_dtype, axis=-1, block_size=bs
        )
        grad_fp8_col, grad_scale_inv_col = quant_fp8_blockwise_impl(
            grad_2d, bwd_dtype, axis=-2, block_size=bs
        )
        a_fp8_col, a_scale_inv_col = quant_fp8_blockwise_impl(input_saved, bwd_dtype, axis=0, block_size=bs)
        b_fp8_bwd, b_scale_inv_bwd = quant_fp8_blockwise_for_weight_impl(
            weight_saved, bwd_dtype, block_size=bs
        )

        grad_input = gemm_fp8_impl(
            grad_fp8_row,
            grad_scale_inv_row,
            False,
            b_fp8_bwd,
            b_scale_inv_bwd,
            False,
            ctx.out_dtype,
            False,
            granularity=gran.value,
            default_backend=BackendType.TRITON.value,
        )
        grad_input = grad_input.reshape(ctx.orig_shape)

        grad_weight = gemm_fp8_impl(
            a_fp8_col,
            a_scale_inv_col,
            True,
            grad_fp8_col,
            grad_scale_inv_col,
            False,
            ctx.out_dtype,
            True,
            granularity=gran.value,
            default_backend=BackendType.TRITON.value,
        )

        return grad_input, grad_weight, None


# ---------------------------------------------------------------------------
# Delayed FP8 scaling helpers — shared by Float8ColumnParallelLinear and
# Float8RowParallelLinear.
# ---------------------------------------------------------------------------


def _update_fp8_scale(scale_buf, scale_inv_buf, amax, fp8_max):
    """TE-aligned scale computation with edge case handling."""
    sf = fp8_max / amax.clamp(min=1e-12)
    sf = torch.where(amax > 0.0, sf, scale_buf)
    sf = torch.where(torch.isfinite(amax), sf, scale_buf)
    sf = torch.where(
        torch.isinf(sf),
        torch.tensor(torch.finfo(torch.float32).max, device=sf.device),
        sf,
    )
    scale_buf.fill_(sf)
    if scale_inv_buf is not None:
        scale_inv_buf.fill_(1.0 / sf)


def _init_delayed_scaling_state(module):
    """Initialize delayed FP8 scaling buffers on a Float8*ParallelLinear."""
    config = module.config
    history_len = getattr(config, "fp8_amax_history_len", 1)
    fp8_fwd_max = torch.finfo(module._fp8_fwd_dtype).max
    fp8_bwd_max = torch.finfo(module._fp8_bwd_dtype).max

    for name in ["amax_history_input", "amax_history_weight", "amax_history_grad"]:
        module.register_buffer(
            name,
            torch.zeros(history_len),
            persistent=False,
        )

    for name in ["scale_input", "scale_weight", "scale_grad"]:
        module.register_buffer(
            name,
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )

    for name in ["staged_input_amax", "staged_grad_amax", "staged_weight_amax"]:
        module.register_buffer(
            name,
            torch.tensor(0.0),
            persistent=False,
        )

    module._fp8_fwd_max = fp8_fwd_max
    module._fp8_bwd_max = fp8_bwd_max
    module._amax_compute_algo = getattr(
        config,
        "fp8_amax_compute_algo",
        "most_recent",
    )
    module._history_idx = 0
    module._first_delayed_step = True


def _batch_compute_scales(amaxes, fp8_max, old_scales):
    """Vectorized scale computation across all modules at once."""
    sf = fp8_max / amaxes.clamp(min=1e-12)
    sf = torch.where(amaxes > 0.0, sf, old_scales)
    sf = torch.where(torch.isfinite(amaxes), sf, old_scales)
    sf = torch.where(torch.isinf(sf), torch.finfo(torch.float32).max, sf)
    return sf


# ---------------------------------------------------------------------------
# Fused Triton kernel: replaces Python-loop-based _batched_update per step
# with a single GPU dispatch over all (module, track) pairs.
# ---------------------------------------------------------------------------


@triton.jit
def _fused_delayed_scale_update_kernel(
    amax_history_ptr,
    staged_amaxes_ptr,
    scales_ptr,
    fp8_maxes_ptr,
    history_idx,
    N: tl.constexpr,
    H: tl.constexpr,
    use_max_algo: tl.constexpr,
    BLOCK_H: tl.constexpr,
    FP32_MAX: tl.constexpr,
    FILTER_ZEROS: tl.constexpr = False,
):
    track = tl.program_id(1)
    mod = tl.program_id(0)

    base = track * N * H + mod * H

    new_amax = tl.load(staged_amaxes_ptr + track * N + mod)
    tl.store(amax_history_ptr + base + history_idx, new_amax)

    if use_max_algo:
        amax = tl.zeros([], dtype=tl.float32)
        has_nonzero = tl.zeros([], dtype=tl.int32)
        for off in range(0, H, BLOCK_H):
            h_idx = off + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            vals = tl.load(amax_history_ptr + base + h_idx, mask=mask, other=0.0)
            if FILTER_ZEROS:
                nonzero_mask = vals > 0.0
                has_nonzero = has_nonzero | tl.sum(nonzero_mask.to(tl.int32), axis=0)
                filtered = tl.where(nonzero_mask, vals, 0.0)
                amax = tl.maximum(amax, tl.max(filtered, axis=0))
            else:
                amax = tl.maximum(amax, tl.max(vals, axis=0))
    else:
        amax = new_amax
        has_nonzero = tl.where(new_amax > 0.0, 1, 0).to(tl.int32)

    fp8_max = tl.load(fp8_maxes_ptr + track)
    old_scale = tl.load(scales_ptr + track * N + mod)
    sf = fp8_max / tl.maximum(amax, 1e-12)
    if FILTER_ZEROS:
        sf = tl.where(has_nonzero > 0, sf, old_scale)
    else:
        sf = tl.where(amax > 0.0, sf, old_scale)
    sf = tl.where(amax == amax, sf, old_scale)
    sf = tl.minimum(sf, FP32_MAX)
    tl.store(scales_ptr + track * N + mod, sf)


# ---------------------------------------------------------------------------
# Global registry: replaces per-module scalar buffers with views into
# contiguous (N,) tensors to eliminate Python iteration in the preamble.
# ---------------------------------------------------------------------------


class _DelayedScalingRegistry:
    __slots__ = (
        "n",
        "modules",
        "fwd_max",
        "bwd_max",
        "algo",
        "history_len",
        "_first_step",
        "amax_history",
        "_history_idx",
        "fp8_maxes",
        "staged_amaxes_3n",
        "scales_3n",
        "reduce_amax",
        "amax_reduce_group",
        "skip_bootstrap",
        "filter_zeros",
    )

    def __init__(self, modules):
        self.n = len(modules)
        self.modules = modules
        m0 = modules[0]
        self.fwd_max = m0._fp8_fwd_max
        self.bwd_max = m0._fp8_bwd_max
        self.algo = m0._amax_compute_algo
        self.history_len = m0.amax_history_input.shape[0]
        device = m0.weight.device

        self._first_step = True

        H = self.history_len
        N = self.n

        self.scales_3n = torch.ones(3, N, dtype=torch.float32, device=device)
        self.staged_amaxes_3n = torch.zeros(3, N, dtype=torch.float32, device=device)

        self.amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=device)
        self._history_idx = 0
        self.fp8_maxes = torch.tensor(
            [self.fwd_max, self.fwd_max, self.bwd_max],
            dtype=torch.float32,
            device=device,
        )

        _config = getattr(m0, "config", None)
        self.reduce_amax = getattr(_config, "fp8_reduce_amax", False) if _config else False
        self.skip_bootstrap = getattr(_config, "fp8_skip_first_step_bootstrap", False) if _config else False
        self.filter_zeros = getattr(_config, "fp8_filter_zeros_in_history", False) if _config else False
        self.amax_reduce_group = None
        if self.reduce_amax:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.amax_reduce_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True,
                    tp_only_amax_red=getattr(_config, "tp_only_amax_red", False),
                )

        for i, m in enumerate(modules):
            m._buffers["scale_input"] = torch.ones((), dtype=torch.float32, device=device)
            m._buffers["scale_weight"] = torch.ones((), dtype=torch.float32, device=device)
            m._buffers["scale_grad"] = torch.ones((), dtype=torch.float32, device=device)
            m._buffers["staged_input_amax"] = torch.zeros((), dtype=torch.float32, device=device)
            m._buffers["staged_weight_amax"] = torch.zeros((), dtype=torch.float32, device=device)
            m._buffers["staged_grad_amax"] = torch.zeros((), dtype=torch.float32, device=device)
            m._buffers["amax_history_input"] = self.amax_history[0, i, :]
            m._buffers["amax_history_weight"] = self.amax_history[1, i, :]
            m._buffers["amax_history_grad"] = self.amax_history[2, i, :]


@torch.no_grad()
def _fast_update_scales(registry):
    """Ultra-fast scale update for most_recent + history_len=1.

    Gathers per-module scalar amaxes into batched tensors, computes scales,
    then scatters back.  Each module has independent storage for its scale
    and amax buffers so that torch.compile version tracking is correct.
    """
    r = registry

    # Detect device migration AND per-module buffer pointer breakage (e.g.
    # after _reset_fp8_local_spec during MLPerf warmup teardown). Mirrors the
    # same check in _fast_update_scales_with_history so the warmup-reset
    # docstring contract holds for history_len==1 configs too: a fresh
    # registry.__init__(modules) call sets _first_step=True, which bootstraps
    # weight amaxes from the restored (post-warmup) weights below.
    if (
        r.scales_3n.device != r.modules[0].weight.device
        or r.amax_history.untyped_storage().data_ptr()
        != r.modules[0].amax_history_input.untyped_storage().data_ptr()
    ):
        r.__init__(r.modules)

    if r._first_step and not r.skip_bootstrap:
        weights = [m.weight.data for m in r.modules]
        try:
            weight_amaxes = torch.stack(torch._foreach_norm(weights, ord=float("inf"))).float()
        except (AttributeError, RuntimeError):
            weight_amaxes = torch.stack([w.abs().amax().float() for w in weights])
        for i, m in enumerate(r.modules):
            m.staged_weight_amax.fill_(weight_amaxes[i].item())
    if r._first_step:
        r._first_step = False

    staged_input = torch.stack([m.staged_input_amax for m in r.modules])
    staged_weight = torch.stack([m.staged_weight_amax for m in r.modules])
    staged_grad = torch.stack([m.staged_grad_amax for m in r.modules])

    if r.reduce_amax and r.amax_reduce_group is not None:
        amaxes_3n = torch.stack([staged_input, staged_weight, staged_grad])
        torch.distributed.all_reduce(
            amaxes_3n,
            op=torch.distributed.ReduceOp.MAX,
            group=r.amax_reduce_group,
        )
        staged_input, staged_weight, staged_grad = amaxes_3n[0], amaxes_3n[1], amaxes_3n[2]

    scale_input = torch.stack([m.scale_input for m in r.modules])
    scale_weight = torch.stack([m.scale_weight for m in r.modules])
    scale_grad = torch.stack([m.scale_grad for m in r.modules])

    new_input = _batch_compute_scales(staged_input, r.fwd_max, scale_input)
    new_weight = _batch_compute_scales(staged_weight, r.fwd_max, scale_weight)
    new_grad = _batch_compute_scales(staged_grad, r.bwd_max, scale_grad)

    # Batched device-only scatter: unbinding a (N,) float32 tensor into N
    # 0-d views and copying with torch._foreach_copy_ keeps the per-module
    # scale_* buffers independent (required for torch.compile version
    # tracking) without the 3*N .item() host syncs the loop version
    # incurred.
    #
    # NOTE (perf): the per-module scalar layout was chosen so that
    # torch.compile sees each module's scale_* as an independent buffer
    # version. A shared (N,) buffer with module-level views would let us
    # skip the unbind+foreach_copy entirely, at the cost of more frequent
    # compile cache invalidation. Worth benchmarking once the compile
    # convergence work settles.
    torch._foreach_copy_(
        [m.scale_input for m in r.modules],
        list(new_input.unbind()),
    )
    torch._foreach_copy_(
        [m.scale_weight for m in r.modules],
        list(new_weight.unbind()),
    )
    torch._foreach_copy_(
        [m.scale_grad for m in r.modules],
        list(new_grad.unbind()),
    )


@torch.no_grad()
def _fast_update_scales_with_history(registry):
    """Fused scale update for delayed scaling with history_len > 1.

    Single Triton kernel dispatch replaces the prior Python-loop-based
    per-module scale update (~200+ tiny GPU ops) with one launch.
    """
    r = registry

    if (
        r.amax_history.device != r.modules[0].weight.device
        or r.amax_history.untyped_storage().data_ptr()
        != r.modules[0].amax_history_input.untyped_storage().data_ptr()
    ):
        r.__init__(r.modules)

    if r._first_step and not r.skip_bootstrap:
        weights = [m.weight.data for m in r.modules]
        try:
            weight_amaxes = torch.stack(torch._foreach_norm(weights, ord=float("inf"))).float()
        except (AttributeError, RuntimeError):
            weight_amaxes = torch.stack([w.abs().amax().float() for w in weights])
        for i, m in enumerate(r.modules):
            m.staged_weight_amax.fill_(weight_amaxes[i].item())
    if r._first_step:
        r._first_step = False

    r.staged_amaxes_3n[0].copy_(torch.stack([m.staged_input_amax for m in r.modules]))
    r.staged_amaxes_3n[1].copy_(torch.stack([m.staged_weight_amax for m in r.modules]))
    r.staged_amaxes_3n[2].copy_(torch.stack([m.staged_grad_amax for m in r.modules]))

    if r.reduce_amax and r.amax_reduce_group is not None:
        torch.distributed.all_reduce(
            r.staged_amaxes_3n,
            op=torch.distributed.ReduceOp.MAX,
            group=r.amax_reduce_group,
        )

    N = r.n
    H = r.history_len
    BLOCK_H = triton.next_power_of_2(H) if H <= 1024 else 1024

    _fused_delayed_scale_update_kernel[(N, 3)](
        r.amax_history,
        r.staged_amaxes_3n,
        r.scales_3n,
        r.fp8_maxes,
        r._history_idx,
        N=N,
        H=H,
        use_max_algo=(r.algo == "max"),
        BLOCK_H=BLOCK_H,
        FP32_MAX=torch.finfo(torch.float32).max,
        FILTER_ZEROS=r.filter_zeros,
    )

    # See _fast_update_scales for the rationale behind _foreach_copy_ +
    # unbind here. r.scales_3n is the (3, N) registry-batched scale tensor
    # produced by the Triton kernel; rows 0/1/2 = input/weight/grad.
    torch._foreach_copy_(
        [m.scale_input for m in r.modules],
        list(r.scales_3n[0].unbind()),
    )
    torch._foreach_copy_(
        [m.scale_weight for m in r.modules],
        list(r.scales_3n[1].unbind()),
    )
    torch._foreach_copy_(
        [m.scale_grad for m in r.modules],
        list(r.scales_3n[2].unbind()),
    )

    r._history_idx = (r._history_idx + 1) % H
    for m in r.modules:
        m._history_idx = r._history_idx


# ---------------------------------------------------------------------------
# Async allreduce split: stage+launch / wait+compute
# ---------------------------------------------------------------------------


@torch.no_grad()
def _stage_and_launch_async_allreduce(registry):
    """Stage per-module amaxes into r.staged_amaxes_3n and launch async allreduce.

    Returns the async work handle (or None if allreduce is not needed).
    Must be called after backward completes so staged_*_amax buffers are
    populated.
    """
    r = registry
    r.staged_amaxes_3n[0].copy_(torch.stack([m.staged_input_amax for m in r.modules]))
    r.staged_amaxes_3n[1].copy_(torch.stack([m.staged_weight_amax for m in r.modules]))
    r.staged_amaxes_3n[2].copy_(torch.stack([m.staged_grad_amax for m in r.modules]))

    if r.reduce_amax and r.amax_reduce_group is not None:
        handle = torch.distributed.all_reduce(
            r.staged_amaxes_3n,
            op=torch.distributed.ReduceOp.MAX,
            group=r.amax_reduce_group,
            async_op=True,
        )
        return handle
    return None


@torch.no_grad()
def _wait_and_compute_scales(registry, handle):
    """Wait on async allreduce handle and compute new scales from reduced amaxes.

    For most_recent + history_len=1, computes scales directly from
    staged_amaxes_3n. For history-based, dispatches the fused Triton kernel.
    Handles weight caching if enabled.
    """
    r = registry

    if handle is not None:
        handle.wait()

    if r.algo == "most_recent" and r.history_len == 1:
        staged_input = r.staged_amaxes_3n[0]
        staged_weight = r.staged_amaxes_3n[1]
        staged_grad = r.staged_amaxes_3n[2]

        scale_input = torch.stack([m.scale_input for m in r.modules])
        scale_weight = torch.stack([m.scale_weight for m in r.modules])
        scale_grad = torch.stack([m.scale_grad for m in r.modules])

        new_input = _batch_compute_scales(staged_input, r.fwd_max, scale_input)
        new_weight = _batch_compute_scales(staged_weight, r.fwd_max, scale_weight)
        new_grad = _batch_compute_scales(staged_grad, r.bwd_max, scale_grad)

        for i, m in enumerate(r.modules):
            m.scale_input.fill_(new_input[i].item())
            m.scale_weight.fill_(new_weight[i].item())
            m.scale_grad.fill_(new_grad[i].item())
    else:
        N = r.n
        H = r.history_len
        BLOCK_H = triton.next_power_of_2(H) if H <= 1024 else 1024

        _fused_delayed_scale_update_kernel[(N, 3)](
            r.amax_history,
            r.staged_amaxes_3n,
            r.scales_3n,
            r.fp8_maxes,
            r._history_idx,
            N=N,
            H=H,
            use_max_algo=(r.algo == "max"),
            BLOCK_H=BLOCK_H,
            FP32_MAX=torch.finfo(torch.float32).max,
            FILTER_ZEROS=r.filter_zeros,
        )

        for i, m in enumerate(r.modules):
            m.scale_input.fill_(r.scales_3n[0, i].item())
            m.scale_weight.fill_(r.scales_3n[1, i].item())
            m.scale_grad.fill_(r.scales_3n[2, i].item())

        r._history_idx = (r._history_idx + 1) % H
        for m in r.modules:
            m._history_idx = r._history_idx


# ---------------------------------------------------------------------------
# Weight extraction helper — called outside autograd.Function to avoid
# tensor-subclass isinstance checks inside the compiled graph.
# ---------------------------------------------------------------------------


def _extract_fp8_weight(weight, fp8_dtype):
    """Return (fp8_data, scale_inv) from weight, handling FP8UnshardedWeightTensor."""
    from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
        FP8UnshardedWeightTensor,
    )

    if isinstance(weight, FP8UnshardedWeightTensor):
        return weight.get_fp8_data_and_scale_inv()
    return _quantize_fp8_tw(weight, fp8_dtype)


# ---------------------------------------------------------------------------
# Granularity -> Function dispatch map
# ---------------------------------------------------------------------------

_FP8_FN_MAP = {
    ScalingGranularity.TENSORWISE: OpaqueFP8LinearTensorwiseFunction,
    ScalingGranularity.ROWWISE: FP8LinearRowwiseFunction,
    ScalingGranularity.BLOCKWISE: FP8LinearBlockwiseFunction,
}

# Public alias
FP8LinearTensorwiseFunction = OpaqueFP8LinearTensorwiseFunction


# ---------------------------------------------------------------------------
# FP8-aware parallel linear layers
# ---------------------------------------------------------------------------


class _Float8LinearMixin:
    """Shared per-module FP8 behavior for Float8{Column,Row}ParallelLinear.

    This is a mixin (not an nn.Module). Each concrete class also inherits a
    Megatron ``*ParallelLinear`` and MUST list this mixin FIRST in its bases so
    the ``_apply`` / ``_forward_impl`` overrides below take precedence over the
    Megatron base (otherwise FP8 would be silently disabled). The concrete
    ``__init__`` calls ``self._init_fp8_state()`` after ``super().__init__()``.

    Requires: tensor_model_parallel_size=1, gradient_accumulation_fusion=False,
    sequence_parallel=False. Fails fast at construction if violated.

    Tensorwise uses the setup_context autograd pattern with pre-extracted FP8
    weight data for graph-break-free torch.compile tracing.
    Rowwise and blockwise still use @allow_in_graph.
    """

    def _init_fp8_state(self):
        cls = type(self).__name__
        if self.config.tensor_model_parallel_size != 1:
            raise ValueError(
                f"{cls} requires tensor_model_parallel_size=1. "
                f"Got {self.config.tensor_model_parallel_size}."
            )
        if self.gradient_accumulation_fusion:
            raise ValueError(f"{cls} requires gradient_accumulation_fusion=False.")
        if self.sequence_parallel:
            raise ValueError(f"{cls} requires sequence_parallel=False.")
        if self.config.fp8 is None:
            raise ValueError(f"{cls} requires config.fp8 to be set (e.g. 'e4m3').")

        self._fp8_config = _build_fp8_config(self.config)
        self._fp8_fn = _FP8_FN_MAP[self._fp8_config.granularity]

        if self._fp8_config.granularity == ScalingGranularity.TENSORWISE:
            self._fp8_fwd_dtype = _get_fp8_dtype(self._fp8_config.format, is_fwd=True)
            self._fp8_bwd_dtype = _get_fp8_dtype(self._fp8_config.format, is_fwd=False)
            self._fp8_gran_value = ScalingGranularity.TENSORWISE.value
            self._fp8_backend_value = BackendType.HIPBLASLT.value
            # Native layout (NN/TN) by default; forced-NT is opt-in per config.
            # Read once at layer init so it stays a compile-time constant.
            self._force_nt = getattr(self.config, "fp8_force_nt_layout", False)
            self._use_delayed_scaling = (
                getattr(self.config, "fp8_scaling_strategy", "dynamic") == "delayed"
                or self.config.fp8_recipe == Fp8Recipe.delayed
            )
            if self._use_delayed_scaling:
                _init_delayed_scaling_state(self)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        if getattr(self, "_use_delayed_scaling", False):
            for name in [
                "scale_input",
                "scale_weight",
                "scale_grad",
                "amax_history_input",
                "amax_history_weight",
                "amax_history_grad",
                "staged_input_amax",
                "staged_grad_amax",
                "staged_weight_amax",
            ]:
                buf = getattr(self, name, None)
                if buf is not None and buf.dtype != torch.float32:
                    self._buffers[name] = buf.data.float()
        return result

    def _forward_impl(self, input, weight, *args, **kwargs):
        bias = kwargs.get("bias", None)

        if self._fp8_config.granularity == ScalingGranularity.TENSORWISE:
            if self._use_delayed_scaling:
                result = DelayedFP8LinearTensorwiseFunction.apply(
                    input,
                    weight,
                    self.scale_input,
                    self.scale_weight,
                    self.scale_grad,
                    self.staged_input_amax,
                    self.staged_weight_amax,
                    self.staged_grad_amax,
                    self._fp8_fwd_dtype,
                    self._fp8_bwd_dtype,
                    self._fp8_gran_value,
                    self._fp8_backend_value,
                    self._force_nt,
                )
                output = result[0]
            else:
                weight_fp8, weight_scale_inv = _extract_fp8_weight(
                    weight,
                    self._fp8_fwd_dtype,
                )
                result = self._fp8_fn.apply(
                    input,
                    weight,
                    weight_fp8,
                    weight_scale_inv,
                    self._fp8_fwd_dtype,
                    self._fp8_bwd_dtype,
                    self._fp8_gran_value,
                    self._fp8_backend_value,
                    self._force_nt,
                )
                output = result[0]
        else:
            output = self._fp8_fn.apply(input, weight, self._fp8_config)

        if bias is not None:
            output = output + bias
        return output


class Float8ColumnParallelLinear(_Float8LinearMixin, ColumnParallelLinear):
    """ColumnParallelLinear with per-module FP8. torch.compile friendly.

    Shared FP8 behavior lives in :class:`_Float8LinearMixin`, which is listed
    first so its ``_apply`` / ``_forward_impl`` overrides take effect.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_fp8_state()


class Float8RowParallelLinear(_Float8LinearMixin, RowParallelLinear):
    """RowParallelLinear with per-module FP8. torch.compile friendly.

    Shared FP8 behavior lives in :class:`_Float8LinearMixin`, which is listed
    first so its ``_apply`` / ``_forward_impl`` overrides take effect.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_fp8_state()
