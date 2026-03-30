###############################################################################
# Some parts of this code are copied and modified from
# Sea AI Lab's zero-bubble-pipeline-parallelism project
# (https://github.com/sail-sg/zero-bubble-pipeline-parallelism).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import functools
from typing import Callable, Tuple

import grouped_gemm
import torch
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_impl
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_impl as _grouped_gemm_fp8_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_impl,
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quant_fp8_blockwise_for_weight_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

from primus.backends.megatron.core.pipeline_parallel.wgrad_adapter import (
    insert_wgrad_func_into_cache,
)


def _get_fp8_dtype(format: Format, is_fwd: bool):
    """Map FP8 format + stage to the concrete torch dtype."""
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


class LinearWithWeightGradientStore(torch.autograd.Function):
    """Linear layer split wgrad and winput"""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        ctx.use_bias = bias is not None
        ctx.save_for_backward(input, weight)
        ctx.weight_main_grad = weight.main_grad

        output = gemm_impl(
            input,
            False,
            weight,
            True,
            input.dtype,
            False,
            default_backend=BackendType.HIPBLASLT.value,
        )
        if ctx.use_bias:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        weight.main_grad = ctx.weight_main_grad

        grad_input = gemm_impl(
            grad_output,
            False,
            weight,
            False,
            input.dtype,
            False,
            default_backend=BackendType.HIPBLASLT.value,
        )
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        try:
            import fused_weight_gradient_mlp_cuda
        except:
            raise ImportError("fused_weight_gradient_mlp_cuda is not available")

        def pre_process(_grad_output_, _input_, async_op=True):
            # gather from SP region if sequence parallel if needed
            return _grad_output_, _input_, None

        def process_wgrad(_weight, _grad_output, _total_input, _handle, wgrad_gemm_accum_func=None):
            wgrad_gemm_accum_func(_total_input, _grad_output, _weight.main_grad)

        if weight.main_grad.dtype == torch.float32:
            wgrad_gemm_accum_func = fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32
        else:
            wgrad_gemm_accum_func = fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16

        insert_wgrad_func_into_cache(
            weight,
            functools.partial(pre_process, grad_output, input),
            functools.partial(process_wgrad, weight, wgrad_gemm_accum_func=wgrad_gemm_accum_func),
        )

        return grad_input, None, grad_bias, None, None


def gemm_with_weight_gradient_store(input, weight, bias):
    return LinearWithWeightGradientStore.apply(input, weight, bias)


class GroupedLinearWithWeightGradientStore(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_b: bool,
        weight_reshape_size: Tuple | None,
        group_gemm_backend_func: Callable,
        wgrad_gemm_backend_func: Callable | None = None,
    ):
        if wgrad_gemm_backend_func is None:
            wgrad_gemm_backend_func = group_gemm_backend_func
        ctx.use_main_grad = hasattr(weight, "main_grad") and weight.main_grad is not None
        if ctx.use_main_grad:
            ctx.weight_main_grad = weight.main_grad
        ctx.weight_shape_ori = weight.shape
        ctx.group_gemm_backend_func = group_gemm_backend_func
        ctx.wgrad_gemm_backend_func = wgrad_gemm_backend_func

        if weight_reshape_size is not None:
            weight = weight.view(*weight_reshape_size)

        ctx.group_offs = group_offs
        ctx.save_for_backward(input, weight, group_lens)

        output = group_gemm_backend_func(
            input,
            weight,
            group_lens,
            trans_a=False,
            trans_b=trans_b,
        )

        ctx.trans_a = False
        ctx.trans_b = trans_b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, group_lens = ctx.saved_tensors
        group_gemm_backend_func = ctx.group_gemm_backend_func
        if ctx.use_main_grad:
            weight.main_grad = ctx.weight_main_grad
        grad_a = group_gemm_backend_func(
            grad_output,
            weight,
            group_lens,
            trans_a=False,
            trans_b=not ctx.trans_b,
        )

        def pre_process(_grad_output_, _input_, trans_b, async_op=True):
            # gather from SP region if sequence parallel if needed
            if trans_b:
                return _grad_output_, _input_, None
            else:
                return _input_, _grad_output_, None

        def process_wgrad(_weight, _weight_shape_ori, _grad_output, _total_input, handle=None):
            _wgrad = ctx.wgrad_gemm_backend_func(
                _grad_output,
                _total_input,
                group_lens,
                trans_a=True,
                trans_b=False,
            )
            _wgrad = _wgrad.view(_weight_shape_ori)
            if ctx.use_main_grad:
                with torch.no_grad():
                    _weight.main_grad.add_(_wgrad)

        insert_wgrad_func_into_cache(
            weight,
            functools.partial(pre_process, grad_output, input, ctx.trans_b),
            functools.partial(process_wgrad, weight, ctx.weight_shape_ori),
        )

        return grad_a, None, None, None, None, None, None, None


def grouped_gemm_with_weight_gradient_store(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = False,
    num_cu: int | None = None,
    weight_reshape_size: Tuple | None = None,
    gg_backend: str = "turbo-gg",
):
    if gg_backend == "turbo-gg":
        if group_offs is None:
            group_offs = grouped_gemm_compute_offs(group_lens)
        group_gemm_backend_func = functools.partial(
            grouped_gemm_impl,
            group_offs=group_offs,
            num_cu=num_cu,
            default_backend=BackendType.CK.value,
        )
        wgrad_gemm_backend_func = functools.partial(
            grouped_gemm_variable_k_impl,
            group_offs=group_offs,
            num_cu=num_cu,
            default_backend=BackendType.CK.value,
        )
    elif gg_backend == "lagacy-gg":
        group_gemm_backend_func = grouped_gemm.backend.gmm
        wgrad_gemm_backend_func = grouped_gemm.backend.gmm
    else:
        raise NotImplementedError(f"Grouped gemm backend {gg_backend} not implemented")

    return GroupedLinearWithWeightGradientStore.apply(
        input,
        weight,
        group_lens,
        group_offs,
        trans_b,
        weight_reshape_size,
        group_gemm_backend_func,
        wgrad_gemm_backend_func,
    )


class LinearFP8WithWeightGradientStore(torch.autograd.Function):
    """Linear layer with FP8 forward/backward and split wgrad for pipeline parallelism.

    Mirrors the backward logic of Primus-Turbo's FP8 GEMM autograd Functions
    (FP8GemmTensorFunction, FP8GemmBlockFunction) but defers weight gradient
    computation via insert_wgrad_func_into_cache for pipeline parallelism.

    Supports TENSORWISE and BLOCKWISE scaling granularity.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        quant_config,
    ):
        granularity = quant_config.granularity
        fwd_dtype = _get_fp8_dtype(quant_config.format, True)
        out_dtype = input.dtype

        ctx.config = quant_config
        ctx.out_dtype = out_dtype
        ctx.weight_ref = weight
        ctx.weight_main_grad = weight.main_grad

        if granularity == ScalingGranularity.TENSORWISE:
            a_fp8, a_scale_inv = quantize_fp8(input, fwd_dtype, granularity)
            b_fp8, b_scale_inv = quantize_fp8(weight, fwd_dtype, granularity)
            out = gemm_fp8_impl(
                a_fp8,
                a_scale_inv,
                False,
                b_fp8,
                b_scale_inv,
                True,
                out_dtype,
                False,
                granularity=granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
            )
            ctx.save_for_backward(a_fp8, a_scale_inv, b_fp8, b_scale_inv)

        elif granularity == ScalingGranularity.BLOCKWISE:
            block_size = quant_config.block_size
            a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
                input,
                fwd_dtype,
                axis=1,
                block_size=block_size,
            )
            b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(
                weight,
                fwd_dtype,
                block_size=block_size,
            )
            out = gemm_fp8_impl(
                a_fp8_row,
                a_scale_inv_row,
                False,
                b_fp8,
                b_scale_inv,
                True,
                out_dtype,
                False,
                granularity=granularity.value,
                default_backend=BackendType.CK.value,
            )
            ctx.save_for_backward(input, b_fp8, b_scale_inv)

        else:
            raise ValueError(
                f"FP8 split-wgrad for linear does not support granularity: {granularity}. "
                f"Supported: TENSORWISE, BLOCKWISE."
            )

        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        config = ctx.config
        granularity = config.granularity
        weight = ctx.weight_ref
        weight.main_grad = ctx.weight_main_grad
        out_dtype = ctx.out_dtype
        bwd_dtype = _get_fp8_dtype(config.format, False)

        if granularity == ScalingGranularity.TENSORWISE:
            a_fp8, a_scale_inv, b_fp8, b_scale_inv = ctx.saved_tensors
            grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, bwd_dtype, granularity)

            grad_input = gemm_fp8_impl(
                grad_out_fp8,
                grad_out_scale_inv,
                False,
                b_fp8,
                b_scale_inv,
                False,
                out_dtype,
                False,
                granularity=granularity.value,
                default_backend=BackendType.HIPBLASLT.value,
            )

            def preprocess(async_op=True):
                return (a_fp8, a_scale_inv), (grad_out_fp8, grad_out_scale_inv), None

            def process_wgrad(
                _weight,
                _out_dtype,
                _granularity_val,
                _a_data,
                _grad_out_data,
                handle=None,
            ):
                _a_fp8, _a_scale_inv = _a_data
                _grad_out_fp8, _grad_out_scale_inv = _grad_out_data
                b_grad = gemm_fp8_impl(
                    _a_fp8,
                    _a_scale_inv,
                    True,
                    _grad_out_fp8,
                    _grad_out_scale_inv,
                    False,
                    _out_dtype,
                    True,
                    granularity=_granularity_val,
                    default_backend=BackendType.HIPBLASLT.value,
                )
                with torch.no_grad():
                    _weight.main_grad.add_(b_grad)

            insert_wgrad_func_into_cache(
                weight,
                preprocess,
                functools.partial(process_wgrad, weight, out_dtype, granularity.value),
            )

        elif granularity == ScalingGranularity.BLOCKWISE:
            input_orig, b_fp8, b_scale_inv = ctx.saved_tensors
            block_size = config.block_size
            a_bwd_dtype = _get_fp8_dtype(config.format, False)

            grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_impl(
                grad_out,
                bwd_dtype,
                -1,
                block_size,
            )
            grad_input = gemm_fp8_impl(
                grad_out_fp8_row,
                grad_out_scale_inv_row,
                False,
                b_fp8,
                b_scale_inv,
                False,
                out_dtype,
                False,
                granularity=granularity.value,
                default_backend=BackendType.CK.value,
            )

            grad_out_fp8_col, grad_out_scale_inv_col = quant_fp8_blockwise_impl(
                grad_out,
                bwd_dtype,
                -2,
                block_size,
            )
            a_fp8_col, a_scale_inv_col = quant_fp8_blockwise_impl(
                input_orig,
                a_bwd_dtype,
                axis=0,
                block_size=block_size,
            )

            def preprocess(async_op=True):
                return (a_fp8_col, a_scale_inv_col), (grad_out_fp8_col, grad_out_scale_inv_col), None

            def process_wgrad(
                _weight,
                _out_dtype,
                _granularity_val,
                _a_data,
                _grad_out_data,
                handle=None,
            ):
                _a_fp8_col, _a_scale_inv_col = _a_data
                _grad_out_fp8_col, _grad_out_scale_inv_col = _grad_out_data
                b_grad = gemm_fp8_impl(
                    _a_fp8_col,
                    _a_scale_inv_col,
                    True,
                    _grad_out_fp8_col,
                    _grad_out_scale_inv_col,
                    False,
                    _out_dtype,
                    True,
                    granularity=_granularity_val,
                    default_backend=BackendType.CK.value,
                )
                with torch.no_grad():
                    _weight.main_grad.add_(b_grad)

            insert_wgrad_func_into_cache(
                weight,
                preprocess,
                functools.partial(process_wgrad, weight, out_dtype, granularity.value),
            )

        return grad_input, None, None


def gemm_fp8_with_weight_gradient_store(input, weight, fp8_gemm_func, quant_config_data):
    return LinearFP8WithWeightGradientStore.apply(input, weight, quant_config_data)


class GroupedLinearFP8WithWeightGradientStore(torch.autograd.Function):
    """Grouped linear with FP8 forward/backward and split wgrad for pipeline parallelism.

    Mirrors Primus-Turbo's GroupedGemmFP8TensorFunc / GroupedGemmFP8BlockFunc
    backward logic but defers weight gradient computation for pipeline parallelism.

    Supports TENSORWISE and BLOCKWISE scaling granularity.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        group_lens: torch.Tensor,
        trans_b: bool,
        weight_reshape_size: Tuple | None,
        quant_config,
    ):
        granularity = quant_config.granularity
        fwd_dtype = _get_fp8_dtype(quant_config.format, True)
        out_dtype = input.dtype

        ctx.config = quant_config
        ctx.out_dtype = out_dtype
        ctx.trans_b = trans_b
        ctx.use_main_grad = hasattr(weight, "main_grad") and weight.main_grad is not None
        if ctx.use_main_grad:
            ctx.weight_main_grad = weight.main_grad
        ctx.weight_ref = weight
        ctx.weight_shape_ori = weight.shape

        if weight_reshape_size is not None:
            weight = weight.view(*weight_reshape_size)

        group_offs = grouped_gemm_compute_offs(group_lens)

        if granularity == ScalingGranularity.TENSORWISE:
            a_fp8, a_scale_inv = quantize_fp8(input, fwd_dtype, granularity)
            b_fp8, b_scale_inv = quantize_fp8(weight, fwd_dtype, granularity)
            out = _grouped_gemm_fp8_impl(
                a_fp8,
                b_fp8,
                a_scale_inv,
                b_scale_inv,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=trans_b,
                out_dtype=out_dtype,
                granularity=granularity.value,
                num_cu=None,
                default_backend=BackendType.CK.value,
            )
            ctx.save_for_backward(a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs)

        elif granularity == ScalingGranularity.BLOCKWISE:
            block_size = quant_config.block_size
            a_fp8_row, a_scale_inv_row = quant_fp8_blockwise_impl(
                input,
                fwd_dtype,
                axis=1,
                block_size=block_size,
            )
            b_fp8, b_scale_inv = quant_fp8_blockwise_for_weight_impl(
                weight,
                fwd_dtype,
                block_size=block_size,
            )
            out = _grouped_gemm_fp8_impl(
                a_fp8_row,
                b_fp8,
                a_scale_inv_row,
                b_scale_inv,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=trans_b,
                out_dtype=out_dtype,
                granularity=granularity.value,
                num_cu=None,
                default_backend=BackendType.CK.value,
            )
            a_fp8_col, a_scale_inv_col, _, _ = quant_fp8_blockwise_segment_m_impl(
                input,
                fwd_dtype,
                block_size,
                group_lens,
                group_offs,
            )
            ctx.save_for_backward(
                a_fp8_col,
                a_scale_inv_col,
                b_fp8,
                b_scale_inv,
                group_lens,
                group_offs,
            )

        else:
            raise ValueError(
                f"FP8 split-wgrad for grouped GEMM does not support granularity: {granularity}. "
                f"Supported: TENSORWISE, BLOCKWISE."
            )

        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        config = ctx.config
        granularity = config.granularity
        weight = ctx.weight_ref
        use_main_grad = ctx.use_main_grad
        if use_main_grad:
            weight.main_grad = ctx.weight_main_grad
        weight_shape_ori = ctx.weight_shape_ori
        trans_b = ctx.trans_b
        out_dtype = ctx.out_dtype
        bwd_dtype = _get_fp8_dtype(config.format, False)

        if granularity == ScalingGranularity.TENSORWISE:
            a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors
            grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, bwd_dtype, granularity)

            grad_a = _grouped_gemm_fp8_impl(
                grad_out_fp8,
                b_fp8,
                grad_out_scale_inv,
                b_scale_inv,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=not trans_b,
                out_dtype=out_dtype,
                granularity=granularity.value,
                num_cu=None,
                default_backend=BackendType.CK.value,
            )

            def preprocess(async_op=True):
                return (a_fp8, a_scale_inv), (grad_out_fp8, grad_out_scale_inv), None

            def process_wgrad(
                _weight,
                _weight_shape_ori,
                _use_main_grad,
                _out_dtype,
                _granularity_val,
                _group_lens,
                _group_offs,
                _trans_b,
                _a_data,
                _grad_out_data,
                handle=None,
            ):
                _a_fp8, _a_scale_inv = _a_data
                _grad_out_fp8, _grad_out_scale_inv = _grad_out_data
                b_grad = grouped_gemm_fp8_variable_k_impl(
                    _a_fp8,
                    _grad_out_fp8,
                    _a_scale_inv,
                    _grad_out_scale_inv,
                    _group_lens,
                    _group_offs,
                    trans_a=True,
                    trans_b=False,
                    trans_c=_trans_b,
                    out_dtype=_out_dtype,
                    granularity=_granularity_val,
                    num_cu=None,
                    default_backend=BackendType.CK.value,
                )
                b_grad = b_grad.view(_weight_shape_ori)
                if _use_main_grad:
                    with torch.no_grad():
                        _weight.main_grad.add_(b_grad)

            insert_wgrad_func_into_cache(
                weight,
                preprocess,
                functools.partial(
                    process_wgrad,
                    weight,
                    weight_shape_ori,
                    use_main_grad,
                    out_dtype,
                    granularity.value,
                    group_lens,
                    group_offs,
                    trans_b,
                ),
            )

        elif granularity == ScalingGranularity.BLOCKWISE:
            (
                a_fp8_col,
                a_scale_inv_col,
                b_fp8,
                b_scale_inv,
                group_lens,
                group_offs,
            ) = ctx.saved_tensors
            block_size = config.block_size

            grad_out_fp8_row, grad_out_scale_inv_row = quant_fp8_blockwise_impl(
                grad_out,
                bwd_dtype,
                axis=1,
                block_size=block_size,
            )
            grad_a = _grouped_gemm_fp8_impl(
                grad_out_fp8_row,
                b_fp8,
                grad_out_scale_inv_row,
                b_scale_inv,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=not trans_b,
                out_dtype=out_dtype,
                granularity=granularity.value,
                num_cu=None,
                default_backend=BackendType.CK.value,
            )

            (
                grad_out_fp8_col,
                grad_out_scale_inv_col,
                var_k_group_lens,
                var_k_group_offs,
            ) = quant_fp8_blockwise_segment_m_impl(
                grad_out,
                bwd_dtype,
                block_size,
                group_lens,
                group_offs,
            )

            def preprocess(async_op=True):
                return (
                    (a_fp8_col, a_scale_inv_col),
                    (grad_out_fp8_col, grad_out_scale_inv_col),
                    None,
                )

            def process_wgrad(
                _weight,
                _weight_shape_ori,
                _use_main_grad,
                _out_dtype,
                _granularity_val,
                _var_k_group_lens,
                _var_k_group_offs,
                _trans_b,
                _a_data,
                _grad_out_data,
                handle=None,
            ):
                _a_fp8_col, _a_scale_inv_col = _a_data
                _grad_out_fp8_col, _grad_out_scale_inv_col = _grad_out_data
                b_grad = grouped_gemm_fp8_variable_k_impl(
                    _a_fp8_col,
                    _grad_out_fp8_col,
                    _a_scale_inv_col,
                    _grad_out_scale_inv_col,
                    _var_k_group_lens,
                    _var_k_group_offs,
                    trans_a=True,
                    trans_b=False,
                    trans_c=_trans_b,
                    out_dtype=_out_dtype,
                    granularity=_granularity_val,
                    num_cu=None,
                    default_backend=BackendType.CK.value,
                )
                b_grad = b_grad.view(_weight_shape_ori)
                if _use_main_grad:
                    with torch.no_grad():
                        _weight.main_grad.add_(b_grad)

            insert_wgrad_func_into_cache(
                weight,
                preprocess,
                functools.partial(
                    process_wgrad,
                    weight,
                    weight_shape_ori,
                    use_main_grad,
                    out_dtype,
                    granularity.value,
                    var_k_group_lens,
                    var_k_group_offs,
                    trans_b,
                ),
            )

        return grad_a, None, None, None, None, None


def grouped_gemm_fp8_with_weight_gradient_store(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_lens: torch.Tensor,
    trans_b: bool = False,
    weight_reshape_size: Tuple | None = None,
    quant_config_data=None,
):
    return GroupedLinearFP8WithWeightGradientStore.apply(
        input,
        weight,
        group_lens,
        trans_b,
        weight_reshape_size,
        quant_config_data,
    )
