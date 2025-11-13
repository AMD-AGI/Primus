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
from primus_turbo.pytorch.kernels.gemm.gemm_csrc_impl import gemm_impl
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_csrc_impl import (
    grouped_gemm_csrc_impl,
    grouped_gemm_variable_k_csrc_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)

from primus.backends.megatron.core.pipeline_parallel.zerobubble.zbpp_utils import (
    WeightGradStore,
)


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

        output = gemm_impl(input, False, weight, True, input.dtype, False)
        if ctx.use_bias:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        weight.main_grad = ctx.weight_main_grad

        grad_input = gemm_impl(grad_output, False, weight, False, input.dtype, False)
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

        WeightGradStore.put(
            weight,
            functools.partial(pre_process, grad_output, input),
            functools.partial(
                process_wgrad,
                weight,
                wgrad_gemm_accum_func=wgrad_gemm_accum_func,
            ),
        )
        # grad_weight = gemm_impl(grad_output.t(), input)

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
        ctx.use_main_grad = hasattr(weight, 'main_grad') and weight.main_grad is not None
        if ctx.use_main_grad:
            ctx.weight_main_grad = weight.main_grad
        ctx.weight_shape_ori = weight.shape
        ctx.group_gemm_backend_func = group_gemm_backend_func
        ctx.wgrad_gemm_backend_func = wgrad_gemm_backend_func

        if weight_reshape_size is not None:
            weight = weight.view(*weight_reshape_size)

        ctx.save_for_backward(input, weight, group_lens, group_offs)

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
        input, weight, group_lens, group_offs = ctx.saved_tensors
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

        WeightGradStore.put(
            weight,
            functools.partial(pre_process, grad_output, input, ctx.trans_b),
            functools.partial(
                process_wgrad,
                weight,
                ctx.weight_shape_ori,
            ),
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
            grouped_gemm_csrc_impl, group_offs=group_offs, num_cu=num_cu
        )
        wgrad_gemm_backend_func = functools.partial(
            grouped_gemm_variable_k_csrc_impl, group_offs=group_offs, num_cu=num_cu
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
