import functools

import torch
from primus_turbo.pytorch.kernels.gemm.gemm_csrc_impl import gemm_impl

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
