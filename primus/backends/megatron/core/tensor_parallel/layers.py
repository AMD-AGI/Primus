###############################################################################
# Some parts of this code are copied and modified from
# Sea AI Lab's zero-bubble-pipeline-parallelism project
# (https://github.com/sail-sg/zero-bubble-pipeline-parallelism).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import functools

import torch
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import (
    is_torch_min_version,
    prepare_input_tensors_for_wgrad_compute,
)
from torch.cuda.amp import custom_bwd, custom_fwd

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
    dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False


_GPTOSS_BF16_LM_HEAD_TOKENS = 32768
_GPTOSS_BF16_LM_HEAD_HIDDEN = 2880
_GPTOSS_BF16_LM_HEAD_VOCAB = 128256


def _is_gptoss_bf16_lm_head_tensor(tensor, rows, columns):
    """Match only the trace-locked GPT-OSS BF16 LM-head matrices."""
    return (
        tensor.is_cuda
        and tensor.dtype == torch.bfloat16
        and tensor.is_contiguous()
        and tensor.shape[-1] == columns
        and tensor.numel() == rows * columns
    )


def _is_gptoss_bf16_lm_head_forward(total_input, weight):
    return (
        _is_gptoss_bf16_lm_head_tensor(
            total_input, _GPTOSS_BF16_LM_HEAD_TOKENS, _GPTOSS_BF16_LM_HEAD_HIDDEN
        )
        and weight.shape == (_GPTOSS_BF16_LM_HEAD_VOCAB, _GPTOSS_BF16_LM_HEAD_HIDDEN)
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    )


def _is_gptoss_bf16_lm_head_wgrad(total_input, grad_output, main_grad):
    return _is_gptoss_bf16_lm_head_forward(
        total_input, main_grad
    ) and _is_gptoss_bf16_lm_head_tensor(
        grad_output, _GPTOSS_BF16_LM_HEAD_TOKENS, _GPTOSS_BF16_LM_HEAD_VOCAB
    )


def _turbo_gemm(a, trans_a, b, trans_b, out_dtype, trans_c=False):
    """Call Turbo lazily so non-Turbo Primus users keep their current import path."""
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_impl

    return gemm_impl(
        a,
        trans_a,
        b,
        trans_b,
        out_dtype,
        trans_c,
        default_backend=BackendType.HIPBLASLT.value,
    )


def _turbo_gemm_accum(a, trans_a, b, trans_b, out_dtype, trans_c, out):
    """Call Turbo's beta=1 dense GEMM entry point."""
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_accum_impl

    gemm_accum_impl(
        a,
        trans_a,
        b,
        trans_b,
        out_dtype,
        trans_c,
        out=out,
        default_backend=BackendType.HIPBLASLT.value,
    )


def _gptoss_bf16_lm_head_forward(total_input, weight):
    if not _is_gptoss_bf16_lm_head_forward(total_input, weight):
        return None
    input_2d = total_input.reshape(-1, total_input.shape[-1])
    output_2d = _turbo_gemm(input_2d, False, weight, True, total_input.dtype)
    return output_2d.reshape(*total_input.shape[:-1], weight.shape[0])


def _gptoss_bf16_lm_head_dgrad(grad_output, weight, input_shape):
    if not (
        _is_gptoss_bf16_lm_head_tensor(
            grad_output, _GPTOSS_BF16_LM_HEAD_TOKENS, _GPTOSS_BF16_LM_HEAD_VOCAB
        )
        and weight.shape == (_GPTOSS_BF16_LM_HEAD_VOCAB, _GPTOSS_BF16_LM_HEAD_HIDDEN)
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    ):
        return None
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    grad_input_2d = _turbo_gemm(grad_output_2d, False, weight, False, grad_output.dtype)
    return grad_input_2d.reshape(input_shape)


def _gptoss_bf16_lm_head_wgrad_accum(total_input, grad_output, main_grad):
    if not _is_gptoss_bf16_lm_head_wgrad(total_input, grad_output, main_grad):
        return False
    input_2d = total_input.reshape(-1, total_input.shape[-1])
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    # Match Megatron's fused extension exactly: main_grad += grad_output.T @ input.
    # Turbo canonicalizes this transposed-output contract to the tuned TN kernel.
    _turbo_gemm_accum(
        input_2d,
        True,
        grad_output_2d,
        False,
        main_grad.dtype,
        True,
        main_grad,
    )
    return True


def _wgrad_gemm_accum(total_input, grad_output, main_grad):
    if _gptoss_bf16_lm_head_wgrad_accum(total_input, grad_output, main_grad):
        return
    if main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            total_input, grad_output, main_grad
        )
    elif main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
            total_input, grad_output, main_grad
        )
    else:
        raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.weight_main_grad = weight.main_grad

        if sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        output = _gptoss_bf16_lm_head_forward(total_input, weight)
        if output is None:
            output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        weight.main_grad = ctx.weight_main_grad
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit

        def pre_process(_grad_output_, _input_, async_op=True):
            if ctx.sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                _dim_size = list(_input_.size())
                _dim_size[0] = _dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(_dim_size, _input_.dtype, "mpu")

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation

                _handle = dist_all_gather_func(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                _total_input = all_gather_buffer
                # We do not support all gather grad_output for now (maybe never).
                _grad_output = _grad_output_
                return _grad_output, _total_input, _handle
            else:
                _total_input = _input_
                _grad_output = _grad_output_
                return _grad_output, _total_input, None

        def prepare_for_wgrad_compute(_grad_output, _total_input, _handle):
            if ctx.sequence_parallel and _handle is not None:
                _handle.wait()
            return prepare_input_tensors_for_wgrad_compute(_grad_output, _total_input)

        def process_wgrad(_weight, _grad_output, _total_input, _handle, wgrad_gemm_accum_func=None):
            grad_output_, total_input_ = prepare_for_wgrad_compute(_grad_output, _total_input, _handle)
            wgrad_gemm_accum_func(total_input_, grad_output_, _weight.main_grad)

        from primus.core.pipeline_parallel.handler.wgrad_handler import (
            WGradRunningCache,
        )

        from ..pipeline_parallel.wgrad_adapter import insert_wgrad_func_into_cache
        from ..pipeline_parallel.zerobubble.zbpp_utils import WeightGradStore

        wgrad_compute = not WeightGradStore.split_bw() and (
            WGradRunningCache.cur_minibatch is None and WGradRunningCache.cur_chunk is None
        )
        if grad_output_buffer is not None and wgrad_compute:
            # save to grad_output_buffer only when split_bw is False
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad_compute:
            grad_output, total_input, handle = pre_process(grad_output, input, async_op=wgrad_compute)
        grad_input = _gptoss_bf16_lm_head_dgrad(grad_output, weight, input.shape)
        if grad_input is None:
            grad_input = grad_output.matmul(weight)

        if wgrad_compute:
            grad_output, total_input = prepare_for_wgrad_compute(grad_output, total_input, handle)

        if ctx.allreduce_dgrad:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=wgrad_compute
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.allreduce_dgrad
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = dist_reduce_scatter_func(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=wgrad_compute
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                _wgrad_gemm_accum(total_input, grad_output, weight.main_grad)
            else:
                insert_wgrad_func_into_cache(
                    weight,
                    functools.partial(pre_process, grad_output, input),
                    functools.partial(
                        process_wgrad,
                        weight,
                        wgrad_gemm_accum_func=_wgrad_gemm_accum,
                    ),
                )

            if hasattr(weight, "grad_added_to_main_grad"):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, "zero_out_wgrad", False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            if not wgrad_compute:
                # We need to compute total_input from input since pre_process is skipped
                if ctx.sequence_parallel:
                    world_size = get_tensor_model_parallel_world_size()
                    _dim_size = list(input.size())
                    _dim_size[0] = _dim_size[0] * world_size
                    all_gather_buffer = get_global_memory_buffer().get_tensor(_dim_size, input.dtype, "mpu")
                    dist_all_gather_func(all_gather_buffer, input, group=get_tensor_model_parallel_group())
                    total_input = all_gather_buffer
                else:
                    total_input = input

            grad_output_reshaped = (
                grad_output.reshape(-1, grad_output.shape[-1]) if grad_output.dim() > 2 else grad_output
            )
            total_input_reshaped = (
                total_input.reshape(-1, total_input.shape[-1]) if total_input.dim() > 2 else total_input
            )
            grad_weight = grad_output_reshaped.t().matmul(total_input_reshaped)
        if use_bias:
            grad_bias = grad_output.sum(dim=0) if grad_output.dim() == 2 else grad_output.sum(dim=(0, 1))
        else:
            grad_bias = None

        if ctx.sequence_parallel:
            if handle is not None:
                handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None

        if ctx.allreduce_dgrad:
            if handle is not None:
                handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
