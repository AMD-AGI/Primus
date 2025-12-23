# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
from typing import List, Tuple, Union

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.pytorch.distributed import (
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
)
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.module.base import (
    _2X_ACC_DGRAD,
    _2X_ACC_FPROP,
    _2X_ACC_WGRAD,
    get_multi_stream_cublas_workspace,
)
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.utils import (
    assert_dim_for_fp8_exec,
    cast_if_needed,
    clear_tensor_data,
    requires_grad,
)

from primus.backends.megatron.core.pipeline_parallel.wgrad_adapter import (
    insert_wgrad_func_into_cache,
)


class _GroupedLinearWithWGradSplit(torch.autograd.Function):
    """GroupedLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        input_quantizers: List[Quantizer],
        weight_quantizers: List[Quantizer],
        output_quantizers: List[Quantizer],
        grad_output_quantizers: List[Quantizer],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        sequence_parallel: bool,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        module,
        skip_fp8_weight_update,
        *weights_and_biases,
    ) -> torch.Tensor:
        assert fuse_wgrad_accumulation, "fuse_wgrad_accumulation need to be true"
        assert not use_bias, "zero bubble not support bias yet"
        # pylint: disable=missing-function-docstring
        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]
        device = inp.device

        # TODO Support MXFP8  # pylint: disable=fixme
        if fp8 and FP8GlobalStateManager.get_fp8_recipe().mxfp8():
            raise NotImplementedError("GroupedLinear does not yet support MXFP8")
        # TODO Support Float8 Current Scaling  # pylint: disable=fixme
        if fp8 and FP8GlobalStateManager.get_fp8_recipe().float8_current_scaling():
            raise NotImplementedError("GroupedLinear does not yet support Float8 Current Scaling")

        # Make sure input dimensions are compatible
        in_features = weights[0].shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmats = torch.split(inp.view(-1, in_features), m_splits)
        if fp8:
            assert_dim_for_fp8_exec(*inputmats, *weights)

        # Cast input to expected dtype
        inputmats_no_fp8 = [cast_if_needed(mat, activation_dtype) for mat in inputmats]
        inputmats = []

        weight_requires_grad = weights[0].requires_grad

        if input_quantizers[0] is not None:
            for input_quantizer in input_quantizers:
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=(is_grad_enabled and weight_requires_grad),
                )
            columnwise_usage = is_grad_enabled and inp.requires_grad
            if not columnwise_usage:
                columnwise_usage = (
                    is_fp8_activation_recompute_enabled() and not in_fp8_activation_recompute_phase()
                )
            if weight_quantizers[0] is not None:
                for weight_quantizer in weight_quantizers:
                    weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        if output_quantizers[0] is not None:
            for output_quantizer in output_quantizers:
                output_quantizer.set_usage(rowwise=True, columnwise=False)

        if fp8:
            inputmats = tex.fused_multi_quantize(
                inputmats_no_fp8, None, input_quantizers, TE_DType[activation_dtype]
            )
            weights_fp8 = []
            bias_dtype = torch.bfloat16 if activation_dtype == torch.float32 else activation_dtype
            if not isinstance(weights[0], QuantizedTensor):
                # FP8 cast to workspace buffer
                update_workspace = is_first_microbatch is None or is_first_microbatch
                for i in range(num_gemms):
                    weight_fp8 = module.get_weight_workspace(
                        tensor=weights[i],
                        quantizer=weight_quantizers[i],
                        cache_name=(None if is_first_microbatch is None else f"weight{i}"),
                        update_workspace=update_workspace,
                        skip_update_flag=skip_fp8_weight_update,
                    )
                    weights_fp8.append(weight_fp8)
            else:
                weights_fp8 = weights

        else:
            inputmats = inputmats_no_fp8
            bias_dtype = activation_dtype
            weights_fp8 = [cast_if_needed(weight, activation_dtype) for weight in weights]

        biases = [cast_if_needed(bias, bias_dtype) for bias in biases] if use_bias else biases

        out = torch.empty(
            [sum(m_splits), weights_fp8[0].size(0)],
            dtype=activation_dtype,
            device=device,
        )

        _ = general_grouped_gemm(
            weights_fp8,
            inputmats,
            [out],
            activation_dtype,
            get_multi_stream_cublas_workspace(),
            single_output=True,
            m_splits=m_splits,
            bias=biases,
            use_bias=use_bias,
            use_split_accumulator=_2X_ACC_FPROP,
        )

        if fp8_calibration:
            for i in range(num_gemms):
                # amax of input
                for i in range(num_gemms):
                    input_quantizers[i].calibrate(inputmats[i])
                for i in range(num_gemms):
                    weight_quantizers[i].calibrate(weights[i])

        if is_grad_enabled:

            ctx.weights_shape_1 = weights[0].shape[1]

            tensors_to_save, tensor_objects = prepare_for_saving(*inputmats, *weights_fp8, *biases)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.weights_requires_grad = weights[0].requires_grad
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                ctx.main_grads = [weights[i].main_grad for i in range(num_gemms)]
            else:
                ctx.main_grads = [None] * num_gemms
            ctx.device = device
            ctx.grad_output_quantizers = grad_output_quantizers
            ctx.m_splits = m_splits
            ctx.num_gemms = num_gemms
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.inp_shape = inp.shape
            ctx.requires_dgrad = inp.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, weights[0], biases[0]):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors or FP8GlobalStateManager.is_first_fp8_module()
                )

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        with torch.cuda.nvtx.range("_GroupedLinear_backward"):
            saved_tensors = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
            N = ctx.num_gemms
            inputmats = saved_tensors[:N]
            weights = saved_tensors[N : 2 * N]
            biases = saved_tensors[2 * N : 3 * N]
            main_grads = ctx.main_grads

            # preprocess grad_output

            grad_output = grad_output.contiguous()
            grad_output_mats = torch.split(grad_output.view(-1, grad_output.shape[-1]), ctx.m_splits)
            grad_output = [None] * ctx.num_gemms
            grad_biases = [None] * ctx.num_gemms
            if ctx.fp8:
                if ctx.use_bias:
                    for i in range(ctx.num_gemms):
                        grad_biases[i], grad_output[i] = tex.bgrad_quantize(
                            grad_output_mats[i], ctx.grad_output_quantizers[i]
                        )
                else:
                    grad_output = tex.fused_multi_quantize(
                        grad_output_mats,
                        None,
                        ctx.grad_output_quantizers,
                        TE_DType[ctx.activation_dtype],
                    )
            else:
                grad_output = grad_output_mats

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.requires_dgrad:
                dgrad = torch.empty(
                    (sum(ctx.m_splits), ctx.weights_shape_1),
                    dtype=ctx.activation_dtype,
                    device=ctx.device,
                )

                general_grouped_gemm(
                    weights,
                    grad_output,
                    [dgrad],
                    ctx.activation_dtype,
                    get_multi_stream_cublas_workspace(),
                    single_output=True,
                    layout="NN",
                    m_splits=ctx.m_splits,
                    grad=True,
                    use_split_accumulator=_2X_ACC_DGRAD,
                )

            if ctx.weights_requires_grad:
                if ctx.fuse_wgrad_accumulation:
                    wgrad_list = main_grads
                else:
                    wgrad_list = [
                        torch.empty(w.size(), dtype=ctx.activation_dtype, device=ctx.device) for w in weights
                    ]

                def handle_custom_ddp_from_mcore(w, wgrad):
                    if ctx.weights_requires_grad:
                        if ctx.fuse_wgrad_accumulation and hasattr(w, "grad_added_to_main_grad"):
                            w.grad_added_to_main_grad = True
                            if getattr(w, "zero_out_wgrad", False):
                                wgrad = torch.zeros(
                                    w.main_grad.shape,
                                    dtype=w.dtype,
                                    device=torch.cuda.current_device(),
                                    requires_grad=False,
                                )
                            else:
                                wgrad = torch.empty(
                                    w.main_grad.shape,
                                    dtype=w.dtype,
                                    device=torch.cuda.current_device(),
                                    requires_grad=False,
                                )
                        elif ctx.fuse_wgrad_accumulation:
                            wgrad = None
                    else:
                        wgrad = None
                    return wgrad

                def pre_process(_grad_output_, _input_, async_op=True):
                    return _grad_output_, _input_, None

                kargs_dict = {
                    "out_dtype": ctx.activation_dtype,
                    "workspaces": get_multi_stream_cublas_workspace(),
                    "layout": "NT",
                    "grad": True,
                    "m_splits": ctx.m_splits,
                    "use_bias": ctx.use_bias if grad_biases[0] is None else None,
                    "bias": biases,
                    "use_split_accumulator": _2X_ACC_WGRAD,
                    "accumulate": accumulate_wgrad_into_param_main_grad,
                }

                def process_wgrad(wgrad_list, kargs_dict, grad_output, inputmats, handle=None):
                    _, grad_biases_, _ = general_grouped_gemm(
                        inputmats, grad_output, wgrad_list, **kargs_dict
                    )
                    for i in range(ctx.num_gemms):
                        if grad_biases[i] is None:
                            grad_biases[i] = grad_biases_[i]
                    del grad_biases_
                    clear_tensor_data(*inputmats)
                    wgrad_list = [
                        handle_custom_ddp_from_mcore(w, wgrad) for w, wgrad in zip(weights, wgrad_list)
                    ]

                insert_wgrad_func_into_cache(
                    wgrad_list,
                    functools.partial(pre_process, grad_output, inputmats),
                    functools.partial(process_wgrad, wgrad_list, kargs_dict),
                )

                wgrad_list_return = [None] * ctx.num_gemms
            else:
                wgrad_list_return = [None] * ctx.num_gemms

            if not ctx.use_bias:
                grad_biases = [None] * ctx.num_gemms

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
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
            None,
            None,
            None,
            None,  # is_grad_enabled
            None,  # is_grad_enabled
            *wgrad_list_return,
            *grad_biases,
        )
