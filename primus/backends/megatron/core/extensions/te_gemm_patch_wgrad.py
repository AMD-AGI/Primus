# This file was modified for portability to AMDGPU
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import functools
from functools import reduce
from operator import mul as multiply_op
from typing import Optional, Tuple, Union

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.cpu_offload import set_offloading_param
from transformer_engine.pytorch.distributed import (
    _fsdp_gather_tensors,
    _fsdp_scatter_tensors,
    allreduce,
    gather_along_first_dim,
    get_distributed_world_size,
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
    reduce_scatter_along_first_dim,
)
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.module._common import _fix_gathered_fp8_transpose
from transformer_engine.pytorch.module.base import (
    _2X_ACC_DGRAD,
    _2X_ACC_FPROP,
    _2X_ACC_WGRAD,
    TransformerEngineBaseModule,
    get_ub,
    get_workspace,
)
from transformer_engine.pytorch.rocm_utils import (
    clear_fp8_weight_transpose_cache,
    create_fp8_weight_transpose_cache,
)
from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import (
    MXFP8TensorBase,
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
    non_tn_fp8_gemm_supported,
    nvtx_range_pop,
    nvtx_range_push,
    requires_grad,
)

from primus.backends.megatron.core.pipeline_parallel.wgrad_adapter import (
    insert_wgrad_func_into_cache,
)


class _LinearWithWGradSplit(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        inp: torch.Tensor,
        bias: Optional[torch.Tensor],
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        input_quantizer: Optional[Quantizer],
        weight_quantizer: Optional[Quantizer],
        output_quantizer: Optional[Quantizer],
        grad_output_quantizer: Optional[Quantizer],
        grad_input_quantizer: Optional[Quantizer],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        is_grad_enabled: bool,
        ub_overlap_rs_fprop: bool,
        ub_overlap_ag_dgrad: bool,
        ub_overlap_ag_fprop: bool,
        ub_overlap_rs_dgrad: bool,
        ub_bulk_dgrad: bool,
        ub_bulk_wgrad: bool,
        ub_name: str,
        fp8_output: bool,  # pylint: disable=unused-argument
        fsdp_group: Union[dist_group_type, None],
        module: torch.nn.Module,
        skip_fp8_weight_update: bool,
        keep_fp8_weight_transpose_cache: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        assert not ub_bulk_wgrad, "not support for ZeroBubble"
        assert bias is None, "not support bias yet"

        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.forward"
        if ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ub_name}"

        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        inp_shape = inp.shape
        assert inp_shape[-1] == in_features, "GEMM not possible"

        tp_world_size = get_distributed_world_size(tp_group)
        backward_needs_input = is_grad_enabled and weight.requires_grad

        # Prepare input tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        nvtx_range_push(f"{nvtx_label}.input_cast_comm")
        inputmat = inp.view(-1, in_features)
        inputmat_total = None
        with_input_all_gather_nccl = (
            parallel_mode == "column" and sequence_parallel and not ub_overlap_ag_fprop
        )
        own_quantized_input = False
        if fp8:
            assert_dim_for_fp8_exec(inputmat, weight)
            if any([ub_overlap_ag_fprop, ub_overlap_rs_fprop]) and not (
                FP8GlobalStateManager.get_fp8_recipe().float8_per_tensor_scaling()
            ):
                raise NotImplementedError(
                    "Comm+GEMM overlap is only supported with FP8 delayed scaling or per-tensor"
                    " current scaling"
                )

            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            if with_input_all_gather_nccl:
                assert not isinstance(inputmat, QuantizedTensor), "All gather of fp8 input is not supported"
                input_quantizer.set_usage(rowwise=True, columnwise=False)
                inputmat_total, _ = gather_along_first_dim(
                    inputmat,
                    tp_group,
                    quantizer=input_quantizer,
                )
            else:
                if FP8GlobalStateManager.get_fp8_recipe().float8_per_tensor_scaling() and ub_bulk_dgrad:
                    # reduce duplicated transpose in `_fix_gathered_fp8_transpose`
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                else:
                    input_quantizer.set_usage(
                        rowwise=True,
                        columnwise=backward_needs_input,
                    )
                if not isinstance(inputmat, QuantizedTensor):
                    inputmat = input_quantizer(inputmat)
                    own_quantized_input = True
                elif backward_needs_input:
                    inputmat.update_usage(rowwise_usage=True, columnwise_usage=True)
                inputmat_total = inputmat
        else:
            inputmat = cast_if_needed(inp, activation_dtype)
            if with_input_all_gather_nccl:
                inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
            else:
                inputmat_total = inputmat
        nvtx_range_pop(f"{nvtx_label}.input_cast_comm")

        # Cast weight to expected dtype
        weightmat = weight
        if not fp8:
            weightmat = cast_if_needed(weightmat, activation_dtype)
        else:
            if not isinstance(weight, QuantizedTensor):
                # Configure quantizer
                if weight_quantizer is not None:
                    columnwise_usage = is_grad_enabled and inp.requires_grad
                    if not columnwise_usage:
                        columnwise_usage = (
                            is_fp8_activation_recompute_enabled() and not in_fp8_activation_recompute_phase()
                        )
                    weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

                # FP8 cast to workspace buffer
                update_workspace = is_first_microbatch is None or is_first_microbatch
                weightmat = module.get_weight_workspace(
                    tensor=weight,
                    quantizer=weight_quantizer,
                    cache_name=(None if is_first_microbatch is None else "weight"),
                    update_workspace=update_workspace,
                    skip_update_flag=skip_fp8_weight_update,
                    fsdp_group=fsdp_group,
                    create_transpose_cache=keep_fp8_weight_transpose_cache,
                )

        # Cast bias to expected dtype
        bias_dtype = activation_dtype
        if fp8 and activation_dtype == torch.float32:
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        # Configure output quantizer
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # Calibrate quantizers if needed
        if not fp8 and fp8_calibration:
            if input_quantizer is not None:
                input_quantizer.calibrate(inputmat_total)
            if weight_quantizer is not None:
                weight_quantizer.calibrate(weight)

        ub_obj = None
        ub_type = None
        rs_out = None
        out_dtype = activation_dtype
        if ub_overlap_rs_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.RS
            out_shape = [reduce(multiply_op, inp_shape[:-1]) // tp_world_size, out_features]
            rs_out = torch.empty(out_shape, dtype=activation_dtype, device=inputmat_total.device)

        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.AG
            if fp8:
                assert ub_obj.is_fp8_ubuf(), "AG overlap with FP8 GEMM inputs requires FP8 buffer."
            ub_obj.copy_into_buffer(inputmat_total, input_quantizer, local_chunk=True)
            inputmat_total = ub_obj.get_buffer(input_quantizer)

        nvtx_range_push(f"{nvtx_label}.gemm")
        fprop_gemm_use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                fprop_gemm_use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

        out, *_, rs_out = general_gemm(
            weightmat,
            inputmat_total,
            get_workspace(),
            quantization_params=output_quantizer,
            out_dtype=out_dtype,
            bias=bias,
            use_split_accumulator=fprop_gemm_use_split_accumulator,
            ub=ub_obj,
            ub_type=ub_type,
            extra_output=rs_out,
        )
        nvtx_range_pop(f"{nvtx_label}.gemm")

        if is_grad_enabled:
            saved_inputmat = None

            ctx.backward_input_needs_gather = (
                weight.requires_grad and parallel_mode == "column" and sequence_parallel
            )

            if backward_needs_input:
                if own_quantized_input and isinstance(inputmat, QuantizedTensor):
                    # For sequence parallel in vanilla FP8, rowwise data is
                    # to gather the input. For MXFP8, columnwise only data
                    # can be allgathered.
                    if isinstance(inputmat, MXFP8TensorBase) or not ctx.backward_input_needs_gather:
                        inputmat.update_usage(rowwise_usage=False)
                saved_inputmat = inputmat

            # Weight with column-wise usage is needed for dgrad GEMM while keeping fp8 weight transpose cache.
            if inp.requires_grad and keep_fp8_weight_transpose_cache:
                if isinstance(weightmat, QuantizedTensor):
                    weightmat.update_usage(columnwise_usage=True)

            if cpu_offloading:
                set_offloading_param(weight, "weight_offloading", True)
                set_offloading_param(weightmat, "weight_offloading", True)
                if saved_inputmat is not None:
                    set_offloading_param(saved_inputmat, "activation_offloading", True)

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
            nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                saved_inputmat,
                weightmat if fp8 and not isinstance(weight, QuantizedTensor) else None,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_scatter")

            if cpu_offloading:
                ctx.grad_added_to_main_grad = hasattr(weight, "grad_added_to_main_grad")

                if ctx.grad_added_to_main_grad:
                    # If you are passing torch.nn.Parameter through the Torch hooks, you will
                    # get back torch.Tensor. Torch rips off the Parameter wrapper.
                    # You need to preserve the weight object to have all the attributes user
                    # sets for the weights. Because of this, it is not recommended to offload
                    # weights if weights are externally touched outside this module
                    ctx.weight_object = weight

            # TODO(ksivamani): Check memory usage
            tensors_to_save, tensor_objects = prepare_for_saving(
                saved_inputmat,
                weightmat,
                weight,
                bias,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.activation_dtype = activation_dtype
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.fp8 = fp8
            ctx.input_quantizer = input_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            if fuse_wgrad_accumulation and weight.requires_grad:
                ctx.main_grad = weight.main_grad

            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_overlap_ag = ub_overlap_ag_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_name = ub_name
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False
            ctx.owns_input = saved_inputmat is not inp
            ctx.keep_fp8_weight_transpose_cache = keep_fp8_weight_transpose_cache
            if ctx.fp8 and requires_grad(inp, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module

        # Row Parallel Linear
        if ub_overlap_rs_fprop:
            out = rs_out
        elif parallel_mode == "row":
            nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
            if sequence_parallel:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
            elif tensor_parallel:
                out, _ = allreduce(out, tp_group)
            nvtx_range_pop(f"{nvtx_label}.row_parallel_comm")

        out = out.view(-1, *inp_shape[1:-1], out_features)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring

        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.backward"
        if ctx.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ctx.ub_name}"

        with torch.cuda.nvtx.range("_Linear_backward"):
            if (
                ctx.fp8
                and any(
                    [
                        ctx.ub_overlap_ag,
                        ctx.ub_overlap_rs_dgrad,
                        ctx.ub_bulk_dgrad,
                        ctx.ub_bulk_wgrad,
                    ]
                )
                and (ctx.fp8_recipe is not None)
            ):
                if not ctx.fp8_recipe.float8_per_tensor_scaling():
                    raise NotImplementedError(
                        "Comm+GEMM overlap is only supported with FP8 delayed scaling or per-tensor"
                        " current scaling"
                    )

            saved_tensors = ctx.saved_tensors
            inputmat, weight_fp8, weight, bias = (  # pylint: disable=unbalanced-tuple-unpacking
                restore_from_saved(ctx.tensor_objects, saved_tensors)
            )
            # Delete the references to tensor objects once they've been consumed
            # by the `restore_from_saved` method to construct back the actual tensors.
            ctx.tensor_objects = None

            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            main_grad = (
                ctx.main_grad
                if weight is not None and ctx.fuse_wgrad_accumulation and ctx.requires_wgrad
                else None
            )

            if ctx.cpu_offloading:
                if ctx.grad_added_to_main_grad:
                    weight = ctx.weight_object
                if ctx.requires_wgrad and ctx.fuse_wgrad_accumulation:
                    weight.main_grad = main_grad

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            nvtx_range_push(f"{nvtx_label}.fsdp_gather")
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                inputmat,
                weight_fp8,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_gather")

            ctx.ub_obj_gradout = None
            ub_obj_dgrad = None
            ub_obj_wgrad = None
            ub_type_dgrad = None
            ub_type_wgrad = None
            dgrad_shape = [reduce(multiply_op, ctx.inp_shape[:-1]), ctx.inp_shape[-1]]
            rs_out = None
            dgrad_bulk = None
            if ctx.ub_overlap_ag:
                # Overlap grad_output all-gather with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.AG

            elif ctx.ub_overlap_rs_dgrad:
                # Overlap dgrad reduce-scatter with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.RS
                rs_out = torch.empty(dgrad_shape, dtype=ctx.activation_dtype, device=grad_output.device)

            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap inputmat all-gather with dgrad compute
                    # NOTE: Copying into communication buffer will always prefer rowwise data,
                    #       and will copy columnwise data if rowwise does not exist. In that case,
                    #       the all-gather will apply to the leading dimension of the transpose,
                    #       which then needs to be interleaved correctly before WGRAD.
                    ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                    ub_obj_dgrad = ctx.ub_obj_gradout
                    ub_type_dgrad = tex.CommOverlapType.AG
                    ub_obj_dgrad.copy_into_buffer(inputmat, ctx.input_quantizer, local_chunk=True)

                if ctx.ub_bulk_wgrad:
                    # Overlap dgrad reduce-scatter with wgrad compute
                    ub_obj_wgrad = get_ub(ctx.ub_name + "_wgrad")
                    ub_type_wgrad = tex.CommOverlapType.RS
                    ub_obj_wgrad.set_buffer_params(ctx.grad_input_quantizer)
                    dgrad_bulk = ub_obj_wgrad.get_buffer(ctx.grad_input_quantizer)

            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            if ctx.grad_output_quantizer is not None:
                # Reduce duplicated transpose, which is performed in grad_output.update_usage
                if ctx.ub_overlap_ag and ctx.fp8_recipe.float8_per_tensor_scaling():
                    ctx.grad_output_quantizer.set_usage(rowwise=True, columnwise=False)
                else:
                    ctx.grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
            nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
            (
                grad_output,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx,
                grad_output,
                ctx.parallel_mode == "row",
                ctx.grad_output_quantizer,
            )
            nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")

            # Prepare input tensor
            # Note: Perform tensor-parallel communication if needed
            inputmat_total = None
            inputmat_total_work = None
            if ctx.backward_input_needs_gather and not ctx.ub_bulk_dgrad:
                quantizer = None
                if ctx.fp8:
                    quantizer = ctx.input_quantizer
                    quantizer.set_usage(rowwise=True, columnwise=True)
                nvtx_range_push(f"{nvtx_label}.column_parallel_comm_input")
                inputmat_total, inputmat_total_work = gather_along_first_dim(
                    inputmat,
                    ctx.tp_group,
                    async_op=True,
                    quantizer=quantizer,
                )
                nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_input")
            else:
                inputmat_total = inputmat

            # Check whether to output wgrad GEMM directly into main grad
            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            # Compute grad input tensor
            dgrad = None
            dgrad_work = None
            if ctx.requires_dgrad:

                # Update quantizer
                if ctx.grad_input_quantizer is not None:
                    ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

                if ctx.fp8 and not ctx.keep_fp8_weight_transpose_cache:
                    create_fp8_weight_transpose_cache(weight_fp8)

                # dgrad GEMM
                nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
                dgrad_gemm_use_split_accumulator = _2X_ACC_DGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_dgrad"):
                        dgrad_gemm_use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator

                dgrad, *_, rs_out = general_gemm(
                    weight_fp8,
                    grad_output,
                    get_workspace(),
                    layout="NN",
                    grad=True,
                    quantization_params=ctx.grad_input_quantizer,
                    out=dgrad_bulk,
                    out_dtype=ctx.activation_dtype,
                    use_split_accumulator=dgrad_gemm_use_split_accumulator,
                    ub=ub_obj_dgrad,
                    ub_type=ub_type_dgrad,
                    extra_output=rs_out,
                    bulk_overlap=ctx.ub_bulk_dgrad,
                )
                nvtx_range_pop(f"{nvtx_label}.dgrad_gemm")

                if ctx.fp8 and not ctx.keep_fp8_weight_transpose_cache:
                    clear_fp8_weight_transpose_cache(weight_fp8)

                # Launch tensor-parallel communication
                if ctx.ub_overlap_rs_dgrad:
                    dgrad = rs_out
                elif ctx.parallel_mode == "column" and not ctx.ub_bulk_wgrad:
                    nvtx_range_push(f"{nvtx_label}.column_parallel_comm_dgrad")
                    if ctx.sequence_parallel:
                        dgrad, dgrad_work = reduce_scatter_along_first_dim(
                            dgrad,
                            ctx.tp_group,
                            async_op=True,
                        )
                    else:
                        dgrad, dgrad_work = allreduce(dgrad, ctx.tp_group, async_op=True)
                    nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_dgrad")

            # Compute grad weight tensor
            wgrad = None
            if ctx.requires_wgrad:
                if ctx.ub_bulk_dgrad:
                    inputmat_total = ub_obj_dgrad.get_buffer(ctx.input_quantizer)
                    if ctx.fp8:
                        if inputmat._data is None:
                            # All-gather executed on columnwise data and result is in rowwise data,
                            # so we need to fix the interleaving before WGRAD.
                            inputmat_total = _fix_gathered_fp8_transpose(inputmat_total, ctx.tp_size)
                        elif not non_tn_fp8_gemm_supported():
                            # FP8 GEMM on Hopper only supports TN layout so the gathered input must
                            # have a valid transpose.
                            inputmat_total._create_transpose()

                else:
                    if inputmat_total_work is not None:
                        # Synchronize tensor-parallel communication
                        inputmat_total_work.wait()
                        inputmat_total_work = None

                if isinstance(grad_output, QuantizedTensor):
                    # This is a no-op if platform supports non-TN FP8 GEMM or the transpose
                    # already exists.
                    grad_output.update_usage(rowwise_usage=True, columnwise_usage=True)

                if ctx.ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                    rs_out = torch.empty(dgrad_shape, dtype=ctx.activation_dtype, device=grad_output.device)

                # wgrad GEMM
                # Note: Fuse with bgrad computation if needed
                def pre_process(_grad_output_, _input_, async_op=True):
                    return _grad_output_, _input_, None

                def process_wgrad(main_grad, grad_output, inputmat_total, handle=None):
                    nvtx_range_push(f"{nvtx_label}.wgrad_gemm")

                    wgrad_gemm_use_split_accumulator = _2X_ACC_WGRAD
                    if ctx.fp8:
                        recipe = ctx.fp8_recipe
                        if hasattr(recipe, "fp8_gemm_wgrad"):
                            wgrad_gemm_use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator
                    # print(f"debug acc {accumulate_wgrad_into_param_main_grad}")
                    wgrad, grad_bias_, _, _ = general_gemm(
                        inputmat_total,
                        grad_output,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        out_dtype=(main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype),
                        bias=None,
                        out=main_grad if ctx.fuse_wgrad_accumulation else None,
                        use_split_accumulator=wgrad_gemm_use_split_accumulator,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        ub=ub_obj_wgrad,
                        ub_type=ub_type_wgrad,
                        extra_output=None,
                        bulk_overlap=False,
                    )

                    nvtx_range_pop(f"{nvtx_label}.wgrad_gemm")

                    # Deallocate input tensor
                    if ctx.owns_input:
                        clear_tensor_data(inputmat_total)
                    # Handle custom DDP from mcore.
                    if (
                        ctx.fuse_wgrad_accumulation
                        and weight is not None
                        and hasattr(weight, "grad_added_to_main_grad")
                    ):
                        weight.grad_added_to_main_grad = True
                        if getattr(weight, "zero_out_wgrad", False):
                            wgrad = torch.zeros(
                                weight.main_grad.shape,
                                dtype=weight.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False,
                            )
                        else:
                            wgrad = torch.empty(
                                weight.main_grad.shape,
                                dtype=weight.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False,
                            )
                    elif ctx.fuse_wgrad_accumulation:
                        pass

                insert_wgrad_func_into_cache(
                    main_grad,
                    functools.partial(pre_process, grad_output, inputmat_total),
                    functools.partial(process_wgrad, main_grad),
                )

            # Synchronize tensor parallel communication
            if inputmat_total_work is not None:
                assert False
                inputmat_total_work.wait()
                inputmat_total_work = None
            if dgrad_work is not None:
                assert False
                dgrad_work.wait()
                dgrad_work = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")

        # Scatter fp8 weight buffers
        if ctx.fp8 and not isinstance(weight, QuantizedTensor):
            _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)

        wgrad = None
        return (
            wgrad,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # input_quantizer
            None,  # weight_quantizer
            None,  # output_quantizer
            None,  # grad_output_quantizer
            None,  # grad_input_quantizer
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # is_grad_enabled
            None,  # ub_overlap_rs_fprop
            None,  # ub_overlap_ag_dgrad
            None,  # ub_overlap_ag_fprop
            None,  # ub_overlap_rs_dgrad
            None,  # ub_bulk_dgrad
            None,  # ub_bulk_wgrad
            None,  # ub_name
            None,  # fp8_output
            None,  # fsdp_group
            None,  # module
            None,  # skip_fp8_weight_update
            None,  # keep_fp8_weight_transpose_cache
        )
