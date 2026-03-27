import torch
import functools
import torch.distributed as dist
from typing import Callable, Optional, Union, Tuple, List

from megatron.core.utils import experimental_api
import transformer_engine as te

import primus_turbo.pytorch as turbo

from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboLowPrecisionGlobalStateManager
from primus.backends.megatron.core.extensions.experimental.utils import divide, init_method_constant

from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled, mark_not_offload, start_offload


class _GroupedLinear(torch.autograd.Function):
    """GroupedLinear semi-top level module
    Calls custom cuda extensions.
    """

    # pylint: disable=keyword-arg-before-vararg
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        non_tensor_args: Tuple,
        *weight_and_bias: Tuple,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Reduce number of arguments to autograd function in order
        # to reduce CPU overhead due to pytorch arg checking.
        (
            num_tokens_per_expert,
            num_tokens_prefix_sum,
            use_bias,
            is_first_microbatch,
            fp8,
            wgrad_store,
            fuse_wgrad_accumulation,
            cpu_offloading,
            sequence_parallel,
            activation_dtype,
            is_grad_enabled,
            save_original_input,
            num_sms,
        ) = non_tensor_args

        weight, bias = weight_and_bias

        device = inp.device
        weight_requires_grad = weight.requires_grad

        # Initialize input tensors
        in_features = weight.size(-1)
        if inp.size(-1) != in_features:
            raise ValueError(
                f"Input tensor (shape={tuple(inp.size())}) is not compatible with "
                f"weight tensor (shape={tuple(weight.size())})"
            )
        inp_view = inp.reshape(-1, in_features)

        if cpu_offloading:
            start_offload(inp)

        bias_dtype = activation_dtype
        if fp8 and activation_dtype == torch.float32:
            bias_dtype = torch.bfloat16  # FP8 GEMM only supports BF16/FP16 bias

        # Perform groupedgemm
        if num_tokens_per_expert.numel() == 1:
            assert weight.size(
                0) == 1, f"Expected first dimension to be 1, got {weight.size(0)}"
            weight_2d = weight.squeeze(0)
            out = turbo.kernels.gemm.gemm_impl(
                inp_view, False, weight_2d, True, inp_view.dtype, False, default_backend=turbo.core.BackendType.HIPBLASLT.value
            )
        else:
            out = turbo.kernels.grouped_gemm.grouped_gemm_impl(
                inp_view,
                weight,
                num_tokens_per_expert,
                num_tokens_prefix_sum,
                trans_a=False,
                trans_b=True,
                num_cu=num_sms,
                default_backend=turbo.core.BackendType.CK.value,
                maybe_pre_sync=True,
            )

        if cpu_offloading:
            mark_not_offload(weight)

        if is_grad_enabled:

            if cpu_offloading:
                ctx.grad_added_to_main_grad = hasattr(
                    weight, "grad_added_to_main_grad")

                if ctx.grad_added_to_main_grad:
                    # If you are passing torch.nn.Parameter through the Torch hooks, you will
                    # get back torch.Tensor. Torch rips off the Parameter wrapper.
                    # You need to preserve the weight object to have all the attributes user
                    # sets for the weights. Because of this, it is not recommended to offload
                    # weights if weights are externally touched outside this module
                    ctx.weight_objects = [weight]

            ctx.save_for_backward(
                inp, weight, bias, num_tokens_per_expert, num_tokens_prefix_sum)

            ctx.weights_requires_grad = weight.requires_grad
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                # This check is needed to ensure that main_grad is not created
                # during the forward pass when using MCore FSDP as it creates
                # the main_grad buffer lazily before backprop
                if hasattr(weight, "__fsdp_param__"):
                    # MCore FSDP creates main_grad lazily before backward
                    ctx.main_grad_funcs = [weight.get_main_grad]
                else:
                    ctx.main_grad_funcs = [
                        lambda: weight.main_grad
                    ]
            else:
                ctx.main_grad_funcs = [lambda: None]
            ctx.device = device
            ctx.num_sms = num_sms
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
            if ctx.fp8 and te.pytorch.utils.requires_grad(inp, weight, bias):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors
                    or PrimusTurboLowPrecisionGlobalStateManager.is_first_fp8_module()
                )
            ctx.wgrad_store = wgrad_store
            ctx.save_original_input = save_original_input

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        with te.pytorch.utils.get_nvtx_range_context("_GroupedLinear_backward"):
            inp, weight, bias, num_tokens_per_expert, num_tokens_prefix_sum = ctx.saved_tensors
            main_grads = [main_grad_func()
                          for main_grad_func in ctx.main_grad_funcs]
            if ctx.cpu_offloading:
                if ctx.grad_added_to_main_grad:
                    weight = ctx.weight_objects[0]
                    ctx.weight_objects = None

            if ctx.fuse_wgrad_accumulation:
                weight.main_grad = main_grads[0]

            # Preprocess grad output
            grad_output_view = grad_output.contiguous(
            ).view(-1, grad_output.shape[-1])

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
                # Make sure weights are available in column-wise format
                # for dgrad computation.
                dgrad = turbo.kernels.grouped_gemm.grouped_gemm_impl(
                    grad_output_view,
                    weight,
                    num_tokens_per_expert,
                    num_tokens_prefix_sum,
                    trans_a=False,
                    trans_b=False,
                    num_cu=ctx.num_sms,
                    default_backend=turbo.core.BackendType.CK.value,
                )

            # TODO(zhenhuang12): support bias and accumulate
            # accumulate_wgrad_into_param_main_grad = True
            assert not accumulate_wgrad_into_param_main_grad, "Accumulate wgrad into param main grad is not supported"

            if ctx.weights_requires_grad:
                if ctx.fuse_wgrad_accumulation:
                    wgrad = main_grads[0]
                else:
                    wgrad = torch.empty(
                        weight.size(), dtype=ctx.activation_dtype, device=ctx.device)

                grouped_gemm_wgrad = functools.partial(
                    turbo.kernels.grouped_gemm.grouped_gemm_variable_k_impl,
                    trans_a=True,
                    trans_b=False,
                    trans_c=False,
                    num_cu=ctx.num_sms,
                    default_backend=turbo.core.BackendType.CK.value,
                )
                # WGRAD
                if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
                    ctx.wgrad_store.put(
                        [inp, grad_output_view, num_tokens_per_expert, num_tokens_prefix_sum], grouped_gemm_wgrad)
                else:
                    _ = grouped_gemm_wgrad(
                        inp,
                        grad_output_view,
                        num_tokens_per_expert,
                        num_tokens_prefix_sum)

                    # TODO(zhenhuang12): support bias
                    # for i in range(ctx.num_gemms):
                    #     if grad_biases[i] is None:
                    #         grad_biases[i] = grad_biases_[i]
                    # del grad_biases_

                    # Deallocate input tensor

                def handle_custom_ddp_from_mcore(weight, wgrad):
                    if ctx.weights_requires_grad:
                        # Handle custom DDP from mcore.
                        if ctx.fuse_wgrad_accumulation and hasattr(
                            weight, "grad_added_to_main_grad"
                        ):
                            weight.grad_added_to_main_grad = True
                            if getattr(weight, "zero_out_wgrad", False):
                                wgrad = te.pytorch.base.get_dummy_wgrad(
                                    list(weight.main_grad.shape),
                                    weight.dtype,
                                    zero=True,
                                )
                            else:
                                wgrad = te.pytorch.base.get_dummy_wgrad(
                                    list(weight.main_grad.shape),
                                    weight.dtype,
                                )
                        elif ctx.fuse_wgrad_accumulation:
                            wgrad = None
                    else:
                        wgrad = None
                    return wgrad

                wgrad = handle_custom_ddp_from_mcore(weight, wgrad)
            else:
                wgrad = None

            # TODO(zhenhuang12): support bias
            # if not ctx.use_bias or (
            #     ctx.wgrad_store is not None
            #     and ctx.wgrad_store.delay_wgrad_compute()
            #     and not ctx.fp8
            # ):
            #     grad_biases = [None] * ctx.num_gemms

        if ctx.reduce_and_update_bwd_fp8_tensors:
            PrimusTurboLowPrecisionGlobalStateManager.reduce_and_update_fp8_tensors(
                forward=False)
        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            None,
            wgrad,
            None,  # bias_grad
        )


@experimental_api
class GroupedLinear(torch.nn.Module):
    """Legacy grouped-linear wrapper with TE-style behavior.

    Supported features:
    - legacy `weight1`/`weight2` big-tensor parameter layout
    - grouped GEMM dispatch via turbo kernels
    - direct FP8 grouped GEMM execution without primary FP8 weights
    - `delay_wgrad_compute` with `backward_dw()`
    - `fuse_wgrad_accumulation` into `main_grad`
    - CPU offload hooks through TE offload helpers
    - explicit exclusion of TP/SP-internal handling
    """

    def __init__(
        self,
        num_lcal_experts: int,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        name: Optional[str] = None,
        num_sms: int = None,
    ) -> None:

        #  check common arguments
        if device != "cuda":
            raise ValueError(
                "PrimusTurboGroupedLinear only supports device=cuda")

        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.num_sms = num_sms

        if self.use_bias:
            raise NotImplementedError(
                "PrimusTurboGroupedLinear does not support bias")

        # ===================================
        # == features 1:TP/SP/TP overlap ====
        # ===================================
        # TODO(zhenhuang12): not support
        if tp_group is None:
            self.tp_size = tp_size

        else:
            self.tp_size = tp_group.size() if tp_group.is_initialized() else 1

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        if self.tp_size > 1 or self.sequence_parallel \
                or ub_overlap_rs or ub_overlap_ag:
            raise ValueError(
                "PrimusTurboGroupedLinear does not support TP/SP/TP overlap")

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)
        elif self.parallel_mode is not None:
            raise ValueError(
                f"parallel_mode {parallel_mode!r} not supported."
            )

        # ===================================
        # == features 2: primary weights
        # ===================================
        # TODO(zhenhuang12): not support
        self.primary_weights_in_fp8 = PrimusTurboLowPrecisionGlobalStateManager.with_fp8_parameters()
        if self.primary_weights_in_fp8:
            raise ValueError(
                "PrimusTurboGroupedLinear does not support primary weights in FP8")

        # ===================================
        # == features 3: fuse_wgrad_accumulation
        # ===================================
        # TODO(zhenhuang12): not support
        if fuse_wgrad_accumulation:
            raise ValueError(
                "PrimusTurboGroupedLinear does not support fuse_wgrad_accumulation")
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        # N * K
        self.register_parameter(
            f"weight",
            torch.nn.Parameter(
                torch.empty(
                    self.out_features * self.num_local_experts,
                    self.in_features,
                    device=device,
                    dtype=self.params_dtype,
                ),
            ),
            init_fn=init_method,
            get_rng_state_tracker=get_rng_state_tracker,
        )
        if self.use_bias:
            self.register_parameter(
                f"bias",
                torch.nn.Parameter(
                    torch.empty(
                        self.out_features,
                        device=device,
                        dtype=self.params_dtype,
                    ),
                ),
                init_fn=init_method_constant(0.0),
            )
        else:
            bias = torch.Tensor().to(dtype=self.params_dtype, device=device)
            setattr(self, f"bias", bias)

        if self.primary_weights_in_fp8:
            raise ValueError(
                "PrimusTurboGroupedLinear does not support primary weights in FP8")

    @te.jit.no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        num_tokens_per_expert : List[int]
                 List of integers representing the split of the input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """

        is_grad_enabled = torch.is_grad_enabled()

        try:

            if is_grad_enabled:
                linear_fn = _GroupedLinear.apply
                autograd_ctx = []
            else:
                linear_fn = _GroupedLinear.forward
                autograd_ctx = [None]

            num_tokens_prefix_sum = turbo.ops.grouped_gemm_compute_offs(
                num_tokens_per_expert)

            non_tensor_args = (
                num_tokens_per_expert,  # need
                num_tokens_prefix_sum,
                self.apply_bias,  # need
                is_first_microbatch,  # ?
                self.wgrad_store,  # need
                self.fuse_wgrad_accumulation,  # need
                is_cpu_offload_enabled(),  # need
                self.sequence_parallel,  # need
                self.activation_dtype,  # need
                is_grad_enabled,  # need
            )
            out, _ = linear_fn(*autograd_ctx, inp, non_tensor_args,
                               self.weight, self.bias)

        finally:
            self.end_forward()

        if self.return_bias:
            return out, None
        return out
