from collections import deque
from typing import Callable, Deque, Optional, Sequence, Tuple, Union

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.extensions.transformer_engine import condition_init_method, _get_extra_te_kwargs
import torch
import torch.distributed as dist
import primus_turbo.pytorch as turbo
import transformer_engine as te
from transformer_engine.pytorch.module._common import WeightGradStore as TEWeightGradStore
from megatron.core.utils import experimental_api
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    get_te_version,
    get_tensor_model_parallel_group_if_none,
    is_te_min_version,
    is_torch_min_version,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_impl
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_impl,
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.ops import grouped_gemm, grouped_gemm_fp8

from primus.backends.megatron.core.extensions.experimental.utils import (
    divide,
    init_method_constant,
)
from primus.backends.megatron.core.extensions.primus_turbo import (
    PrimusTurboLowPrecisionGlobalStateManager,
)


def _ensure_group_lens(
    num_tokens_per_expert: Union[torch.Tensor, Sequence[int]],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(num_tokens_per_expert, torch.Tensor):
        return num_tokens_per_expert.to(device=device, dtype=torch.long)
    return torch.tensor(list(num_tokens_per_expert), device=device, dtype=torch.long)


def _validate_runtime_shapes(
    inp_view: torch.Tensor,
    weight: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_tokens_prefix_sum: torch.Tensor,
) -> None:
    if weight.ndim != 3:
        raise ValueError(
            f"Expected stacked weight with shape [G, N, K], got {tuple(weight.shape)}."
        )
    if num_tokens_per_expert.ndim != 1:
        raise ValueError(
            "num_tokens_per_expert must be a 1D tensor of expert token counts."
        )
    if num_tokens_prefix_sum.ndim != 1 or (
        num_tokens_prefix_sum.numel() != num_tokens_per_expert.numel() + 1
    ):
        raise ValueError(
            "num_tokens_prefix_sum must be a 1D exclusive prefix sum with length G + 1."
        )
    if weight.size(0) != num_tokens_per_expert.numel():
        raise ValueError(
            f"Expected weight.size(0) == number of experts, got {weight.size(0)} "
            f"and {num_tokens_per_expert.numel()}."
        )
    if inp_view.size(-1) != weight.size(-1):
        raise ValueError(
            f"Input tensor (shape={tuple(inp_view.shape)}) is not compatible with "
            f"weight tensor (shape={tuple(weight.shape)})."
        )
    total_tokens = int(num_tokens_per_expert.sum().item())
    if inp_view.size(0) != total_tokens:
        raise ValueError(
            f"Input contains {inp_view.size(0)} rows, but num_tokens_per_expert sums to "
            f"{total_tokens}."
        )


def _apply_grouped_bias(
    out: torch.Tensor,
    bias: torch.Tensor,
    num_tokens_prefix_sum: torch.Tensor,
) -> torch.Tensor:
    if bias.numel() == 0:
        return out
    if bias.ndim != 2:
        raise ValueError(
            f"Expected bias shape [G, N], got {tuple(bias.shape)}.")

    if bias.size(0) == 1:
        return out + bias[0].to(dtype=out.dtype)

    out = out.clone()
    for expert_idx in range(bias.size(0)):
        start = int(num_tokens_prefix_sum[expert_idx].item())
        end = int(num_tokens_prefix_sum[expert_idx + 1].item())
        if start == end:
            continue
        out[start:end] = out[start:end] + bias[expert_idx].to(dtype=out.dtype)
    return out


def _grouped_linear_forward_kernel(
    inp_view: torch.Tensor,
    weight: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_tokens_prefix_sum: torch.Tensor,
    num_sms: Optional[int],
) -> torch.Tensor:
    if num_tokens_per_expert.numel() == 1:
        if weight.size(0) != 1:
            raise ValueError(
                f"Expected single-expert weight shape [1, N, K], got {tuple(weight.shape)}."
            )
        return gemm_impl(
            inp_view,
            False,
            weight.squeeze(0),
            True,
            inp_view.dtype,
            False,
            default_backend=BackendType.HIPBLASLT.value,
        )

    return grouped_gemm_impl(
        inp_view,
        weight,
        num_tokens_per_expert,
        num_tokens_prefix_sum,
        trans_a=False,
        trans_b=True,
        num_cu=num_sms,
        default_backend=BackendType.CK.value,
        maybe_pre_sync=True,
    )


def _grouped_linear_dgrad_kernel(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_tokens_prefix_sum: torch.Tensor,
    num_sms: Optional[int],
) -> torch.Tensor:
    if num_tokens_per_expert.numel() == 1:
        return gemm_impl(
            grad_output,
            False,
            weight.squeeze(0),
            False,
            grad_output.dtype,
            False,
            default_backend=BackendType.HIPBLASLT.value,
        )

    return grouped_gemm_impl(
        grad_output,
        weight,
        num_tokens_per_expert,
        num_tokens_prefix_sum,
        trans_a=False,
        trans_b=False,
        num_cu=num_sms,
        default_backend=BackendType.CK.value,
    )


def _grouped_linear_wgrad_kernel(
    inp_view: torch.Tensor,
    grad_output: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_tokens_prefix_sum: torch.Tensor,
    num_sms: Optional[int],
) -> torch.Tensor:
    if num_tokens_per_expert.numel() == 1:
        return gemm_impl(
            grad_output,
            True,
            inp_view,
            False,
            grad_output.dtype,
            False,
            default_backend=BackendType.HIPBLASLT.value,
        ).unsqueeze(0)

    return grouped_gemm_variable_k_impl(
        inp_view,
        grad_output,
        num_tokens_per_expert,
        num_tokens_prefix_sum,
        trans_a=True,
        trans_b=False,
        trans_c=True,
        num_cu=num_sms,
        default_backend=BackendType.CK.value,
    )


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
class GroupedLinear(te.pytorch.GroupedLinear):
    """Experimental Primus-Turbo backed GroupedLinear.

    Unlike TE's default per-GEMM `weight{i}` layout, this module stores a single
    stacked weight tensor with shape `[num_gemms, out_features, in_features]` so it
    maps directly to Primus-Turbo's grouped GEMM operator.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        num_sms: Optional[int] = None,
    ):

        self.config = config

        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache

        extra_kwargs = _get_extra_te_kwargs(config)

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError(
                    "Only TE with version >=2.3.0 supports delay_wgrad_compute now."
                )

        extra_kwargs["ub_name"] = tp_comm_buffer_name

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if is_expert:
            extra_kwargs["rng_tracker_name"] = get_expert_parallel_rng_tracker_name()

        # The comms between TP and EP group is explicitly handled by MoE token dispatcher.
        # So we disable comms by making TE agnostic of model parallel.
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self._pg_collection = pg_collection
        assert is_expert, "TEGroupedLinear only supports expert parallelism"
        tp_group = pg_collection.expt_tp
        self._tp_group = tp_group
        tp_size = get_pg_size(tp_group)
        tp_group_for_te = tp_group

        self.explicit_expert_comm = is_expert and (
            tp_size > 1 or self.expert_parallel)

        if self.explicit_expert_comm:
            if parallel_mode == "column":
                output_size = divide(output_size, tp_size)
            elif parallel_mode == "row":
                input_size = divide(input_size, tp_size)
            parallel_mode = None
            tp_size = 1
            tp_group_for_te = None
            
        self.single_grouped_parameter = True

        super().__init__(
            num_gemms=num_gemms,
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=tp_group_for_te if torch.distributed.is_initialized() else None,
            tp_size=tp_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=condition_init_method(config, init_method),
            bias=bias,
            return_bias=skip_bias_add and bias,
            parallel_mode=parallel_mode,
            **extra_kwargs,
        )

        # check unsupported features
        self._check_unsupported_features()

        self.te_quant_params = None
        for param in self.parameters():
            setattr(param, "allreduce", not (
                is_expert and self.expert_parallel))

        def merge_extra_states(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            """
            Merge multiple "_extra_state" into one.
            """
            self.init_fp8_metadata(num_gemms=self.num_gemms)
            # When resume training, loading ckpt is out of fp8_autocast context.
            # So we need to manually detect from the state_dict.
            fp8_checkpoint = any("_extra_state" in str(key)
                                 for key in state_dict.keys())

            if not fp8_checkpoint:
                return

            try:
                state_list = [
                    state_dict.pop(f"{prefix}_extra_state{i}") for i in range(1, self.num_gemms)
                ]
            except KeyError:
                # "_extra_state{i}" only exists for dist-ckpt. Return for torch native ckpt.
                return

            # Early return conditions:
            # 1. Empty state_dict
            # 2. Empty state_list
            # 3. _extra_state is None
            # 4. _extra_state does not contain any information
            if (
                not state_dict
                or not state_list
                or state_dict.get(f"{prefix}_extra_state") is None
                or self._decode_extra_state(state_dict[f"{prefix}_extra_state"]) is None
            ):
                return

            state_list = [state_dict.pop(f"{prefix}_extra_state")] + state_list
            state_list = [self._decode_extra_state(
                state) for state in state_list]
            extra_fp8_variables = state_list[0]["extra_fp8_variables"]
            extra_fp8_variables["num_gemms"] = self.num_gemms
            extra_state = {"extra_fp8_variables": extra_fp8_variables}
            # TE 2.0 adds recipe in extra_state
            if is_te_min_version("2.0.0"):
                self.fp8_meta["recipe"] = state_list[0]["recipe"]
                extra_state["recipe"] = self.fp8_meta["recipe"]
            # Only delayed scaling has global fp8 meta tensors. We're not using
            # self.fp8_meta["recipe"].delayed() because it's available in TE 2.0 and later.
            if isinstance(self.fp8_meta["recipe"], te.common.recipe.DelayedScaling):
                extra_state.update(
                    {
                        "scale_fwd": torch.cat(
                            [state["scale_fwd"].view(-1, 1) for state in state_list], dim=1
                        ).view(-1),
                        "amax_history_fwd": torch.cat(
                            [state["amax_history_fwd"].view(-1, 1)
                             for state in state_list],
                            dim=1,
                        ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                        "scale_bwd": torch.cat(
                            [state["scale_bwd"].view(-1, 1) for state in state_list], dim=1
                        ).view(-1),
                        "amax_history_bwd": torch.cat(
                            [state["amax_history_bwd"].view(-1, 1)
                             for state in state_list],
                            dim=1,
                        ).view(self.fp8_meta["recipe"].amax_history_len, -1),
                    }
                )
                # TE 2.0 removes scale_inv_fwd and scale_inv_bwd
                if not is_te_min_version("2.0.0"):
                    extra_state.update(
                        {
                            "scale_inv_fwd": torch.cat(
                                [state["scale_inv_fwd"].view(-1, 1)
                                 for state in state_list],
                                dim=1,
                            ).view(-1),
                            "scale_inv_bwd": torch.cat(
                                [state["scale_inv_bwd"].view(-1, 1)
                                 for state in state_list],
                                dim=1,
                            ).view(-1),
                        }
                    )
            state_dict[f"{prefix}_extra_state"] = self._encode_extra_state(
                extra_state)

        self._register_load_state_dict_pre_hook(
            merge_extra_states, with_module=True)

    def _check_unsupported_features(self) -> None:
        if self.fuse_wgrad_accumulation:
            raise ValueError(
                "PrimusTurboGroupedLinear does not yet support fuse_wgrad_accumulation."
            )
        if self.ub_overlap_rs or self.ub_overlap_ag:
            raise ValueError(
                "PrimusTurboGroupedLinear does not support UserBuffer overlap."
            )
            
    def make_grouped_weights(self, defer_init=False) -> None:
        """
        Convert parameters into a GroupedTensor and re-register them as parameters.
        """

        if defer_init:
            return

        weight_quantizers = self._get_weight_quantizers()
        recipe = (
            weight_quantizers[0]._get_compatible_recipe()
            if weight_quantizers and weight_quantizers[0] is not None
            else None
        )
        if recipe is not None and (recipe.delayed() or recipe.float8_current_scaling()):
            self.set_tensor_parallel_attributes(defer_init=defer_init)
            return

        weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]

        # Create the weight storage.
        grouped_weights = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=self.num_gemms,
            shapes=[(self.out_features, self.in_features)] * self.num_gemms,
            quantizer=weight_quantizers[0],
            dtype=self.params_dtype,
            device=weights[0].device,
        )

        # Copy existing params into storage.
        with torch.no_grad():
            for i in range(self.num_gemms):
                if self.primary_weights_in_fp8:
                    grouped_weights.quantized_tensors[i].copy_from_storage(weights[i])
                else:
                    grouped_weights.quantized_tensors[i].copy_(weights[i])

        # Re-register as a single grouped weight parameter.
        # Re-register as a single grouped weight parameter.
        if not (
            isinstance(grouped_weights, torch.Tensor)
            and (weight_quantizers[0] is None or not weight_quantizers[0].internal)
        ):
            raise RuntimeError("Found internal quantizer with `single_grouped_parameter=True`.")
        self.register_parameter(
            "weight",
            torch.nn.Parameter(grouped_weights),
            init_fn=self.init_method,
            get_rng_state_tracker=self.get_rng_state_tracker,
            fp8_meta_index=self._offsets["weight"],
        )
        for i in range(self.num_gemms):
            self.register_parameter(f"weight{i}", None)

        self.set_tensor_parallel_attributes(defer_init=defer_init)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)
        # Grouped tensor weights is an opt-in feature.
        if self.single_grouped_parameter:
            self.make_grouped_weights(defer_init=defer_init)

    def _get_weight_tensor(self) -> torch.Tensor:
        """Get the weight tensors of the module."""
        grouped_weight = getattr(self, "weight", None)
        assert grouped_weight is not None, "Weight is not found"
        weight_tensors = grouped_weight.quantized_tensors
        if weight_tensors is None:
            # TODO(ksivaman): Remove this after GEMM integration.
            weight_tensors = grouped_weight.split_into_quantized_tensors()
        return weight_tensors[0]
    
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: Union[list, torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        # quant_context = _get_fp8_autocast_for_quant_params(
        #     self.te_quant_params, self.training)

        # with quant_context:
        #     out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)

        if isinstance(m_splits, list):
            m_splits = torch.tensor(
                m_splits, device=inp.device, dtype=torch.long)

        if len(m_splits) != self.num_gemms:
            raise ValueError(
                f"Number of splits ({len(m_splits)}) should match number of"
                f" GEMMs ({self.num_gemms})."
            )

        is_grad_enabled = torch.is_grad_enabled()

        try:
            weight_tensors = self._get_weight_tensors()
            # bias_tensors = [getattr(self, f"bias{i}")
            #                 for i in range(self.num_gemms)]

            if is_grad_enabled:
                linear_fn = _GroupedLinear.apply
                autograd_ctx = []
            else:
                linear_fn = _GroupedLinear.forward
                autograd_ctx = [None]

            m_offs = turbo.ops.grouped_gemm_compute_offs(m_splits)

            non_tensor_args = (
                m_splits,  # need
                m_offs,
                self.apply_bias,  # need
                _is_first_microbatch,  # ?
                self.wgrad_store,  # need
                self.fuse_wgrad_accumulation,  # need
                te.pytorch.cpu_offload.is_cpu_offload_enabled(),  # need
                self.sequence_parallel,  # need
                self.activation_dtype,  # need
                is_grad_enabled,  # need
            )
            out, _ = linear_fn(*autograd_ctx, inp, non_tensor_args,
                               self.weight, self.bias)

        finally:
            self.end_forward()

        self.is_first_microbatch = False

        return out,  [cast_if_needed(b, self.activation_dtype) for b in bias_tensors] if self.return_bias else None


class ColumnParallelGroupedLinear(GroupedLinear):
    """
    Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
    to column-parallel style.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            pg_collection=pg_collection,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """
        For each gemm, sharding along axis 0, bias sharded.
        Assume sharded_offsets[-1] is the expert parallel offset.
        """
        tp_axis_map = {}
        for gemm_idx in range(self.num_gemms):
            tp_axis_map.update(
                {f"{gemm_idx}.weight": 0, f"{gemm_idx}.bias": 0})
        return super()._sharded_state_dict_grouped(
            tp_axis_map, prefix, sharded_offsets, metadata
        )


class RowParallelGroupedLinear(GroupedLinear):
    """
    Wrapper for the Transformer-Engine's `GroupedLinear` layer but specialized
    to row-parallel style.
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            pg_collection=pg_collection,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """
        For each gemm, sharding along axis 1, bias not sharded.
        Assume sharded_offsets[-1] is the expert parallel offset.
        """
        tp_axis_map = {
            f"{gemm_idx}.weight": 1 for gemm_idx in range(self.num_gemms)}
        return super()._sharded_state_dict_grouped(
            tp_axis_map, prefix, sharded_offsets, metadata
        )
