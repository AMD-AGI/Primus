from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import (
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.transformer.moe.experts import TEGroupedMLP, TEGroupedMLPSubmodules
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.training.global_vars import get_args

from primus.core.utils.module_utils import warning_rank_0

_TURBO_FUSED_GROUPED_GEMM_FALLBACK_WARNED = False


def is_turbo_fused_grouped_mlp_supported() -> Tuple[bool, Optional[str]]:
    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboLowPrecisionGlobalStateManager,
    )

    if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
        quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        if not (quant_config.current_scaling() or quant_config.mxfp8_scaling()):
            return False, "turbo_fused_grouped_gemm only supports current scaling or mxfp8 scaling currently"
    elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
        quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        if not quant_config.mxfp4_scaling():
            return False, "turbo_fused_grouped_gemm only supports mxfp4 scaling currently"
    else:
        return False, "`turbo_fused_grouped_gemm` requires Turbo FP8 (current or mxfp8 scaling) or Turbo FP4."

    return True, None


class PrimusGroupedMLP(TEGroupedMLP):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: TEGroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        args = get_args()

        super().__init__(
            num_local_experts,
            config,
            submodules,
            pg_collection,
        )

        # NOTE: use_turbo_fused_act_with_probs is prioritized over use_te_activation_func and bias_activation_fusion
        self.use_turbo_fused_act_with_probs = args.use_turbo_fused_act_with_probs
        self.moe_router_padding_for_quantization = args.moe_router_padding_for_quantization
        # PrimusTurbo grouped GEMMs consume the original GPU tokens_per_expert
        # tensor for every supported quantization recipe. Keep TE's explicit
        # zero-padding fallback only when the no-padding path is not enabled.
        self.turbo_grouped_gemm_without_padding = getattr(args, "turbo_grouped_gemm_without_padding", False)
        self.turbo_fused_grouped_gemm = getattr(args, "turbo_fused_grouped_gemm", False)
        if self.turbo_fused_grouped_gemm:
            self._check_turbo_fused_grouped_gemm_contract()

    def _check_turbo_fused_grouped_gemm_contract(self) -> None:
        """Fail early on configs the fused grouped-MLP op does not implement."""
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboGroupedLinear,
        )

        assert isinstance(self.linear_fc1, PrimusTurboGroupedLinear) and isinstance(
            self.linear_fc2, PrimusTurboGroupedLinear
        ), "`turbo_fused_grouped_gemm` requires PrimusTurboGroupedLinear (enable `use_turbo_grouped_gemm`)."
        assert (
            not self.config.add_bias_linear
        ), "`turbo_fused_grouped_gemm` does not support `add_bias_linear`."
        assert (
            self.config.glu_linear_offset == 0.0
        ), "`turbo_fused_grouped_gemm` does not support a non-zero glu_linear_offset."
        assert (
            not self.config.activation_func_fp8_input_store
        ), "`turbo_fused_grouped_gemm` does not support activation_func_fp8_input_store."
        assert not self.config.moe_apply_probs_on_input, (
            "`turbo_fused_grouped_gemm` applies probs inside the fused GLU; "
            "`moe_apply_probs_on_input` (scale before fc1) is a different contract."
        )
        assert (
            not self.offload_expert_fc1 and not self.offload_moe_act
        ), "`turbo_fused_grouped_gemm` does not support expert fc1 / moe-act offload."

    def _use_explicit_quantization_padding(self) -> bool:
        """Whether this forward must pad expert groups before quantization."""
        if not (self.config.fp8 or self.config.fp4):
            return False
        if self.moe_router_padding_for_quantization:
            return False
        return not self.turbo_grouped_gemm_without_padding

    def bias_act_func(self, intermediate_parallel, bias_parallel, permuted_probs):
        """
        Applies bias and activation function to the output of linear_fc1.
        """
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if permuted_probs is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if self.activation_func == F.silu and self.config.gated_linear_unit:
                from primus.backends.megatron.core.fusions.fused_bias_swiglu import (
                    weighted_bias_swiglu_impl,
                )

                # dtype is handled inside the fused kernel
                intermediate_parallel = weighted_bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                    self.config.activation_func_clamp_value,
                )
            elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                intermediate_parallel = weighted_bias_quick_geglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    permuted_probs,
                    self.config.activation_func_fp8_input_store,
                    self.config.glu_linear_offset,
                    self.config.activation_func_clamp_value,
                )
            else:
                raise ValueError("Only support fusion of swiglu and quick_gelu in TEGroupedMLP.")
        elif self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu:
            assert bias_parallel is None, "Bias is not supported with fused weighted squared relu."
            intermediate_parallel = weighted_squared_relu_impl(intermediate_parallel, permuted_probs)
        else:
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (x_linear + self.config.glu_linear_offset)

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)
            original_dtype = intermediate_parallel.dtype
            intermediate_parallel = intermediate_parallel * permuted_probs
            intermediate_parallel = intermediate_parallel.to(original_dtype)
        return intermediate_parallel

    def bias_act_func_with_mask(
        self,
        intermediate_parallel: torch.Tensor,
        bias_parallel: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_experts: Union[torch.Tensor, None] = None,
    ):
        if self.use_turbo_fused_act_with_probs:
            from primus.backends.megatron.core.extensions.primus_turbo import (
                fused_bias_act_with_probs,
            )

            assert (
                tokens_per_experts is not None
            ), "tokens_per_experts is required when `use_turbo_fused_act_with_probs` is True."

            # TODO(ruibin)： The turbo kernel takes no offset / fp8-input-store arguments
            assert (
                self.config.glu_linear_offset == 0.0
            ), "`use_turbo_fused_act_with_probs` does not support a non-zero glu_linear_offset."
            assert (
                not self.config.activation_func_fp8_input_store
            ), "`use_turbo_fused_act_with_probs` does not support activation_func_fp8_input_store."

            if self.activation_func == F.silu and self.config.gated_linear_unit:
                activation = "silu"
            elif self.activation_func == F.gelu and self.config.gated_linear_unit:
                activation = "gelu"
            else:
                raise ValueError(
                    "Only support fusion of swiglu and gelu in PrimusGroupedMLP when `use_turbo_fused_act_with_probs` is True."
                )

            # `forward()` unsqueeze(-1)'s `permuted_probs` to [tokens, 1] so the non-fused
            # asserts ndim == 1. Squeeze back to 1D for the fused kernel only.
            probs_1d = permuted_probs.squeeze(-1) if permuted_probs.dim() == 2 else permuted_probs

            # dtype is handled inside the fused kernel
            return fused_bias_act_with_probs(
                intermediate_parallel,
                bias_parallel,
                probs_1d,
                tokens_per_experts,
                activation,
                self.config.activation_func_clamp_value,
            )
        else:
            # use the original bias_act_func from TEGroupedMLP, ignore the tokens_per_experts
            return self.bias_act_func(intermediate_parallel, bias_parallel, permuted_probs)

    def _forward_with_turbo_fused_grouped_mlp(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboLowPrecisionGlobalStateManager,
        )

        assert (
            tokens_per_expert.device == permuted_local_hidden_states.device
        ), "tokens_per_expert and permuted_local_hidden_states must be on the same device"

        if self.activation_func == F.silu and self.config.gated_linear_unit:
            activation = "silu"
        elif self.activation_func == F.gelu and self.config.gated_linear_unit:
            activation = "gelu"
        else:
            raise ValueError(
                "Only support fusion of silu and gelu in PrimusGroupedMLP when `turbo_fused_grouped_gemm` is True."
            )

        x, w1, fuse_w1_grad_accum = self.linear_fc1.prepare_weights(permuted_local_hidden_states)
        x, w2, fuse_w2_grad_accum = self.linear_fc2.prepare_weights(x)
        assert fuse_w1_grad_accum == fuse_w2_grad_accum, (
            "fc1 and fc2 resolved different fuse_wgrad_accum_pattern values: "
            f"fc1={fuse_w1_grad_accum!r}, fc2={fuse_w2_grad_accum!r}"
        )
        fuse_wgrad_accum_pattern = fuse_w1_grad_accum

        if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
            from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8

            mlp_op = grouped_mlp_fp8
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
            from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4

            mlp_op = grouped_mlp_fp4
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        else:
            raise ValueError(
                "`turbo_fused_grouped_gemm` only available when Turbo FP8 "
                "(current or mxfp8 scaling) or Turbo FP4 is enabled."
            )

        def _turbo_fused_grouped_mlp(hidden, group_lens, probs):
            # w1/w2 stay in the closure: they may be a QuantizedTensorPair (not a
            # Tensor checkpoint arg). Weight prep stays outside the checkpoint so
            # the first-microbatch cache / de-osc marker / grad bridge run once.
            return mlp_op(
                hidden,
                w1,
                w2,
                group_lens,
                probs=probs,
                trans_w1=True,
                trans_w2=True,
                config=quant_config.data(),
                activation=activation,
                fuse_wgrad_accum_pattern=fuse_wgrad_accum_pattern,
                clamp_limit=self.config.activation_func_clamp_value,
            )

        if self.activation_recompute:
            output = tensor_parallel.checkpoint(
                _turbo_fused_grouped_mlp, False, x, tokens_per_expert, permuted_probs
            )
        else:
            output = _turbo_fused_grouped_mlp(x, tokens_per_expert, permuted_probs)

        # NOTE: None means no bias
        return output, None

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel

        # NOTE: tokens_per_expert is on GPU, so we need to convert it to a list of ints.
        tokens_per_expert_cpu = tokens_per_expert.tolist()

        return TEGroupedMLP._apply_bias(
            intermediate_parallel, bias_parallel, tokens_per_expert_cpu, permuted_probs
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of PrimusGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        if self.turbo_fused_grouped_gemm:
            supported, reason = is_turbo_fused_grouped_mlp_supported()
            if supported:
                return self._forward_with_turbo_fused_grouped_mlp(
                    permuted_local_hidden_states, tokens_per_expert, permuted_probs
                )
            global _TURBO_FUSED_GROUPED_GEMM_FALLBACK_WARNED
            if not _TURBO_FUSED_GROUPED_GEMM_FALLBACK_WARNED:
                warning_rank_0(
                    f"`turbo_fused_grouped_gemm` is not used because {reason}. "
                    "Falling back to unfused PrimusGroupedMLP."
                )
                _TURBO_FUSED_GROUPED_GEMM_FALLBACK_WARNED = True

        use_explicit_quantization_padding = self._use_explicit_quantization_padding()
        if use_explicit_quantization_padding:
            # NOTE: When moe_router_padding_for_quantization is true the token is padded. So we can skip the padding here to reduce cpu sync.
            tokens_per_expert_cpu: list[int] = tokens_per_expert.tolist()
            actual_tokens_per_expert_cpu: list[int] = tokens_per_expert_cpu
            permuted_local_hidden_states, tokens_per_expert_cpu = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert_cpu
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert_cpu
            )
            tokens_per_expert = torch.tensor(
                tokens_per_expert_cpu, device=permuted_local_hidden_states.device
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        with off_interface(
            self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
        ) as permuted_local_hidden_states:
            fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output = off_interface.group_commit(
                fc1_output,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = self.activation_checkpoint.checkpoint(
                    self.bias_act_func_with_mask, fc1_output, bias_parallel, permuted_probs, tokens_per_expert
                )
        else:
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                # Honor use_turbo_fused_act_with_probs independently of activation
                # recomputation. The helper falls back to bias_act_func when disabled.
                bias_act_output = self.bias_act_func_with_mask(
                    fc1_output,
                    bias_parallel,
                    permuted_probs,
                    tokens_per_expert,
                )
        output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # Delay the offload of the moe act until after the linear_fc2 has been computed
        # to make sure the fc1_output is reloaded to GPU before recomputing moe_act.
        if self.offload_moe_act:
            output = off_interface.group_commit(output, name="moe_act", forced_released_tensors=[fc1_output])
        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

        # upad and concat the output
        if use_explicit_quantization_padding:
            output = self.quantization_unpadding(output, actual_tokens_per_expert_cpu)

        output_bias = None

        return output, output_bias
