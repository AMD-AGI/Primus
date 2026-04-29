###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Legacy GroupedMLP weight gradient split patch for pipeline parallelism.

When moe_use_legacy_grouped_gemm=True and use_turbo_grouped_mlp=False, MoE
experts use Megatron's native GroupedMLP whose forward calls gg.ops.gmm
(grouped_gemm.ops.GroupedGemm autograd Function).  That autograd Function
computes dgrad and wgrad together in backward, which prevents the Primus
pipeline scheduler from deferring wgrad.

This patch replaces GroupedMLP.forward so that the gmm calls go through
GroupedLinearWithWeightGradientStore (via the existing "lagacy-gg" backend
in zbpp_gemm.py), which splits dgrad/wgrad and feeds the wgrad closure to
WGradRunningCache / zero-bubble WeightGradStore.
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _make_patched_forward():
    """Build a replacement forward that uses wgrad-split grouped gemm."""
    from megatron.core import tensor_parallel

    from primus.backends.megatron.core.extensions.zbpp_gemm import (
        grouped_gemm_with_weight_gradient_store,
    )

    def _forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        assert self.config.bf16, "Currently GroupedMLP for MoE only supports bf16."
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            w1 = self.weight1
            w2 = self.weight2
            w1_reshape = (self.num_local_experts, self.config.hidden_size, -1)
            w2_reshape = (self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = grouped_gemm_with_weight_gradient_store(
                permuted_local_hidden_states,
                w1,
                tokens_per_expert,
                trans_b=False,
                weight_reshape_size=w1_reshape,
                gg_backend="lagacy-gg",
            )
            if self.activation_recompute:
                intermediate_parallel = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs,
                    fc1_output,
                    permuted_probs.unsqueeze(-1),
                )
                fc2_output = grouped_gemm_with_weight_gradient_store(
                    intermediate_parallel,
                    w2,
                    tokens_per_expert,
                    trans_b=False,
                    weight_reshape_size=w2_reshape,
                    gg_backend="lagacy-gg",
                )
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = grouped_gemm_with_weight_gradient_store(
                    intermediate_parallel,
                    w2,
                    tokens_per_expert,
                    trans_b=False,
                    weight_reshape_size=w2_reshape,
                    gg_backend="lagacy-gg",
                )
        else:
            assert torch.count_nonzero(tokens_per_expert) == 0
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                h = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None

    return _forward


@register_patch(
    "megatron.pp.legacy_grouped_mlp_wgrad_split",
    backend="megatron",
    phase="before_train",
    description="Patch legacy GroupedMLP.forward to split wgrad for pipeline parallelism",
    condition=lambda ctx: (
        (
            getattr(get_args(ctx), "patch_primus_pipeline", False)
            or getattr(get_args(ctx), "patch_zero_bubble", False)
        )
        and getattr(get_args(ctx), "moe_use_legacy_grouped_gemm", False)
        and not getattr(get_args(ctx), "use_turbo_grouped_mlp", False)
    ),
)
def patch_legacy_grouped_mlp_wgrad(ctx: PatchContext):
    from megatron.core.transformer.moe.experts import GroupedMLP

    GroupedMLP.forward = _make_patched_forward()
    log_rank_0(
        "[Patch:megatron.pp.legacy_grouped_mlp_wgrad_split] "
        "Patched GroupedMLP.forward for wgrad split (lagacy-gg backend)"
    )
