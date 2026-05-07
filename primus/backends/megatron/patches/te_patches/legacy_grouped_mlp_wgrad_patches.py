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

Megatron upstream removed ``GroupedMLP`` (commit 2caa681a1, "Cleanup: Remove
the deprecated GroupedMLP"). To preserve the legacy grouped-GEMM delayed
wgrad path that the Primus pipeline scheduler depends on, this patch falls
back to ``DeprecatedGroupedMLP`` from
``primus.backends.megatron.core.transformer.moe.deprecated_20251209.experts``
and re-injects it as ``megatron.core.transformer.moe.experts.GroupedMLP``
plus ``moe_module_specs.GroupedMLP`` so that downstream MoE spec
construction can reach it. The PrimusTurbo spec provider is also extended
(see ``transformer_engine_spec_provider.py``) to actually return this class
when the legacy + non-turbo + grouped-GEMM combination is selected.
"""

import sys

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _resolve_grouped_mlp_class():
    """Return the GroupedMLP class to patch.

    Prefers the upstream Megatron class when present; otherwise falls back to
    the Primus-maintained ``DeprecatedGroupedMLP`` and injects it back into
    Megatron's namespaces so that downstream spec providers and importers
    that still reference ``GroupedMLP`` continue to work.
    """
    try:
        from megatron.core.transformer.moe.experts import GroupedMLP

        return GroupedMLP, "megatron.core.transformer.moe.experts.GroupedMLP"
    except ImportError:
        pass

    from primus.backends.megatron.core.transformer.moe.deprecated_20251209.experts import (
        DeprecatedGroupedMLP,
    )

    experts_mod = sys.modules.get("megatron.core.transformer.moe.experts")
    if experts_mod is not None:
        experts_mod.GroupedMLP = DeprecatedGroupedMLP

    try:
        from megatron.core.models.gpt import moe_module_specs

        moe_module_specs.GroupedMLP = DeprecatedGroupedMLP
    except ImportError:
        pass

    return (
        DeprecatedGroupedMLP,
        "primus.backends.megatron.core.transformer.moe.deprecated_20251209.experts.DeprecatedGroupedMLP",
    )


def _make_patched_forward():
    """Build a replacement forward that uses wgrad-split grouped gemm.

    The patched forward mirrors upstream Megatron's last ``GroupedMLP.forward``
    (commit ``2caa681a1^``, before the class was removed) but routes the two
    ``gg.ops.gmm`` calls through ``grouped_gemm_with_weight_gradient_store``
    with ``gg_backend='lagacy-gg'`` so the weight-gradient is split out and
    queued into the Primus pipeline wgrad cache.

    ``permuted_probs`` is required by the new ``MoELayer.routed_experts_compute``
    signature; we always apply it (multiplicatively, after activation) to keep
    the router gradient path intact, mirroring upstream
    ``activation_func_with_probs``. ``moe_apply_probs_on_input=True`` matches
    upstream behaviour by applying probs on the input first then resetting the
    post-activation factor to ``ones_like(probs)`` so the autograd graph stays
    fully connected.
    """
    from primus.backends.megatron.core.extensions.zbpp_gemm import (
        grouped_gemm_with_weight_gradient_store,
    )

    def _grouped_fc1(weight, hidden, tokens_per_expert, num_local_experts, hidden_size):
        return grouped_gemm_with_weight_gradient_store(
            hidden,
            weight,
            tokens_per_expert,
            trans_b=False,
            weight_reshape_size=(num_local_experts, hidden_size, -1),
            gg_backend="lagacy-gg",
        )

    def _grouped_fc2(weight, intermediate, tokens_per_expert, num_local_experts, hidden_size):
        return grouped_gemm_with_weight_gradient_store(
            intermediate,
            weight,
            tokens_per_expert,
            trans_b=False,
            weight_reshape_size=(num_local_experts, -1, hidden_size),
            gg_backend="lagacy-gg",
        )

    def _ensure_activation_func(self):
        """Resolve the ``activation_func`` attribute used by patched forward.

        Upstream ``GroupedMLP.__init__`` sets ``self.activation_func`` (a
        ``glu`` closure for gated activations or ``config.activation_func``
        otherwise). The Primus ``DeprecatedGroupedMLP`` keeps the same
        attribute. Fall back to constructing one on-the-fly so the patch
        still works on classes that do not eagerly populate it.
        """
        existing = getattr(self, "activation_func", None)
        if existing is not None:
            return existing
        if self.config.gated_linear_unit:

            def glu(x):
                x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x_glu) * x_linear

            return glu
        return self.config.activation_func

    def _activation_with_probs(self, x, probs_unsqueezed):
        """Mirror upstream ``activation_func_with_probs`` semantics.

        ``activation_func(x) * probs`` while preserving the input dtype.
        ``probs_unsqueezed`` is expected to be ``permuted_probs.unsqueeze(-1)``.
        """
        activation = _ensure_activation_func(self)
        dtype = x.dtype
        res = activation(x) * probs_unsqueezed
        return res.to(dtype)

    def _forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor = None,
        routing_map: torch.Tensor = None,
    ):
        del routing_map  # unused by the BF16 grouped-gemm path
        assert self.config.bf16, "Currently GroupedMLP for MoE only supports bf16."
        assert (
            permuted_probs is not None
        ), "legacy_grouped_mlp_wgrad_split patch requires permuted_probs from MoELayer"

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied; reset post-activation factor to ones to
            # keep the autograd graph connected without double-applying.
            permuted_probs = torch.ones_like(permuted_probs)

        hidden_size = self.config.hidden_size
        num_local_experts = self.num_local_experts

        if permuted_local_hidden_states.nelement() != 0:
            fc1_output = _grouped_fc1(
                self.weight1,
                permuted_local_hidden_states,
                tokens_per_expert,
                num_local_experts,
                hidden_size,
            )
            intermediate_parallel = _activation_with_probs(
                self, fc1_output, permuted_probs.unsqueeze(-1)
            )
            fc2_output = _grouped_fc2(
                self.weight2,
                intermediate_parallel,
                tokens_per_expert,
                num_local_experts,
                hidden_size,
            )
        else:
            assert torch.count_nonzero(tokens_per_expert) == 0
            # No tokens routed locally: keep a gradient path on expert weights
            # via plain matmul to avoid grouped_gemm's empty-input edge cases.
            w1_flat = self.weight1.view(hidden_size, -1)
            w2_flat = self.weight2.view(-1, hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1_flat)
            h = _activation_with_probs(self, h, permuted_probs.unsqueeze(-1))
            fc2_output = torch.matmul(h, w2_flat)

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
    grouped_mlp_cls, source = _resolve_grouped_mlp_class()
    grouped_mlp_cls.forward = _make_patched_forward()
    log_rank_0(
        "[Patch:megatron.pp.legacy_grouped_mlp_wgrad_split] "
        f"Patched {source}.forward for wgrad split (lagacy-gg backend)"
    )
