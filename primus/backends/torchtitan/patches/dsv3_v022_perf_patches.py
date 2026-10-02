###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan v0.2.2 DeepSeek-V3 MoE memory fixes (Primus patches).

Reduce DSv3-16B HBM on MI355X with torchtitan v0.2.2 without modifying the
upstream submodule. The balanced-routing field migration
(``training.debug_moe_force_load_balance`` -> ``debug.moe_force_load_balance``)
is already handled in the DeepSeek configs on main.

Compile whole TransformerBlocks with graph breaks enabled, while explicitly
keeping GroupedExperts eager. This places the graph break at the routed-expert
EP/FSDP boundary: surrounding attention, routing, combine, residual, and dense
compute can still be captured without pulling expert communication and FSDP
hooks into the same graph. It avoids the BF16 HBM regression from separately
compiled MoE children and the FP8 NaNs from compiling GroupedExperts.

The MoE forward replacement also changes its fp32 ``bmm`` combine into a bf16
weighted sum, dropping the fp32 activation copy retained across MoE layers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0


def _model_name_str(ctx: PatchContext) -> str:
    """Normalize ctx.model_name (str or model namespace) to a plain string."""
    model = ctx.model_name
    if model is None:
        return ""
    if isinstance(model, str):
        return model
    name = getattr(model, "name", None)
    if name is not None:
        return str(name)
    return str(model)


def _is_deepseek_model(ctx: PatchContext) -> bool:
    return "deepseek" in _model_name_str(ctx).lower()


def _compile_enabled(ctx: PatchContext) -> bool:
    return bool(get_param(ctx, "compile.enable", False))


def _apply_unified_compile(model: nn.Module, compile_config: Any, ep_enabled: bool) -> None:
    """Compile whole blocks with an explicit graph break at routed experts."""
    import torchtitan.models.moe.moe as moe_module
    from torchtitan.tools.logging import logger

    # Preserve TorchTitan's required setup for dynamic token-choice MoE shapes.
    torch._dynamo.config.capture_scalar_outputs = True

    experts_forward = moe_module.GroupedExperts.forward
    if not getattr(experts_forward, "_primus_graph_break", False):
        experts_forward = torch.compiler.disable(experts_forward, recursive=False)
        experts_forward._primus_graph_break = True
        moe_module.GroupedExperts.forward = experts_forward

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(
            transformer_block,
            backend=compile_config.backend,
            fullgraph=False,
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info(
        "Compiling whole TransformerBlocks with an eager GroupedExperts "
        "EP/FSDP boundary (Primus unified policy)"
    )


@register_patch(
    "torchtitan.dsv3.whole_block_compile",
    backend="torchtitan",
    phase="setup",
    description="Compile whole blocks with an eager GroupedExperts boundary",
    condition=lambda ctx: _is_deepseek_model(ctx) and _compile_enabled(ctx),
)
def patch_whole_block_compile(ctx: PatchContext) -> None:
    """Install the unified, numerically safe whole-block compile policy."""
    import torchtitan.models.deepseek_v3.infra.parallelize as deepseek_parallelize
    import torchtitan.models.llama4.infra.parallelize as llama4_parallelize

    # DeepSeek imports apply_compile by value, so patch both the source module
    # and the already-bound local alias regardless of import order.
    llama4_parallelize.apply_compile = _apply_unified_compile
    deepseek_parallelize.apply_compile = _apply_unified_compile
    log_rank_0(
        "[Patch:torchtitan.dsv3.whole_block_compile] "
        "Patched DeepSeek apply_compile with unified graph-break compilation",
    )


def _moe_forward_bf16_combine(self: Any, x: torch.Tensor) -> torch.Tensor:
    """MoE.forward with bf16 weighted combine (no fp32 bmm copy)."""
    bs, slen, dim = x.shape
    x = x.view(-1, dim)

    (
        top_scores,
        selected_experts_indices,
        num_tokens_per_expert,
    ) = self.router(x, self.expert_bias)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    (
        top_scores_experts_sorted,
        token_indices_experts_sorted,
        num_tokens_per_expert,
    ) = self.reorderer(top_scores, selected_experts_indices)

    routed_input = x[token_indices_experts_sorted // self.router.top_k]

    if self.score_before_experts:
        routed_input = (routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

    routed_output = self.experts(routed_input, num_tokens_per_expert)

    out = self.shared_experts(x) if self.shared_experts is not None else None

    routed_output_unsorted = torch.zeros(
        (bs * slen * self.router.top_k, dim),
        dtype=routed_output.dtype,
        device=routed_output.device,
    )
    routed_output_unsorted[token_indices_experts_sorted] = routed_output
    routed_output_unsorted = routed_output_unsorted.reshape(-1, self.router.top_k, dim)

    if not self.score_before_experts:
        out_experts = (routed_output_unsorted * top_scores.reshape(-1, self.router.top_k, 1)).sum(dim=1)
    else:
        out_experts = routed_output_unsorted.sum(dim=1)

    if out is None:
        return out_experts.reshape(bs, slen, dim)
    return (out + out_experts).reshape(bs, slen, dim)


@register_patch(
    "torchtitan.dsv3.moe_bf16_combine",
    backend="torchtitan",
    phase="setup",
    description="MoE expert combine in bf16 (avoid fp32 bmm activation retention)",
    condition=lambda ctx: _is_deepseek_model(ctx),
)
def patch_moe_bf16_combine(ctx: PatchContext) -> None:
    """Replace MoE.forward to drop the fp32 bmm copy in the combine step."""
    import torchtitan.models.moe.moe as moe_module

    # The outer MoE TransformerBlock is already eager. An additional
    # whole-function compiler.disable wrapper has no numerical or memory effect.
    moe_module.MoE.forward = _moe_forward_bf16_combine
    log_rank_0("[Patch:torchtitan.dsv3.moe_bf16_combine] Patched MoE.forward with bf16 weighted combine")
