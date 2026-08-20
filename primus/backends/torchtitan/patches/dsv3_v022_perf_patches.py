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

Whole-block compilation is still needed to avoid the fragmented reserved HBM
caused by v0.2.2's per-submodule compilation. Compiling the entire MoE forward
is unsafe, however: it pulls expert-parallel token dispatch/combine and
GroupedExperts FSDP hooks into one inductor graph. On MI300X that silently
produces a NaN forward loss from the first step.

Use a safe dense-only boundary instead: dense blocks compile as a full graph,
while MoE TransformerBlocks remain eager. Compiling even the outer MoE block
with a graph break around ``MoE.forward`` still produces the NaN, whereas
v0.2.2's per-submodule policy reintroduces the fragmented reserved HBM this
patch was created to avoid. Keeping MoE blocks eager avoids both failure modes;
the expensive attention and grouped-GEMM work still uses Primus-Turbo kernels.
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


def _apply_dense_only_compile(model: nn.Module, compile_config: Any, ep_enabled: bool) -> None:
    """Whole-block compile dense layers and leave MoE TransformerBlocks eager."""
    from torchtitan.tools.logging import logger

    del ep_enabled  # MoE blocks stay eager, including EP dispatch.
    for layer_id, transformer_block in model.layers.named_children():
        if not transformer_block.moe_enabled:
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling dense TransformerBlocks; leaving MoE TransformerBlocks eager (Primus patch)")


@register_patch(
    "torchtitan.dsv3.whole_block_compile",
    backend="torchtitan",
    phase="setup",
    description="Whole-block compile dense layers; leave MoE TransformerBlocks eager",
    condition=lambda ctx: _is_deepseek_model(ctx) and _compile_enabled(ctx),
)
def patch_whole_block_compile(ctx: PatchContext) -> None:
    """Install low-fragment dense-only compilation."""
    from torchtitan.models.llama4.infra import parallelize as parallelize_module

    parallelize_module.apply_compile = _apply_dense_only_compile
    log_rank_0(
        "[Patch:torchtitan.dsv3.whole_block_compile] "
        "Patched apply_compile with eager MoE TransformerBlocks",
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

    # Keep this explicit eager boundary in case a caller wraps a larger parent
    # module in torch.compile outside the DeepSeek parallelization path.
    moe_module.MoE.forward = torch.compiler.disable(_moe_forward_bf16_combine)
    log_rank_0(
        "[Patch:torchtitan.dsv3.moe_bf16_combine] "
        "Patched MoE.forward (eager boundary, bf16 weighted combine)",
    )
