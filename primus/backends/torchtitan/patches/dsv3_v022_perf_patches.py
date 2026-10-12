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

Compiling an outer MoE TransformerBlock is unsafe: it pulls expert-parallel
token dispatch/combine and GroupedExperts FSDP hooks into one inductor graph,
which silently produces NaNs. Keeping the whole MoE block eager is correct but
costs about 18 GiB on DSv3-16B FP8.

Use a selective boundary instead. Dense blocks compile as a full graph; MoE
parents stay eager while their pure-compute attention, norms, router,
reorderer, and shared experts compile independently. Routed experts, EP
dispatch/combine, FSDP hooks, output unsort, and residuals remain eager. This
keeps training finite while recovering most of the eager-MoE memory and
throughput loss.

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


_GROUPED_MM_SENTINEL = "_run_experts_grouped_mm_dynamic"
_MOE_BLOCK_COMPILE_ATTRS = ("attention", "attention_norm", "ffn_norm")
_MOE_COMPILE_ATTRS = ("router", "reorderer", "shared_experts")


def _compile_attr(module: nn.Module, attr_name: str, backend: str) -> None:
    submodule = getattr(module, attr_name, None)
    if submodule is not None:
        setattr(
            module,
            attr_name,
            torch.compile(submodule, backend=backend, fullgraph=True),
        )


def _compile_grouped_mm(backend: str, ep_enabled: bool) -> None:
    """Keep Turbo grouped-MM eager; compile TorchTitan's implementation."""
    import torchtitan.models.moe.moe as moe_module

    if _GROUPED_MM_SENTINEL in moe_module._run_experts_grouped_mm.__qualname__:
        return

    compiled_fn = torch.compile(
        moe_module._run_experts_grouped_mm,
        backend=backend,
        fullgraph=True,
    )
    if not ep_enabled:
        moe_module._run_experts_grouped_mm = compiled_fn
        return

    def _run_experts_grouped_mm_dynamic(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        torch._dynamo.mark_dynamic(x, 0)
        return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

    moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic


def _apply_selective_compile(model: nn.Module, compile_config: Any, ep_enabled: bool) -> None:
    """Compile pure MoE compute while keeping unsafe communication boundaries eager."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
    )
    from torchtitan.tools.logging import logger

    backend = compile_config.backend

    for layer_id, transformer_block in model.layers.named_children():
        if not transformer_block.moe_enabled:
            transformer_block = torch.compile(
                transformer_block,
                backend=backend,
                fullgraph=True,
            )
        else:
            block = (
                transformer_block._checkpoint_wrapped_module
                if isinstance(transformer_block, CheckpointWrapper)
                else transformer_block
            )
            for attr_name in _MOE_BLOCK_COMPILE_ATTRS:
                _compile_attr(block, attr_name, backend)
            for attr_name in _MOE_COMPILE_ATTRS:
                _compile_attr(block.moe, attr_name, backend)

        model.layers.register_module(layer_id, transformer_block)

    _compile_grouped_mm(backend, ep_enabled)
    logger.info(
        "Compiling dense TransformerBlocks and safe MoE compute submodules; "
        "leaving outer MoE/EP/FSDP boundaries eager (Primus patch)"
    )


@register_patch(
    "torchtitan.dsv3.whole_block_compile",
    backend="torchtitan",
    phase="setup",
    description="Compile dense blocks and safe MoE compute submodules",
    condition=lambda ctx: _is_deepseek_model(ctx) and _compile_enabled(ctx),
)
def patch_whole_block_compile(ctx: PatchContext) -> None:
    """Install the numerically safe, low-memory selective compile policy."""
    import torchtitan.models.deepseek_v3.infra.parallelize as deepseek_parallelize
    import torchtitan.models.llama4.infra.parallelize as llama4_parallelize

    # DeepSeek imports apply_compile by value, so patch both the source module
    # and the already-bound local alias regardless of import order.
    llama4_parallelize.apply_compile = _apply_selective_compile
    deepseek_parallelize.apply_compile = _apply_selective_compile
    log_rank_0(
        "[Patch:torchtitan.dsv3.whole_block_compile] "
        "Patched DeepSeek apply_compile with safe MoE selective compilation",
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
