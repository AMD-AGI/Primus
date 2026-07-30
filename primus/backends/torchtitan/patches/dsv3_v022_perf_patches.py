###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan v0.2.2 DeepSeek-V3 MoE performance / memory fixes (Primus patches).

Fixes the false DSv3-16B regression on MI355X (image 46075ff) without modifying
the upstream torchtitan submodule:

1. Map legacy ``training.debug_moe_force_load_balance`` to
   ``debug.moe_force_load_balance`` (v0.2.2 field rename).
2. Replace v0.2.2 ``apply_compile`` (capture_scalar_outputs + per-submodule compile)
   with whole-block compile matching v0.1.0 behavior.
3. Replace MoE combine fp32 ``bmm`` with bf16 weighted sum (drops ~22GB live HBM).
"""

from __future__ import annotations

from types import SimpleNamespace
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


@register_patch(
    "torchtitan.dsv3.moe_force_load_balance_compat",
    backend="torchtitan",
    phase="build_args",
    description="Map training.debug_moe_force_load_balance to debug.moe_force_load_balance",
    condition=lambda ctx: _is_deepseek_model(ctx),
)
def patch_moe_force_load_balance_compat(ctx: PatchContext) -> None:
    """Honor Primus legacy training.debug_moe_force_load_balance on torchtitan v0.2.2."""
    backend_args = ctx.extra.get("backend_args")
    if backend_args is None:
        return

    training = getattr(backend_args, "training", None)
    legacy = bool(getattr(training, "debug_moe_force_load_balance", False))

    debug = getattr(backend_args, "debug", None)
    if debug is None:
        debug = SimpleNamespace()
        backend_args.debug = debug

    current = bool(getattr(debug, "moe_force_load_balance", False))
    if legacy and not current:
        debug.moe_force_load_balance = True
        log_rank_0(
            "[Patch:torchtitan.dsv3.moe_force_load_balance_compat] "
            "Mapped training.debug_moe_force_load_balance=True -> debug.moe_force_load_balance=True",
        )


@register_patch(
    "torchtitan.dsv3.whole_block_compile",
    backend="torchtitan",
    phase="setup",
    description="Whole-block torch.compile; leave capture_scalar_outputs disabled",
    condition=lambda ctx: _is_deepseek_model(ctx) and _compile_enabled(ctx),
)
def patch_whole_block_compile(ctx: PatchContext) -> None:
    """Replace v0.2.2 apply_compile with v0.1.0-style whole TransformerBlock compile."""
    from torchtitan.config.job_config import Compile as CompileConfig
    from torchtitan.models.llama4.infra import parallelize as parallelize_module
    from torchtitan.tools.logging import logger

    def apply_compile_patched(
        model: nn.Module, compile_config: CompileConfig, ep_enabled: bool
    ) -> None:
        # Match v0.1.0: do NOT set capture_scalar_outputs=True (avoids fragmentation).
        for layer_id, transformer_block in model.layers.named_children():
            fullgraph = not transformer_block.moe_enabled
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=fullgraph,
            )
            model.layers.register_module(layer_id, transformer_block)

        logger.info("Compiling each TransformerBlock with torch.compile (Primus whole-block patch)")

    parallelize_module.apply_compile = apply_compile_patched
    log_rank_0(
        "[Patch:torchtitan.dsv3.whole_block_compile] "
        "Patched torchtitan.models.llama4.infra.parallelize.apply_compile",
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
        routed_input = (
            routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

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
        out_experts = (
            routed_output_unsorted * top_scores.reshape(-1, self.router.top_k, 1)
        ).sum(dim=1)
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

    moe_module.MoE.forward = _moe_forward_bf16_combine
    log_rank_0(
        "[Patch:torchtitan.dsv3.moe_bf16_combine] "
        "Patched torchtitan.models.moe.moe.MoE.forward (bf16 weighted combine)",
    )
