###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Initialize the GPT-OSS MoE router gate bias on TorchTitan v0.2.2.

Upstream added a biased router projection for GPT-OSS but its explicit
``init_weights`` path initializes only ``gate.weight``. TorchTitan constructs
the model on the meta device and materializes it with ``to_empty``, so the
constructor's original bias initialization is discarded. Depending on the
reused allocator contents, the uninitialized bias can contain NaN/Inf and make
the first router softmax (and therefore the first training loss) non-finite.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_ATTR = "_primus_gptoss_router_bias_init"


def _gptoss_router_bias_enabled(ctx: PatchContext) -> bool:
    return get_param(ctx, "model.name", None) == "gpt_oss"


def _make_router_init_weights_wrapper(
    original: Callable[[Any, float], None],
) -> Callable[[Any, float], None]:
    """Run upstream initialization, then deterministically zero gate bias."""

    def init_weights(self: Any, init_std: float) -> None:
        original(self, init_std)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    setattr(init_weights, _PATCH_ATTR, True)
    return init_weights


@register_patch(
    "torchtitan.gptoss.router_bias_init",
    backend="torchtitan",
    backend_versions=["0.2.2"],
    phase="setup",
    description="Initialize GPT-OSS MoE router gate bias after meta-device materialization",
    condition=_gptoss_router_bias_enabled,
)
def patch_gptoss_router_bias_init(ctx: PatchContext) -> None:  # noqa: ARG001
    from torchtitan.models.moe.moe import TokenChoiceTopKRouter

    original = TokenChoiceTopKRouter.init_weights
    if getattr(original, _PATCH_ATTR, False):
        return

    TokenChoiceTopKRouter.init_weights = _make_router_init_weights_wrapper(original)
    log_rank_0(
        "[Patch:torchtitan.gptoss.router_bias_init] "
        "GPT-OSS MoE router gate bias will be initialized to zero",
    )


_FORCE_BALANCE_ATTR = "_primus_gptoss_force_load_balance"


def _gptoss_force_load_balance_enabled(ctx: PatchContext) -> bool:
    return get_param(ctx, "model.name", None) == "gpt_oss" and bool(
        get_param(ctx, "debug.moe_force_load_balance", False)
    )


@register_patch(
    "torchtitan.gptoss.debug_force_load_balance",
    backend="torchtitan",
    backend_versions=["0.2.2"],
    phase="setup",
    description="Honor debug.moe_force_load_balance on GPT-OSS, whose ModelArgs never read it",
    condition=_gptoss_force_load_balance_enabled,
)
def patch_gptoss_debug_force_load_balance(ctx: PatchContext) -> None:  # noqa: ARG001
    # DeepSeek-V3, Qwen3 and Llama 4 copy debug.moe_force_load_balance onto
    # moe_args._debug_force_load_balance in update_from_config; GPT-OSS does not,
    # so the flag parses and is silently ignored. Overriding moe_args from the
    # command line instead replaces the whole dataclass with its defaults.
    from torchtitan.models.moe.moe import TokenChoiceTopKRouter

    original = TokenChoiceTopKRouter.__init__
    if getattr(original, _FORCE_BALANCE_ATTR, False):
        return

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["_debug_force_load_balance"] = True
        original(self, *args, **kwargs)

    setattr(__init__, _FORCE_BALANCE_ATTR, True)
    TokenChoiceTopKRouter.__init__ = __init__
    log_rank_0(
        "[Patch:torchtitan.gptoss.debug_force_load_balance] "
        "GPT-OSS MoE routers use debug round-robin routing (debug.moe_force_load_balance)",
    )
