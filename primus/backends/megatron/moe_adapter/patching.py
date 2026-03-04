###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import sys


def patch_megatron_topk_router(router_cls, *, use_deprecated_moe: bool = False) -> None:
    """Centralized TopKRouter patch entrypoint."""
    sys.modules["megatron.core.transformer.moe.router"].TopKRouter = router_cls

    from megatron.core.transformer.moe import moe_layer

    moe_layer.TopKRouter = router_cls

    if use_deprecated_moe:
        from primus.backends.megatron.core.transformer.moe import deprecated_20251209

        deprecated_20251209.moe_layer.TopKRouter = router_cls


def patch_megatron_moe_dispatcher(dispatcher_cls) -> None:
    """Centralized MoE dispatcher patch entrypoint."""
    from megatron.core.transformer.moe import moe_layer, token_dispatcher

    token_dispatcher.MoEFlexTokenDispatcher = dispatcher_cls
    moe_layer.MoEFlexTokenDispatcher = dispatcher_cls
