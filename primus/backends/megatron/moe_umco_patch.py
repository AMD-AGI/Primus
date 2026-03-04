from __future__ import annotations

import logging
import os
from typing import Any

from primus.moe_umco.megatron.hooks import maybe_wrap_megatron_moe_dispatcher

logger = logging.getLogger("primus.moe_umco")
_PATCHED = False


def _enabled() -> bool:
    return os.environ.get("PRIMUS_UMCO_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}


def apply_umco_patches(exp_config: dict[str, Any] | None = None) -> None:
    global _PATCHED
    if _PATCHED:
        return
    if not _enabled():
        return

    try:
        from megatron.core.transformer.moe import moe_layer, token_dispatcher
    except Exception as exc:
        logger.warning("UMCO patch skipped: failed importing megatron MoE modules: %s", exc)
        return

    dispatcher_cls = getattr(token_dispatcher, "MoEFlexTokenDispatcher", None)
    if dispatcher_cls is None:
        logger.warning("UMCO patch skipped: MoEFlexTokenDispatcher not found.")
        return

    wrapped_cls = maybe_wrap_megatron_moe_dispatcher(dispatcher_cls, exp_config=exp_config)
    if wrapped_cls is dispatcher_cls:
        return

    setattr(token_dispatcher, "MoEFlexTokenDispatcher", wrapped_cls)
    if hasattr(moe_layer, "MoEFlexTokenDispatcher"):
        setattr(moe_layer, "MoEFlexTokenDispatcher", wrapped_cls)

    _PATCHED = True
    logger.info("UMCO patch applied: MoEFlexTokenDispatcher -> %s", wrapped_cls.__name__)
