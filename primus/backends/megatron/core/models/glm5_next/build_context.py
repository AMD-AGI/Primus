###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One cached :class:`Glm5NextSpecProvider` per config (DeepSeek-V4 / K3 pattern)."""

from __future__ import annotations

_PROVIDER_ATTR = "_glm5_next_spec_provider_singleton"


def resolve_glm5_next_provider(config):
    cached = getattr(config, _PROVIDER_ATTR, None)
    if cached is not None:
        return cached
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        Glm5NextSpecProvider,
    )

    provider = Glm5NextSpecProvider(config=config)
    try:
        setattr(config, _PROVIDER_ATTR, provider)
    except (AttributeError, TypeError):
        # Best-effort cache attach: some config objects may disallow dynamic attrs.
        # Returning `provider` keeps behavior correct even when caching is unavailable.
        pass
    return provider


__all__ = ["resolve_glm5_next_provider"]
