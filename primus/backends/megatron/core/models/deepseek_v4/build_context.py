###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-2 P18 — DeepSeek-V4 build context / provider singleton.

This module hosts the helpers that make sure a single
:class:`DeepSeekV4SpecProvider` is constructed per builder call and
threaded down to every spec helper. Without this, the provider was
re-instantiated inside ``_build_projection`` (block.py),
``DeepseekV4TransformerBlock.__init__`` (block.py), the layer-spec
factory (``deepseek_v4_layer_specs.py``), and the MTP spec helper
(``deepseek_v4_mtp_specs.py``) — every call paid the
``BackendSpecProvider`` setup cost and there was no single place to
audit which provider was actually wiring the V4 modules.

The implementation deliberately avoids module-level globals: the
provider is cached **on the config object** under a private attribute
name. This keeps each ``DeepSeekV4TransformerConfig`` instance
self-contained (different configs get different providers) and avoids
leaking state across unit tests.

Usage:

.. code-block:: python

    from .build_context import resolve_v4_provider

    def some_builder(*, config):
        provider = resolve_v4_provider(config)
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid eager torch import in this lightweight module
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        DeepSeekV4SpecProvider,
    )

_PROVIDER_ATTR = "_v4_spec_provider_singleton"


def resolve_v4_provider(config) -> "DeepSeekV4SpecProvider":
    """Return a cached :class:`DeepSeekV4SpecProvider` for ``config``.

    The provider is cached on the ``config`` object itself (under a
    private attribute name) so:

    * repeated calls during a single builder invocation reuse the
      same provider instance,
    * different configs always get fresh providers,
    * the cache is naturally garbage-collected when the config is
      released,
    * the helper is fully thread-safe-by-construction (each builder
      thread holds its own config).

    Args:
        config: a :class:`DeepSeekV4TransformerConfig` instance. The
            helper accesses ``config.__dict__`` directly so it works
            on dataclasses without additional setattr support.

    Returns:
        The cached :class:`DeepSeekV4SpecProvider`. The first call
        constructs it; later calls reuse it.
    """
    cached = getattr(config, _PROVIDER_ATTR, None)
    if cached is not None:
        return cached

    # Lazy import: this module must be lightweight enough to import
    # from the dataclass module without cyclic risks.
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        DeepSeekV4SpecProvider,
    )

    provider = DeepSeekV4SpecProvider(config=config)
    try:
        setattr(config, _PROVIDER_ATTR, provider)
    except Exception:
        # Some MagicMock-style configs in unit tests reject setattr; that
        # is OK — we just don't cache and pay the cost on each call.
        pass
    return provider


def reset_v4_provider_cache(config) -> None:
    """Drop the cached provider on ``config``. Intended for unit tests
    that need to force a re-build (e.g. after monkey-patching the
    provider class)."""
    if hasattr(config, _PROVIDER_ATTR):
        try:
            delattr(config, _PROVIDER_ATTR)
        except AttributeError:
            pass


__all__ = ["resolve_v4_provider", "reset_v4_provider_cache"]
