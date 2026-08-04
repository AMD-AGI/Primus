###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 build context / spec-provider singleton.

Copied in shape from ``deepseek_v4/build_context.py`` (plan-2 P18), for
the same reason: without it every spec helper — the MoE spec factory, the
layer-spec factory, the block ``__init__`` — re-instantiates a provider,
paying the ``BackendSpecProvider`` setup cost repeatedly and leaving no
single place to audit which provider actually wired the K3 modules.

The provider is cached **on the config object** rather than in a
module-level global, so different configs get different providers and no
state leaks between unit tests.

Usage:

.. code-block:: python

    from primus.backends.megatron.core.models.kimi_k3.build_context import (
        resolve_k3_provider,
    )

    def some_builder(*, config):
        provider = resolve_k3_provider(config)
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an eager torch import in this lightweight module
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        KimiK3SpecProvider,
    )

_PROVIDER_ATTR = "_k3_spec_provider_singleton"


def resolve_k3_provider(config) -> "KimiK3SpecProvider":
    """Return a cached :class:`KimiK3SpecProvider` for ``config``.

    Args:
        config: a :class:`KimiK3TransformerConfig` instance (or anything
            that tolerates ``setattr``).

    Returns:
        The cached provider. The first call constructs it; later calls
        reuse it.
    """
    cached = getattr(config, _PROVIDER_ATTR, None)
    if cached is not None:
        return cached

    # Lazy import so this module stays importable from the dataclass module
    # without cyclic risk.
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        KimiK3SpecProvider,
    )

    provider = KimiK3SpecProvider(config=config)
    try:
        setattr(config, _PROVIDER_ATTR, provider)
    except Exception:
        # Some MagicMock-style configs in unit tests reject setattr; that is
        # fine, we just pay the construction cost on each call.
        pass
    return provider


def reset_k3_provider_cache(config) -> None:
    """Drop the cached provider on ``config``.

    For unit tests that need to force a rebuild, e.g. after monkey-patching
    the provider class.
    """
    if hasattr(config, _PROVIDER_ATTR):
        try:
            delattr(config, _PROVIDER_ATTR)
        except AttributeError:
            pass


__all__ = ["resolve_k3_provider", "reset_k3_provider_cache"]
