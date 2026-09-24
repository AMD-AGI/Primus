# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Backend selection for the WAN layer specs.

``config.transformer_impl`` plus the precision fields (``fp4`` / ``fp8``) decide
which :class:`BackendSpecProvider` builds a model's linears, norms, and
attention core. :func:`resolve_diffusion_backend` mirrors the ladder inlined in
``flux/layer_spec.get_flux_layer_spec``; the two are independent copies and a
new precision branch has to be added to both.

The second return value is the provider for *sensitive* layers -- the first and
last N blocks that stay at higher precision when ``sensitive_layers_enabled``
is set. It is ``None`` whenever sensitive layers are off or the main backend is
not a low-precision one.
"""

from typing import Optional, Tuple

try:
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE_SPEC_PROVIDER = True
except ImportError:
    HAVE_TE_SPEC_PROVIDER = False
    TESpecProvider = None

try:
    from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
        PrimusTurboFloat8LocalSpecProvider,
        PrimusTurboLocalSpecProvider,
    )

    HAVE_PRIMUS_TURBO_LOCAL = True
except ImportError:
    HAVE_PRIMUS_TURBO_LOCAL = False
    PrimusTurboLocalSpecProvider = None
    PrimusTurboFloat8LocalSpecProvider = None

# Separate guard so a missing primus_turbo_mxfp4_local does not break FP8.
try:
    from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
        PrimusTurboMXFP4LocalSpecProvider,
    )
except ImportError:
    PrimusTurboMXFP4LocalSpecProvider = None


def resolve_diffusion_backend(config) -> Tuple[object, Optional[object]]:
    """Select the spec providers for a diffusion model's layers.

    Returns ``(backend, sensitive_backend)``. ``sensitive_backend`` is ``None``
    unless the main backend is MXFP4 and ``sensitive_layer_precision`` names a
    higher-precision provider for the first/last layers.
    """
    sensitive_backend = None

    if config.transformer_impl == "local":
        if config.fp4 is not None and PrimusTurboMXFP4LocalSpecProvider is not None:
            backend = PrimusTurboMXFP4LocalSpecProvider()

            sensitive_precision = getattr(config, "sensitive_layer_precision", "bf16")
            if sensitive_precision == "tw_fp8":
                sensitive_backend = PrimusTurboFloat8LocalSpecProvider()
            elif sensitive_precision == "bf16":
                sensitive_backend = PrimusTurboLocalSpecProvider()
        elif (
            config.fp8 is not None
            and HAVE_PRIMUS_TURBO_LOCAL
            and PrimusTurboFloat8LocalSpecProvider is not None
        ):
            backend = PrimusTurboFloat8LocalSpecProvider()
        elif HAVE_PRIMUS_TURBO_LOCAL and PrimusTurboLocalSpecProvider is not None:
            backend = PrimusTurboLocalSpecProvider()
        else:
            from megatron.core.models.backends import LocalSpecProvider

            backend = LocalSpecProvider()
    elif HAVE_TE_SPEC_PROVIDER:
        backend = TESpecProvider()
    else:
        from megatron.core.models.backends import LocalSpecProvider

        backend = LocalSpecProvider()

    return backend, sensitive_backend


def sensitive_layer_range(config, num_layers: int) -> Tuple[int, int]:
    """``(num_start, num_end)`` layers to keep at the sensitive precision."""
    if not getattr(config, "sensitive_layers_enabled", False):
        return 0, 0
    num_start = getattr(config, "sensitive_layers_start", 0)
    num_end = getattr(config, "sensitive_layers_end", 0)
    if num_start + num_end > num_layers:
        raise ValueError(f"sensitive layers ({num_start} + {num_end}) exceed num_layers ({num_layers})")
    return num_start, num_end


__all__ = [
    "resolve_diffusion_backend",
    "sensitive_layer_range",
    "HAVE_TE_SPEC_PROVIDER",
    "HAVE_PRIMUS_TURBO_LOCAL",
]
