###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the deferred host-sync train loop and the on-device FLUX latent ids.

Both are gated on ``primus_train.defer_host_sync``; the latent ids apply
only to runs that train FLUX.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

options.declare("primus_train", "defer_host_sync")

_is_flux = transformer_is("FluxTransformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.training import deferred_host_sync

    return deferred_host_sync.is_enabled()


@register_patch(
    "automodel.training.deferred_host_sync",
    backend="automodel",
    phase="before_train",
    description="Defer train-loop loss/grad-norm host syncs to logging steps (primus_train.defer_host_sync)",
    condition=_enabled,
    # Rewrites the loop's source, so it must run before the profiler wraps the loop (90).
    priority=60,
)
def apply_train_loop(ctx: PatchContext) -> None:
    from primus.backends.automodel.training import deferred_host_sync

    deferred_host_sync.install()


@register_patch(
    "automodel.models.flux.latent_ids",
    backend="automodel",
    phase="before_train",
    description="Build FLUX latent image ids on device (primus_train.defer_host_sync)",
    condition=lambda ctx: _is_flux(ctx) and _enabled(ctx),
    priority=60,
)
def apply_flux_latent_ids(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import latent_ids

    latent_ids.install()
