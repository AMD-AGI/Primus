###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Build FLUX latent image position ids on the target device.

AutoModel's ``FluxAdapter._prepare_latent_image_ids`` builds the
[(H/2)*(W/2), 3] position-id tensor on the CPU and copies it to the GPU every
step. The copy is from pageable memory, so it blocks the host for a moment every
step, which shows once the train loop's own host syncs are deferred. Building the
ids on the device gives identical values with no copy.

Re-check on an AutoModel bump: once the adapter builds the ids on the device
itself, as the FLUX.2 adapter already does, this can be deleted. If the method
is missing or its signature has changed, ``install`` logs a warning and leaves
AutoModel's method in place.
"""
from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusFluxLatentIds]"
_PATCH_FLAG = "_primus_device_latent_ids"
_PARAMETERS = ("self", "batch_size", "height", "width", "device", "dtype")


def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
    import torch

    latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device)
    latent_image_ids[..., 1] = torch.arange(height // 2, device=device)[:, None]
    latent_image_ids[..., 2] = torch.arange(width // 2, device=device)[None, :]
    return latent_image_ids.reshape(-1, 3).to(dtype=dtype)


def install() -> bool:
    """Replace ``FluxAdapter._prepare_latent_image_ids``.

    Idempotent: returns True if it is already replaced. Returns False, and changes
    nothing, when the adapter is not importable or the method is not the one this
    replaces.
    """
    try:
        from nemo_automodel.components.flow_matching.adapters.flux import FluxAdapter
    except ImportError as exc:
        logger.warning("%s FluxAdapter is not importable (%s); not installed", _LOG_PREFIX, exc)
        return False

    if getattr(FluxAdapter, _PATCH_FLAG, False):
        return True
    upstream = getattr(FluxAdapter, "_prepare_latent_image_ids", None)
    if upstream is None or tuple(inspect.signature(upstream).parameters) != _PARAMETERS:
        logger.warning(
            "%s FluxAdapter._prepare_latent_image_ids is missing or has a new signature; not installed",
            _LOG_PREFIX,
        )
        return False

    _prepare_latent_image_ids.__doc__ = upstream.__doc__
    FluxAdapter._prepare_latent_image_ids = _prepare_latent_image_ids
    setattr(FluxAdapter, _PATCH_FLAG, True)
    logger.info("%s FluxAdapter latent image ids are built on device", _LOG_PREFIX)
    return True
