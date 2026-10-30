###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Build FLUX latent image position ids on the target device.

AutoModel's ``FluxAdapter._prepare_latent_image_ids`` builds the
[(H/2)*(W/2), 3] position-id tensor on the CPU and copies it to the GPU every
step. The copy is from pageable memory, so it blocks the host and adds a small
stall even once the train loop's own syncs are deferred. Building the ids on the
device gives identical values with no copy.

Applied with ``primus_train.defer_host_sync``, alongside the train-loop
deferral.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_primus_device_latent_ids"


def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
    import torch

    latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device)
    latent_image_ids[..., 1] = torch.arange(height // 2, device=device)[:, None]
    latent_image_ids[..., 2] = torch.arange(width // 2, device=device)[None, :]
    return latent_image_ids.reshape(-1, 3).to(dtype=dtype)


def install() -> bool:
    """Replace ``FluxAdapter._prepare_latent_image_ids``; True once it is in place."""
    from nemo_automodel.components.flow_matching.adapters.flux import FluxAdapter

    if getattr(FluxAdapter, _PATCH_FLAG, False):
        return True

    _prepare_latent_image_ids.__doc__ = FluxAdapter._prepare_latent_image_ids.__doc__
    FluxAdapter._prepare_latent_image_ids = _prepare_latent_image_ids
    setattr(FluxAdapter, _PATCH_FLAG, True)
    logger.info("[PrimusFluxLatentIds] FluxAdapter latent image ids are built on device")
    return True
