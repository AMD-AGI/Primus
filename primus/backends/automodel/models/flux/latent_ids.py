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
itself, as the FLUX.2 adapter already does, this can be deleted. ``install``
replaces the method only while its parameters and body are those of the pinned
AutoModel's, kept here as ``_pinned_upstream``; otherwise it logs a warning and
leaves AutoModel's method in place.
"""
from __future__ import annotations

import ast
import inspect
import logging
import textwrap

import torch

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusFluxLatentIds]"
_PATCH_FLAG = "_primus_device_latent_ids"


def _pinned_upstream(self, batch_size, height, width, device, dtype):
    """AutoModel's method at the pinned commit, apart from its docstring: what ``install`` replaces."""
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = torch.arange(width // 2)[None, :]

    latent_image_ids = latent_image_ids.reshape(-1, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
    """Position ids [(H//2)*(W//2), 3] of (0, row, column), built on ``device``."""
    latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device)
    latent_image_ids[..., 1] = torch.arange(height // 2, device=device)[:, None]
    latent_image_ids[..., 2] = torch.arange(width // 2, device=device)[None, :]
    return latent_image_ids.reshape(-1, 3).to(dtype=dtype)


def _body(method):
    """The statements of ``method`` after its docstring, as AST dumps; None if its source is unreadable."""
    try:
        function = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    except (OSError, TypeError, SyntaxError):
        return None
    statements = function.body
    if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        statements = statements[1:]
    return [ast.dump(statement) for statement in statements]


def _is_pinned_upstream(method) -> bool:
    parameters = tuple(inspect.signature(method).parameters)
    pinned_parameters = tuple(inspect.signature(_pinned_upstream).parameters)
    return parameters == pinned_parameters and _body(method) == _body(_pinned_upstream)


def install() -> bool:
    """Replace ``FluxAdapter._prepare_latent_image_ids``.

    Idempotent: returns True if it is already replaced. Returns False, and changes
    nothing, when the adapter is not importable or its method is not the pinned
    one this replaces.
    """
    try:
        from nemo_automodel.components.flow_matching.adapters.flux import FluxAdapter
    except ImportError as exc:
        logger.warning("%s FluxAdapter is not importable (%s); not installed", _LOG_PREFIX, exc)
        return False

    if getattr(FluxAdapter, _PATCH_FLAG, False):
        return True
    upstream = getattr(FluxAdapter, "_prepare_latent_image_ids", None)
    if upstream is None or not _is_pinned_upstream(upstream):
        logger.warning(
            "%s FluxAdapter._prepare_latent_image_ids is missing or not the pinned AutoModel's; not installed",
            _LOG_PREFIX,
        )
        return False

    FluxAdapter._prepare_latent_image_ids = _prepare_latent_image_ids
    setattr(FluxAdapter, _PATCH_FLAG, True)
    logger.info("%s FluxAdapter latent image ids are built on device", _LOG_PREFIX)
    return True
