###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Patches

This package hosts Primus patch definitions for the TorchTitan backend.

Patches are registered via the ``@register_patch`` decorator from
``primus.core.patches``. Importing this package is enough to make all
patches discoverable by the global patch registry.

TorchTitan-specific code (e.g. trainers or backend adapters) should
import this package once, for example:

    import primus.backends.torchtitan  # noqa: F401

which in turn triggers this package's import and registers all patches.
"""

# For now we keep imports explicit for clarity. If the number of TorchTitan
# patches grows significantly, we can mirror the auto-discovery logic used
# by ``primus.backends.megatron.patches``.

from primus.backends.torchtitan.patches import (  # noqa: F401
    dcp_consolidate_patches,
    embedding_amp_patches,
    flex_attention_patches,
    logger_patches,
    metrics_output_format,
    mock_dataset_patches,
    model_override_patches,
    pipelining_schedule_patches,
    turbo,
    wandb_patches,
)
