###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Mock HF Dataset Patch

This patch mirrors ``patch_mock_hf_dataset`` from
``primus.modules.trainer.torchtitan.patch_utils`` using the generic Primus
patch system, so that HF dataset mocking can be enabled via config without
tightly coupling it to the trainer implementation.
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.modules.trainer.torchtitan.patch_utils import patch_mock_hf_dataset


@register_patch(
    "torchtitan.training.mock_hf_dataset",
    backend="torchtitan",
    phase="setup",
    description="Enable mock HuggingFace dataset mode for TorchTitan",
    condition=lambda ctx: get_param(ctx, "training.mock_data", False),
)
def patch_torchtitan_mock_hf_dataset(ctx: PatchContext) -> None:  # noqa: ARG001
    """
    Patch HF datasets.load_dataset with a lightweight mock implementation.

    Delegates to ``patch_mock_hf_dataset`` to keep behavior identical to
    the trainer-side implementation.
    """
    patch_mock_hf_dataset()
