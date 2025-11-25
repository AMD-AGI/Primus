###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-LM version detection utility.

This module provides a centralized version detection function that can be
used by both MegatronAdapter and MegatronBaseTrainer to avoid code duplication.
"""


def detect_megatron_version() -> str:
    """
    Detect Megatron-LM version using the official method.

    Returns:
        Version string (e.g., "0.15.0rc8")

    Raises:
        RuntimeError: If version cannot be detected
    """
    try:
        from megatron.core import package_info

        return package_info.__version__
    except Exception as e:
        raise RuntimeError(
            "Failed to detect Megatron-LM version. "
            "Please ensure Megatron-LM is properly installed and "
            "megatron.core.package_info is available."
        ) from e
