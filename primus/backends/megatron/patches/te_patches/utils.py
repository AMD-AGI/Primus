###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine Patches Utilities

Common helper functions for TE patches.
"""


def is_te_min_version(version: str) -> bool:
    """
    Check if Transformer Engine version meets minimum requirement.

    Args:
        version: Minimum version string (e.g., "2.0", "1.8")

    Returns:
        True if TE version >= specified version, False otherwise
    """
    try:
        from megatron.core.utils import is_te_min_version as megatron_is_te_min_version

        return megatron_is_te_min_version(version)
    except Exception:
        return False


def is_te_v2_or_above() -> bool:
    """Check if Transformer Engine version is 2.0 or above."""
    return is_te_min_version("2.0")


def is_te_below_v2() -> bool:
    """Check if Transformer Engine version is below 2.0."""
    return not is_te_min_version("2.0")


def make_get_extra_te_kwargs_with_override(original_func, **overrides):
    """
    Create a wrapped version of _get_extra_te_kwargs with custom overrides.

    This is a common pattern for TE patches that need to customize layer
    initialization parameters by temporarily overriding _get_extra_te_kwargs.

    Args:
        original_func: The original _get_extra_te_kwargs function
        **overrides: Key-value pairs to override in the returned kwargs

    Returns:
        A wrapped function that applies the overrides
    """

    def _wrapped(config):
        """Call the original function and apply the configured overrides to its kwargs."""
        kwargs = original_func(config)
        kwargs.update(overrides)
        return kwargs

    return _wrapped
