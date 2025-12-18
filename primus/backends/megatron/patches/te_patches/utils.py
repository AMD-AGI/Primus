###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine Patches Utilities

Common helper functions for TE patches.
"""


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
        kwargs = original_func(config)
        kwargs.update(overrides)
        return kwargs

    return _wrapped
