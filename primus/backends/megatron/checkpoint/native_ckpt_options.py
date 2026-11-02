###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native-conversion config options that the conversion hook validates up front.

This file imports only the standard library and ``primus.core.utils.number_utils``
(stdlib only): the posttrain conversion hook loads it by file path (no torch, no
Megatron) so that a bad value stops every rank before any converter runs, with the
same rule the converter applies.
"""

from primus.core.utils.number_utils import parse_positive_int

# A single process builds the whole model on CPU and a second full state dict; above
# this the host RAM need is in the TB range (the released 816B model): refuse it.
MAX_PARAMS_KEY = "native_ckpt_max_params"  # sft_trainer / post_trainer config key (hook option)
DEFAULT_MAX_PARAMS = 60 * 10**9


def parse_max_params(value) -> int:
    """The conversion cap from the ``native_ckpt_max_params`` config value.

    None or empty -> ``DEFAULT_MAX_PARAMS``. A positive integer count; a string may use ``_``
    separators, surrounding spaces and an integral float form (``60e9``). Anything else
    (a bool, a fraction, NaN, <= 0) raises a ValueError that names the key.
    """
    return parse_positive_int(
        value,
        DEFAULT_MAX_PARAMS,
        f"{MAX_PARAMS_KEY}={value!r} is not a positive integer parameter count (e.g. 60000000000 or 60e9)",
    )
