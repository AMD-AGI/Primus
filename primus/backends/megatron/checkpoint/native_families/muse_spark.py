###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark (``muse_spark`` / ``muse_spark_text``) in the native converter's registry.

The converter is ``native_loader_muse_spark_text`` (the text decoder). Its
hook-side steps are in ``native_loader_muse_spark`` and are imported only when a Muse
Spark checkpoint converts:

* ``native_ckpt_max_params`` -> ``convert(max_params=...)``: an optional cap on the
  converter's estimated parameter count (unset: no cap; the converter's host-memory check
  always runs).
* ``dtype_from_config``: text_config's ``dtype`` / ``torch_dtype``, else the top-level
  ``dtype`` (transformers >= 4.56 writes ``dtype``, not ``torch_dtype``).
* ``prepare``: refuses a top-level ``torch_dtype`` that differs from text_config's unless
  ``native_ckpt_dtype`` is set; records node 0's start for the waiting nodes.
* ``wait``: the other nodes follow node 0's conversion heartbeat (hours at full size)
  and stop as soon as it fails.
"""

from primus.backends.megatron.checkpoint.native_registry import (
    NativeFamily,
    NativeOption,
    register_native_family,
)

_LOADER = "primus.backends.megatron.checkpoint.native_loader_muse_spark"

register_native_family(
    NativeFamily(
        name="muse_spark",
        label="Muse Spark",
        converter="primus.backends.megatron.checkpoint.native_loader_muse_spark_text",
        architectures=("MuseSparkForConditionalGeneration",),
        model_types=("muse_spark", "muse_spark_text"),
        model_type="muse_spark",
        options=(
            NativeOption(
                key="native_ckpt_max_params",
                kwarg="max_params",
                help="optional cap on the estimated parameter count (default: none)",
            ),
        ),
        dtype_from_config=f"{_LOADER}:hook_dtype_from_config",
        prepare=f"{_LOADER}:hook_prepare",
        wait=f"{_LOADER}:hook_wait",
    )
)
