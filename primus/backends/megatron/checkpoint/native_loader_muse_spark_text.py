###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Converter entry point named after the HF ``text_config.model_type`` (``muse_spark_text``).

The posttrain conversion hook imports this module for the Muse Spark family. The
implementation is in ``native_loader_muse_spark``; this module only names the API.
"""

from primus.backends.megatron.checkpoint.native_loader_muse_spark import (
    MappedStore,
    MuseSparkConverter,
    build_margs,
    build_mcore_state_dict,
    build_model,
    check_convert_size,
    convert,
    derive_megatron_fields,
    vocab_size_with_padding,
)

__all__ = [
    "MuseSparkConverter",
    "MappedStore",
    "build_margs",
    "build_mcore_state_dict",
    "build_model",
    "check_convert_size",
    "convert",
    "derive_megatron_fields",
    "vocab_size_with_padding",
]
