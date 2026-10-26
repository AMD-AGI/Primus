###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Deprecated alias of :mod:`primus.backends.automodel.argument_builder`."""

from primus.backends.automodel.argument_builder import (
    PRIMUS_ONLY_TOP_KEYS,
    AutoModelConfigBuilder,
    export_params_to_yaml,
    namespace_to_dict,
    strip_primus_keys,
)

NemoAutomodelConfigBuilder = AutoModelConfigBuilder

__all__ = [
    "PRIMUS_ONLY_TOP_KEYS",
    "NemoAutomodelConfigBuilder",
    "export_params_to_yaml",
    "namespace_to_dict",
    "strip_primus_keys",
]
