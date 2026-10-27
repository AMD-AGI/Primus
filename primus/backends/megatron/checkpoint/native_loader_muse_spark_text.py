###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``muse_spark_text`` (the ``text_config.model_type`` of Muse Spark) -> the ``muse_spark`` converter."""

from primus.backends.megatron.checkpoint.native_loader_muse_spark import *  # noqa: F401,F403
from primus.backends.megatron.checkpoint.native_loader_muse_spark import (  # noqa: F401
    build_margs,
    build_mcore_state_dict,
    build_model,
    convert,
    derive_megatron_fields,
)
