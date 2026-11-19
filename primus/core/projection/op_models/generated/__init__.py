###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Op models added by the tuning agent's op bring-up stage.

Every module here is imported on first use of the op-model registry. Each one
registers a single :class:`~primus.core.projection.op_models.base.OpModel` and
passed :mod:`primus.core.projection.op_models.validate` on the config that
triggered it.
"""
