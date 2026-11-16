###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Recommendation-model backend (first model: DLRMv4, HSTU on yambda-5b)."""

from primus.backends.recommendation.recommendation_adapter import RecommendationAdapter
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_adapter("recommendation", RecommendationAdapter)
