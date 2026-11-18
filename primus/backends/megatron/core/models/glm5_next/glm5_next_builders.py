###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (``glm5_next``) builder names; ``MODEL_FAMILIES["glm5_next"]`` in ``registry.py``
declares the build, which needs no family hooks."""

from functools import partial

from primus.backends.megatron.core.models.builder import (
    build_model,
    family_model_provider,
)

__all__ = ["glm5_next_builder", "model_provider"]

glm5_next_builder = partial(build_model, "glm5_next")
model_provider = family_model_provider(glm5_next_builder)
