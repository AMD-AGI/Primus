###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (``glm5_next``) model package: KDA + DSA hybrid with mHC."""

from primus.backends.megatron.core.models.glm5_next.glm5_next_block import (
    Glm5NextLayer,
    Glm5NextTransformerBlock,
)
from primus.backends.megatron.core.models.glm5_next.glm5_next_builders import (
    glm5_next_builder,
    model_provider,
)
from primus.backends.megatron.core.models.glm5_next.glm5_next_model import Glm5NextModel

__all__ = [
    "Glm5NextLayer",
    "Glm5NextModel",
    "Glm5NextTransformerBlock",
    "glm5_next_builder",
    "model_provider",
]
