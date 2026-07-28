###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn


@dataclass
class WanComponents:
    """
    A thin container for model components.

    Keep this intentionally simple: it's just a bundle of modules that a pipeline can use.
    """

    dit: nn.Module
    vae: nn.Module
    text_encoder: nn.Module
    image_encoder: Optional[nn.Module] = None
