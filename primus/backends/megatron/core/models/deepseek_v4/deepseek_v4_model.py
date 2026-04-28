###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 top-level model (Phase 2 stub).

Phase 2: thin subclass of ``megatron.core.models.gpt.GPTModel`` so the
dispatch path

    model_type=deepseek_v4
        -> deepseek_v4_builder
        -> DeepseekV4Model

is wired end-to-end without changing model behaviour. Phase 3 will replace
``self.decoder`` with :class:`DeepseekV4TransformerBlock` after super-init
runs, and Phase 4 will plug in HC + hybrid attention.
"""

from megatron.core.models.gpt import GPTModel


class DeepseekV4Model(GPTModel):
    """Phase-2 placeholder. Behaviour identical to ``GPTModel``."""


__all__ = ["DeepseekV4Model"]
