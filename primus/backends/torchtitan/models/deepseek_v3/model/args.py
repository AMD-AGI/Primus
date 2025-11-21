###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from dataclasses import dataclass

from torchtitan.models.deepseek_v3 import DeepSeekV3ModelArgs


# Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
@dataclass
class DeepSeekV3ClassicModelArgs(DeepSeekV3ModelArgs):
    # Classical Attention
    n_heads: int = 128
    q_head: int = 40
    n_kv_heads: int = 8
    head_dim: int = 2048 // 128
