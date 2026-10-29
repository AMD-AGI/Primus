###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# SDPA replacement for Wan2.1 ``wan/modules/attention.py`` used by the CLIP
# visual tower. q/k/v are [B, L, heads, dim] and the output keeps that layout.

from __future__ import annotations

import torch.nn.functional as F


def flash_attention(q, k, v, dropout_p=0.0, causal=False, version=None, **kwargs):
    del version, kwargs
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=causal)
    return out.transpose(1, 2).to(dtype=q.dtype)
