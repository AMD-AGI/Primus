###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Adapted from Tencent HunyuanVideo-1.5 / HY-WorldPlay official implementation.

"""ByT5 output mapper used by the WorldPlay transformer."""

import torch.nn as nn


class ByT5Mapper(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, out_dim1, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x2 = self.act_fn(x)
        x2 = self.fc3(x2)
        if self.use_residual:
            x2 = x2 + residual
        return x2
