###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RoPE cos / sin table in one launch (``PRIMUS_LAUNCH_ROPE_TBL=1``): ``RoPECache.forward_pos_cast`` is
bit-equal to the torch chain (shape and contiguity included) for several bases, YaRN settings, rotary dims,
fp32 / bf16 ``inv_freq`` buffers and position layouts (``[1, S]``, ``[2, S]`` with offset, random).
"""

import pytest
import torch

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
_CFGS = [
    dict(theta=10000.0),
    dict(theta=160000.0, yarn_factor=16.0, original_max_position_embeddings=65536),
    dict(theta=10000.0, yarn_factor=40.0, original_max_position_embeddings=4096),
]


@pytest.mark.parametrize("rd", [64, 128])
@pytest.mark.parametrize("cfg", _CFGS, ids=["plain", "yarn16", "yarn40"])
@pytest.mark.parametrize("idt", [torch.float32, torch.bfloat16])
def test_rope_table_one_launch_matches_torch_chain(monkeypatch, rd, cfg, idt):
    from primus.backends.megatron.core.transformer import dual_rope as DR

    rc = DR.RoPECache(rd, **cfg).to(DEV)
    rc.inv_freq = rc.inv_freq.to(idt)  # Float16Module turns the buffer into bf16
    torch.manual_seed(0)
    S = 1024
    positions = (
        torch.arange(S, device=DEV).unsqueeze(0),
        torch.arange(2 * S, device=DEV).view(2, S) + 123,
        torch.randint(0, 1 << 20, (1, S), device=DEV),
    )
    for pos in positions:
        res = []
        for on in ("0", "1"):
            monkeypatch.setenv("PRIMUS_LAUNCH_ROPE_TBL", on)
            DR._MB_POS_MEMO.clear()
            c, s = rc.forward_pos_cast(pos, torch.bfloat16)
            res.append((c.clone(), s.clone()))
        for a, b in zip(*res):
            assert a.shape == b.shape and a.is_contiguous() == b.is_contiguous()
            assert torch.equal(
                a.view(torch.int16), b.view(torch.int16)
            ), f"table differs for pos {tuple(pos.shape)}"
