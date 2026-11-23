###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Joint per-step attention-sink cast (``PRIMUS_LAUNCH_SINK_CAST=1``, ``launch_sink_cast.sink_f32``).

The first step only registers the sink parameters (no rows returned); afterwards one launch produces fp32 rows
that equal ``sink.float()`` for every registered sink, including sinks allocated more than 2 GB apart, see
parameter updates on the next step, and replay correctly from a CUDA graph.
"""

import torch

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


def test_sink_cast_rows_equal_float_cast(monkeypatch):
    monkeypatch.setenv("PRIMUS_LAUNCH_SINK_CAST", "1")
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        launch_sink_cast as SC,
    )

    torch.manual_seed(0)
    sinks, gap = [], []
    for i in range(4):
        sinks.append(torch.nn.Parameter(torch.randn(64, device=DEV).to(torch.bfloat16)))
        # push the next sink > 2 GB away from this one
        gap.append(torch.empty(int(2.2 * 2**30), dtype=torch.uint8, device=DEV) if i < 2 else None)

    def step():
        return [SC.sink_f32(s) for s in sinks]

    assert all(x is None for x in step()), "registration step must return no rows"
    r1 = step()
    assert all(x is not None and torch.equal(x, s.float()) for x, s in zip(r1, sinks))
    with torch.no_grad():
        for s in sinks:
            s.add_(1.0)
    assert all(torch.equal(x, s.float()) for x, s in zip(step(), sinks)), "updated values not seen"

    outs = [torch.empty(64, device=DEV) for _ in sinks]
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for o, x in zip(outs, step()):
                o.copy_(x * 2)
    torch.cuda.current_stream().wait_stream(st)
    with torch.no_grad():
        for s in sinks:
            s.mul_(-0.5)
    g.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(o, s.float() * 2) for o, s in zip(outs, sinks)), "graph replay differs"
    del gap
