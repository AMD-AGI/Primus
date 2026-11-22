# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""launch line, PRIMUS_LAUNCH_SINK_CAST=1 (default off): the fp32 copies of all layers' attention sinks in ONE launch.

The sparse-MLA forward needs ``sink.float()`` per layer (8 tiny cast kernels per step). Sinks only change in the
optimizer step, so the first attention forward of a step casts every registered sink (absolute-address table, see the
>2 GB pointer-offset trap) into one fp32 buffer and every layer uses its row. Same values (bf16 -> fp32 is exact).
Registry order = first-seen order; the first registered sink triggers the joint cast, so a layer seen for the first
time (warm-up step) falls back to its own cast and registers itself.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def enabled() -> bool:
    return os.environ.get("PRIMUS_LAUNCH_SINK_CAST", "0") == "1"


@triton.jit
def _sink_cast_all_kernel(TABLE, OUT, H: tl.constexpr):
    i = tl.program_id(0)
    ptr = tl.load(TABLE + i).to(tl.pointer_type(tl.bfloat16))
    h = tl.arange(0, H)
    tl.store(OUT + i * H + h, tl.load(ptr + h).to(tl.float32))


class _Reg:
    def __init__(self):
        self.params = []  # in first-seen order
        self.index = {}  # id(param) -> row
        self.out = None
        self.table = None
        self.H = None


_REG = {}


def sink_f32(sink: torch.Tensor):
    """fp32 [H] copy of ``sink`` (a row of the per-step joint buffer), or None when not applicable."""
    if not enabled() or not sink.is_cuda or sink.dtype != torch.bfloat16 or sink.dim() != 1 or not sink.is_contiguous():
        return None
    H = sink.numel()
    if H & (H - 1) or H > 1024:
        return None
    reg = _REG.setdefault((sink.device, H), _Reg())
    row = reg.index.get(id(sink))
    if row is None or reg.params[row] is not sink or sink.data_ptr() != reg.ptrs[row]:
        # new sink (warm-up step): own cast, then register it (table rebuilt eagerly, outside any capture)
        if torch.cuda.is_current_stream_capturing():
            return None
        if row is not None:  # a registered sink moved or its id was reused: start over
            reg.params, reg.index = [], {}
        reg.params.append(sink)
        reg.index[id(sink)] = len(reg.params) - 1
        reg.ptrs = [p.data_ptr() for p in reg.params]
        reg.table = torch.tensor(reg.ptrs, dtype=torch.int64, device=sink.device)
        reg.out = torch.empty((len(reg.params), H), dtype=torch.float32, device=sink.device)
        return None
    if row == 0:  # first attention forward of the step: all sinks at once
        n = len(reg.params)
        _sink_cast_all_kernel[(n,)](reg.table, reg.out, H=H, num_warps=1)
    return reg.out[row]
