###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Diagnostic (PRIMUS_GRAPH_DEBUG=<out.tsv>): per-module forward output stats for two train_step calls.

Forward hooks on every module record mean|out| and a NaN/Inf flag as device tensors (captured as
kernels when the step is graph-captured; the Python references then hold the replayed values).
After train_step call PRIMUS_GRAPH_DEBUG_ITERS (default "2,3": last eager warm-up + capture/replay
of a full_iteration graph) the stats are written as TSV: call, order, module, mean_abs, nonfinite.
"""

import os

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _out_tensor(o):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (tuple, list)):
        for x in o:
            t = _out_tensor(x)
            if t is not None:
                return t
    if isinstance(o, dict):
        for x in o.values():
            t = _out_tensor(x)
            if t is not None:
                return t
    return None


@register_patch(
    "megatron.graph_debug",
    backend="megatron",
    phase="before_train",
    description="Per-module forward stats for graph-vs-eager debugging (PRIMUS_GRAPH_DEBUG)",
    priority=98,
    condition=lambda ctx: bool(os.environ.get("PRIMUS_GRAPH_DEBUG")),
)
def patch_graph_debug(ctx: PatchContext) -> None:
    del ctx
    import megatron.training.training as mt

    out = os.environ["PRIMUS_GRAPH_DEBUG"]
    iters = {int(x) for x in os.environ.get("PRIMUS_GRAPH_DEBUG_ITERS", "2,3").split(",")}
    st = {"call": 0, "rec": None, "hooked": False}
    fh = open(out, "w")
    orig = mt.train_step

    def hook(name):
        def f(mod, inp, outp):
            rec = st["rec"]
            if rec is None:
                return
            t = _out_tensor(outp)
            if t is None or not t.is_floating_point() or t.numel() == 0:
                return
            tf = t.detach().float()
            rec.append((name, tf.abs().mean(), (~torch.isfinite(tf)).any()))
        return f

    def train_step(*args, **kwargs):
        model = args[2] if len(args) > 2 else kwargs["model"]
        if not st["hooked"]:
            for n, m in model[0].named_modules():
                m.register_forward_hook(hook(n or "<root>"))
            st["hooked"] = True
        c = st["call"]
        st["rec"] = [] if c in iters else None
        try:
            return orig(*args, **kwargs)
        finally:
            rec, st["rec"] = st["rec"], None
            if rec is not None:
                torch.cuda.synchronize()
                for i, (n, m, bad) in enumerate(rec):
                    fh.write(f"{c}\t{i}\t{n}\t{m.item():.6e}\t{int(bad.item())}\n")
                fh.flush()
            st["call"] = c + 1

    mt.train_step = train_step
    log_rank_0(f"[Patch:megatron.graph_debug] per-module forward stats for calls {sorted(iters)} -> {out}")
