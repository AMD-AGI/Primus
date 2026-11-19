###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Low-overhead host timing of a training step (diagnostic, opt-in: PRIMUS_HOST_TIMING=1).

Records per iteration, with time.perf_counter only (no profiler):
  fwd / bwd wall (schedules.forward_step / backward_step), optimizer.step wall, rest of train_step,
  time outside train_step (logging, data, ...), and the time the host spent blocked in syncs
  (torch.cuda/Event/Stream synchronize, Tensor.item/tolist/cpu/__bool__/__float__/__int__ on GPU tensors).
CPU issue time of a phase = wall - blocked. Written as JSON lines to PRIMUS_HOST_TIMING_OUT.
PRIMUS_HOST_TIMING_SYNCDEBUG_ITER=<n>: torch.cuda.set_sync_debug_mode("warn") during iteration n; each
synchronizing aten op is recorded with its Python call site.
"""

import json
import os
import time
import traceback
import warnings

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_S = {"phase": "other", "acc": {}, "blk": {}, "nsync": {}, "it": 0, "t_end": None, "sites": {}}


def _enabled() -> bool:
    return os.environ.get("PRIMUS_HOST_TIMING", "0") == "1"


def _add(d, k, v):
    d[k] = d.get(k, 0.0) + v


def _timed_phase(name, fn):
    def wrapper(*a, **k):
        prev = _S["phase"]
        _S["phase"] = name
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _add(_S["acc"], name, time.perf_counter() - t)
            _S["phase"] = prev

    wrapper.__wrapped__ = fn
    return wrapper


def _blocking(fn, gpu_check=None):
    def wrapper(*a, **k):
        if gpu_check is not None and not gpu_check(a):
            return fn(*a, **k)
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            ph = _S["phase"]
            _add(_S["blk"], ph, time.perf_counter() - t)
            _add(_S["nsync"], ph, 1)

    wrapper.__wrapped__ = fn
    return wrapper


@register_patch(
    "megatron.host_timing",
    backend="megatron",
    phase="before_train",
    description="Diagnostic per-phase host timing (PRIMUS_HOST_TIMING=1)",
    priority=99,
    condition=lambda ctx: _enabled(),
)
def patch_host_timing(ctx: PatchContext) -> None:
    del ctx
    import torch
    import megatron.core.pipeline_parallel.schedules as sch
    import megatron.training.training as mt

    out = os.environ.get("PRIMUS_HOST_TIMING_OUT", "/tmp/host_timing.jsonl")
    dbg_iter = int(os.environ.get("PRIMUS_HOST_TIMING_SYNCDEBUG_ITER", "-1"))
    fh = open(out, "w")

    sch.forward_step = _timed_phase("fwd", sch.forward_step)
    sch.backward_step = _timed_phase("bwd", sch.backward_step)

    def is_gpu(a):
        return len(a) > 0 and getattr(a[0], "is_cuda", False)

    torch.cuda.synchronize = _blocking(torch.cuda.synchronize)
    torch.cuda.Event.synchronize = _blocking(torch.cuda.Event.synchronize)
    torch.cuda.Stream.synchronize = _blocking(torch.cuda.Stream.synchronize)
    for m in ("item", "tolist", "cpu", "__bool__", "__float__", "__int__"):
        setattr(torch.Tensor, m, _blocking(getattr(torch.Tensor, m), is_gpu))

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" not in str(message):
            return orig_show(message, category, filename, lineno, file, line)
        st = traceback.extract_stack()[:-3]
        frames = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in st[-6:]]
        key = str(message)[:60] + " @ " + " < ".join(reversed(frames))
        _S["sites"][key] = _S["sites"].get(key, 0) + 1

    orig_show = warnings.showwarning
    orig_train_step = mt.train_step

    def train_step(*args, **kwargs):
        it = _S["it"]
        t0 = time.perf_counter()
        outside = (t0 - _S["t_end"]) if _S["t_end"] is not None else 0.0
        blk_outside = _S["blk"].get("other", 0.0)
        ns_outside = _S["nsync"].get("other", 0)
        _S["acc"], _S["blk"], _S["nsync"] = {}, {}, {}
        opt = args[3] if len(args) > 3 else kwargs.get("optimizer")
        if opt is not None and not getattr(opt.step, "_ht", False):
            opt.step = _timed_phase("opt", opt.step)
            opt.step._ht = True
        if it == dbg_iter:
            warnings.simplefilter("always")
            warnings.showwarning = showwarning
            torch.cuda.set_sync_debug_mode("warn")
        _S["phase"] = "step_rest"
        try:
            return orig_train_step(*args, **kwargs)
        finally:
            t1 = time.perf_counter()
            if it == dbg_iter:
                torch.cuda.set_sync_debug_mode(0)
                warnings.showwarning = orig_show
            _S["phase"] = "other"
            acc = dict(_S["acc"])
            step = t1 - t0
            acc["step_rest"] = step - sum(acc.values())
            rec = {
                "it": it,
                "step_ms": step * 1e3,
                "outside_ms": outside * 1e3,
                "outside_blk_ms": blk_outside * 1e3,
                "outside_nsync": ns_outside,
                "wall_ms": {k: v * 1e3 for k, v in acc.items()},
                "blk_ms": {k: v * 1e3 for k, v in _S["blk"].items()},
                "nsync": dict(_S["nsync"]),
            }
            if it == dbg_iter:
                rec["sync_sites"] = _S["sites"]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            _S["acc"], _S["blk"], _S["nsync"] = {}, {}, {}
            _S["it"] = it + 1
            _S["t_end"] = time.perf_counter()

    mt.train_step = train_step
    log_rank_0(f"[Patch:megatron.host_timing] per-phase host timing -> {out}")
