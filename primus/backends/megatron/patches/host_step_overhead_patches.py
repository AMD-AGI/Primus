###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Host-side step overhead trims (no numerical effect).

1. ``megatron.core.timers.Timer.start/stop(barrier=True)`` with a single rank:
   the ``torch.distributed.barrier`` has nobody to wait for, and the timer calls
   ``torch.cuda.synchronize()`` right after it anyway. Skip only the barrier.
   The ``interval-time`` timer does this twice per logged step.

2. Main-thread CPU pinning (``PRIMUS_MAIN_THREAD_CPU``, off by default).
   The forward is host-bound on the Python main thread. perf shows that thread
   about 70% frontend-bound (around 120 L1i misses per 1000 instructions) while it
   launches the forward. Pinning it to one core on the GPU's NUMA node, and moving
   every other thread of the process off that core and its SMT sibling, keeps its
   code caches warm across steps. Pinning happens at the start of the second
   ``train_step``, after the autograd worker thread exists (threads inherit the
   creating thread's mask). Values: ``auto`` (a core on the GPU's NUMA node) or a
   CPU id. Other threads keep the GPU's NUMA node minus that core.
"""

from __future__ import annotations

import os
import threading

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.core.timers.single_rank_no_barrier",
    backend="megatron",
    phase="before_train",
    description="Skip the timer barrier when the world size is 1 (the timer still synchronizes the device).",
)
def patch_timer_single_rank_barrier(ctx: PatchContext) -> None:
    import torch
    from megatron.core import timers as mt  # type: ignore

    Timer = mt.Timer
    if getattr(Timer, "_primus_single_rank_barrier", False):
        return
    orig_start, orig_stop = Timer.start, Timer.stop

    def _no_barrier(barrier):
        return (
            barrier and torch.distributed.is_initialized() and torch.distributed.get_world_size() == 1
        )

    def start(self, barrier=False):
        return orig_start(self, barrier=False if _no_barrier(barrier) else barrier)

    def stop(self, barrier=False):
        return orig_stop(self, barrier=False if _no_barrier(barrier) else barrier)

    Timer.start, Timer.stop = start, stop
    Timer._primus_single_rank_barrier = True


def _parse_cpulist(text: str) -> list:
    cpus = []
    for part in text.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        cpus.extend(range(int(a), int(b or a) + 1))
    return cpus


def _gpu_numa_cpus():
    import torch

    try:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        node = int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read())
        if node >= 0:
            return _parse_cpulist(open(f"/sys/devices/system/node/node{node}/cpulist").read())
    except Exception:
        pass
    return None


def _pin_main_thread(spec: str) -> str:
    allowed = set(os.sched_getaffinity(0))
    node_cpus = _gpu_numa_cpus()
    pool = [c for c in (node_cpus or sorted(allowed)) if c in allowed]
    if spec == "auto":
        # A core away from CPU 0 (interrupts, housekeeping), first SMT thread of it.
        firsts = []
        for c in pool:
            sib = _parse_cpulist(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read())
            if c == min(sib):
                firsts.append(c)
        if len(firsts) < 4:
            return "skipped (too few cores)"
        cpu = firsts[min(8, len(firsts) - 1)]
    else:
        cpu = int(spec)
    sib = set(_parse_cpulist(open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read()))
    others = set(pool) - sib
    if cpu not in allowed or not others:
        return f"skipped (cpu {cpu} not usable)"
    me = threading.get_native_id()
    n = 0
    for t in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(t), {cpu} if int(t) == me else others)
            n += 1
        except OSError:
            pass
    return f"main thread -> cpu {cpu}; {n} threads; others on {len(others)} cpus"


@register_patch(
    "megatron.training.main_thread_cpu_pinning",
    backend="megatron",
    phase="before_train",
    description="Pin the Python main thread to its own core on the GPU's NUMA node (PRIMUS_MAIN_THREAD_CPU).",
    condition=lambda ctx: bool(os.environ.get("PRIMUS_MAIN_THREAD_CPU")),
)
def patch_main_thread_cpu_pinning(ctx: PatchContext) -> None:
    import megatron.training.training as training  # type: ignore

    original_train_step = training.train_step
    if getattr(original_train_step, "_primus_main_thread_pin", False):
        return
    state = {"n": 0}
    spec = os.environ["PRIMUS_MAIN_THREAD_CPU"].strip().lower()

    def train_step(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 2:
            try:
                msg = _pin_main_thread(spec)
            except Exception as e:  # pinning is an optimisation only
                msg = f"skipped ({e!r})"
            log_rank_0(f"[Patch:megatron.main_thread_cpu_pinning] {msg}")
        return original_train_step(*args, **kwargs)

    train_step._primus_main_thread_pin = True
    training.train_step = train_step
