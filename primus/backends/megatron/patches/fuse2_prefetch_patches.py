###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_FUSE2_PREFETCH=1 : the next step's batch is fetched and copied to the GPU while
the current step's CUDA graph runs.

With ``--cuda_graph_impl full_iteration`` Megatron's ``FullCudaGraphWrapper.data_read`` calls ``next(iterator)``
and copies every batch tensor host -> device into the graph's static input buffers right before the replay, on the
critical path between two steps (on MI455X ~0.25-0.3 ms of GPU idle per step: 2-3 32 KB HtoD copies at 40-150 us
each, plus the dataloader ``next``). Here, right after a training replay is launched, the next ``num_microbatches``
batches are taken from the same iterator, pinned, and copied into device staging buffers on a side stream; the next
``data_read`` only waits for that copy and moves staging -> static buffers device to device. Same batches in the
same order (bit-identical training). PRIMUS_FUSE2_PREFETCH=2 also moves the staging -> static copies off the step
boundary: they run on the loader stream right after the replay that last reads the static buffers (overlapping the
optimizer step), so the next replay starts without any copy. Falls back to Megatron's ``data_read`` for validation, virtual pipeline /
multi-model lists, a different iterator object, or an exhausted iterator.
"""

from __future__ import annotations

import os

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_MODE = int(os.environ.get("PRIMUS_FUSE2_PREFETCH", "0") or 0)
_ENABLED = _MODE in (1, 2)
_PACK = os.environ.get("PRIMUS_FUSE2_PREFETCH_PACK", "1") == "1"


def _iter0(data_iterator, model):
    if isinstance(model, list) and len(model) != 1:
        return None
    if isinstance(data_iterator, list):
        if len(data_iterator) != 1:
            return None
        data_iterator = data_iterator[0]
    return data_iterator if hasattr(data_iterator, "__next__") else None


def install() -> None:
    import megatron.core.full_cuda_graph as fcg

    W = fcg.FullCudaGraphWrapper
    if getattr(W, "_primus_fuse2_prefetch", False):
        return
    orig_read = W.data_read
    orig_call = W.__call__

    def _prefetch(self, it, n, done):
        st = self._f2_stream
        # staging buffers are reused: the previous staging -> static copies (loader stream) must be done. Do NOT
        # wait on the current stream: it holds the replay just launched, and the copy must overlap it.
        st.wait_stream(self.static_loader.stream)
        if _PACK:
            try:
                staged_list, hosts = _prefetch_packed(self, it, n, st)
            except StopIteration:
                self._f2_pf, self._f2_off = None, True
                return
        else:
            staged_list, hosts = None, None
        if staged_list is None:
            staged_list, hosts = _prefetch_per_tensor(self, it, n, st)
            if staged_list is None:
                self._f2_pf, self._f2_off = None, True
                return
        _finish_prefetch(self, it, n, done, st, staged_list, hosts)

    def _prefetch_packed(self, it, n, st):
        """PRIMUS_FUSE2_PREFETCH_PACK (default 1): the n batches' host tensors are packed (CPU memcpy) into one
        pinned buffer and moved with ONE host -> device copy into one device staging buffer (views per tensor).
        The per-tensor path issues ~4 small HtoD copies per micro-batch (~70 us each on MI455X); when the copy
        stream shares a hardware queue with the main stream they run back to back after the replay and delay the
        optimizer step (~0.8 ms/step at 3 micro-batches). Same bytes."""
        batches = []
        for _ in range(n):
            h = next(it)
            if isinstance(h, tuple) and len(h) > 0 and isinstance(h[0], dict):
                h = h[0]
            if not isinstance(h, dict):
                raise TypeError(f"unexpected batch type {type(h)}")
            batches.append(h)
        layout, off = [], 0
        for h in batches:
            for k, v in h.items():
                if isinstance(v, torch.Tensor):
                    if v.device.type != "cpu":
                        return _stage_given(self, batches, st)
                    nbytes = v.numel() * v.element_size()
                    layout.append((k, off, nbytes, v))
                    off += (nbytes + 255) // 256 * 256
        total = max(off, 256)
        pk = getattr(self, "_f2_pack", None)
        if pk is None or pk["total"] != total:
            with torch.cuda.stream(st):
                dev = torch.empty(total, dtype=torch.uint8, device="cuda")
            pk = self._f2_pack = {
                "total": total,
                "dev": dev,
                "host": [torch.empty(total, dtype=torch.uint8, pin_memory=True) for _ in range(2)],
                "ev": [None, None],
                "i": 0,
            }
        i = pk["i"]
        pk["i"] ^= 1
        if pk["ev"][i] is not None:
            pk["ev"][i].synchronize()  # the H2D that last read this pinned buffer (two steps ago) is done
        host = pk["host"][i]
        for k, o, nb, v in layout:
            host[o : o + nb].view(v.dtype).view(v.shape).copy_(v)
        dev = pk["dev"]
        with torch.cuda.stream(st):
            dev.copy_(host, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(st)
        pk["ev"][i] = ev
        staged_list, j = [], 0
        for h in batches:
            staged = {}
            for k, v in h.items():
                if isinstance(v, torch.Tensor):
                    _, o, nb, _ = layout[j]
                    j += 1
                    staged[k] = dev[o : o + nb].view(v.dtype).view(v.shape)
                else:
                    staged[k] = v
            staged_list.append(staged)
        return staged_list, [host] + staged_list

    def _stage_given(self, batches, st):
        it = iter(batches)
        return _prefetch_per_tensor(self, it, len(batches), st)

    def _prefetch_per_tensor(self, it, n, st):
        bufs = self._f2_bufs
        hosts = []
        try:
            for b in range(n):
                h = next(it)
                if isinstance(h, tuple) and len(h) > 0 and isinstance(h[0], dict):
                    h = h[0]
                if not isinstance(h, dict):
                    raise TypeError(f"unexpected batch type {type(h)}")
                if b >= len(bufs):
                    bufs.append({})
                slot = bufs[b]
                staged = {}
                for k, v in h.items():
                    if isinstance(v, torch.Tensor):
                        buf = slot.get(k)
                        with torch.cuda.stream(st):
                            if buf is None or buf.shape != v.shape or buf.dtype != v.dtype:
                                buf = torch.empty(v.shape, dtype=v.dtype, device="cuda")
                                slot[k] = buf
                            src = v
                            if v.device.type == "cpu" and not v.is_pinned():
                                src = v.pin_memory()
                            buf.copy_(src, non_blocking=True)
                            hosts.append(src)  # keep the (pinned) host tensor alive until the copy is done
                        staged[k] = buf
                    else:
                        staged[k] = v
                hosts.append(staged)
        except StopIteration:
            return None, None
        staged_list = [h for h in hosts if isinstance(h, dict)]
        return staged_list, hosts

    def _finish_prefetch(self, it, n, done, st, staged_list, hosts):
        ev = torch.cuda.Event()
        ev.record(st)
        landed = False
        sb = fcg.StaticBufferLoader.static_buffers["training"]
        if (
            _MODE == 2
            and len(sb) >= n
            and all(
                isinstance(sb[b].get(k), torch.Tensor)
                and sb[b][k].shape == v.shape
                and sb[b][k].dtype == v.dtype
                for b in range(n)
                for k, v in staged_list[b].items()
                if isinstance(v, torch.Tensor)
            )
        ):
            # PRIMUS_FUSE2_PREFETCH=2: the replay just launched is the last reader of the static buffers; copy the
            # staged batch into them right after it (overlapping the optimizer step), not before the next replay.
            ls = self.static_loader.stream
            ls.wait_event(done)
            ls.wait_event(ev)
            with torch.cuda.stream(ls):
                for b in range(n):
                    for k, v in staged_list[b].items():
                        if isinstance(v, torch.Tensor):
                            sb[b][k].copy_(v, non_blocking=True)
            landed = True
        self._f2_pf = (it, n, staged_list, hosts, ev, landed)

    def _rerun_disabled():
        try:
            from megatron.core.rerun_state_machine import (
                RerunMode,
                get_rerun_state_machine,
            )

            return get_rerun_state_machine().mode == RerunMode.DISABLED
        except Exception:  # noqa: BLE001
            return False

    def data_read(self, data_iterator, model, training, num_microbatches):
        pf = getattr(self, "_f2_pf", None)
        if not training or pf is None:
            return orig_read(self, data_iterator, model, training, num_microbatches)
        it = _iter0(data_iterator, model)
        if it is None or pf[0] is not it or pf[1] != num_microbatches:
            # the prefetched batches would be skipped (data order would change): refuse
            self._f2_pf, self._f2_off = None, True
            raise RuntimeError(
                "PRIMUS_FUSE2_PREFETCH: prefetched batches not consumed by the next training step"
            )
        self._f2_pf = None
        self._f2_hold = pf  # pinned host copies stay referenced until the next prefetch
        if pf[5]:  # already copied into the static buffers after the previous replay (mode 2)
            sb = fcg.StaticBufferLoader.static_buffers["training"]
            for b in range(num_microbatches):
                for k, v in pf[2][b].items():
                    if not isinstance(v, torch.Tensor):
                        sb[b][k] = v
            torch.cuda.current_stream().wait_stream(self.static_loader.stream)
            return [iter([sb[b] for b in range(num_microbatches)])]
        torch.cuda.current_stream().wait_event(pf[4])
        data_list = [self.static_loader(pf[2][b], "training", b) for b in range(num_microbatches)]
        return [iter(data_list)]

    def __call__(self, *args, **kwargs):
        res = orig_call(self, *args, **kwargs)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream())  # after the replay / eager fwd-bwd just enqueued
        if (
            not kwargs.get("forward_only", False)
            and not getattr(self, "_f2_off", False)
            and _rerun_disabled()
        ):
            it = _iter0(kwargs.get("data_iterator"), kwargs.get("model"))
            if it is not None:
                if not hasattr(self, "_f2_stream"):
                    self._f2_stream = torch.cuda.Stream()
                    self._f2_bufs = []
                _prefetch(self, it, int(kwargs["num_microbatches"]), done)
        return res

    W.data_read = data_read
    W.__call__ = __call__
    W._primus_fuse2_prefetch = True


@register_patch(
    "megatron.fuse2.batch_prefetch",
    backend="megatron",
    phase="before_train",
    description="Full-iteration CUDA graph: next batch fetched + copied to the GPU during the current step "
    "(PRIMUS_FUSE2_PREFETCH=1).",
    condition=lambda ctx: _ENABLED,
    priority=61,
)
def patch_fuse2_prefetch(ctx: PatchContext) -> None:
    install()
    log_rank_0(
        "[Patch:megatron.fuse2.batch_prefetch] next-batch prefetch during the graph replay (PRIMUS_FUSE2_PREFETCH=1)"
    )
