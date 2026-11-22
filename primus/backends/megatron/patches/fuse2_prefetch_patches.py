###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_FUSE2_PREFETCH=1 (fuse2 line, default off): the next step's batch is fetched and copied to the GPU while
the current step's CUDA graph runs.

With ``--cuda_graph_impl full_iteration`` Megatron's ``FullCudaGraphWrapper.data_read`` calls ``next(iterator)``
and copies every batch tensor host -> device into the graph's static input buffers right before the replay, on the
critical path between two steps (on MI455X ~0.25-0.3 ms of GPU idle per step: 2-3 32 KB HtoD copies at 40-150 us
each, plus the dataloader ``next``). Here, right after a training replay is launched, the next ``num_microbatches``
batches are taken from the same iterator, pinned, and copied into device staging buffers on a side stream; the next
``data_read`` only waits for that copy and moves staging -> static buffers device to device. Same batches in the
same order (bit-identical training). Falls back to Megatron's ``data_read`` for validation, virtual pipeline /
multi-model lists, a different iterator object, or an exhausted iterator.
"""

from __future__ import annotations

import os

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_ENABLED = os.environ.get("PRIMUS_FUSE2_PREFETCH", "0") == "1"


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

    def _prefetch(self, it, n):
        st = self._f2_stream
        # staging buffers are reused: the previous staging -> static copies (loader stream) must be done. Do NOT
        # wait on the current stream: it holds the replay just launched, and the copy must overlap it.
        st.wait_stream(self.static_loader.stream)
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
            self._f2_pf, self._f2_off = None, True
            return
        ev = torch.cuda.Event()
        ev.record(st)
        staged_list = [h for h in hosts if isinstance(h, dict)]
        self._f2_pf = (it, n, staged_list, hosts, ev)

    def _rerun_disabled():
        try:
            from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine

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
            raise RuntimeError("PRIMUS_FUSE2_PREFETCH: prefetched batches not consumed by the next training step")
        self._f2_pf = None
        torch.cuda.current_stream().wait_event(pf[4])
        self._f2_hold = pf  # pinned host copies stay referenced until the next prefetch
        data_list = [self.static_loader(pf[2][b], "training", b) for b in range(num_microbatches)]
        return [iter(data_list)]

    def __call__(self, *args, **kwargs):
        res = orig_call(self, *args, **kwargs)
        if not kwargs.get("forward_only", False) and not getattr(self, "_f2_off", False) and _rerun_disabled():
            it = _iter0(kwargs.get("data_iterator"), kwargs.get("model"))
            if it is not None:
                if not hasattr(self, "_f2_stream"):
                    self._f2_stream = torch.cuda.Stream()
                    self._f2_bufs = []
                _prefetch(self, it, int(kwargs["num_microbatches"]))
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
    log_rank_0("[Patch:megatron.fuse2.batch_prefetch] next-batch prefetch during the graph replay (PRIMUS_FUSE2_PREFETCH=1)")
