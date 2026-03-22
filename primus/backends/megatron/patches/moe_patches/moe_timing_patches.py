###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MoE A2A Timing Patch

Instruments MoELayer's dispatch and combine phases with CUDA event timers to
measure actual A2A communication time, expert GEMM time, and idle time on the
compute stream. This enables quantifying the benefit of overlap_moe_expert_parallel_comm.

Usage:
  Enable via --moe_timing_enabled True in the training script.

Output (per-step, rank-0 only):
  [MoETiming] step=N layer=L  dispatch_ms=X.X  expert_ms=Y.Y  combine_ms=Z.Z  total_ms=W.W

After training, call MoETimingStats.summary() to get aggregate statistics.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Optional

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


class _LayerTimer:
    """Per-layer CUDA event timer for one MoE forward pass."""

    def __init__(self):
        self.dispatch_start = torch.cuda.Event(enable_timing=True)
        self.dispatch_end = torch.cuda.Event(enable_timing=True)
        self.expert_start = torch.cuda.Event(enable_timing=True)
        self.expert_end = torch.cuda.Event(enable_timing=True)
        self.combine_start = torch.cuda.Event(enable_timing=True)
        self.combine_end = torch.cuda.Event(enable_timing=True)


class MoETimingStats:
    """Thread-safe accumulator for MoE timing statistics."""

    _lock = threading.Lock()
    _data: dict[int, list[dict]] = defaultdict(list)   # layer_number → [{...}, ...]
    _step: int = 0

    @classmethod
    def reset_step(cls):
        with cls._lock:
            cls._step += 1
            cls._data.clear()

    @classmethod
    def record(cls, layer_number: int, dispatch_ms: float, expert_ms: float, combine_ms: float):
        with cls._lock:
            cls._data[layer_number].append({
                "step": cls._step,
                "dispatch_ms": dispatch_ms,
                "expert_ms": expert_ms,
                "combine_ms": combine_ms,
            })

    @classmethod
    def log_summary(cls, step: int):
        """Log per-layer averages for the current step."""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() != 0:
            return
        with cls._lock:
            for layer_num in sorted(cls._data.keys()):
                records = cls._data[layer_num]
                if not records:
                    continue
                avg_dispatch = sum(r["dispatch_ms"] for r in records) / len(records)
                avg_expert = sum(r["expert_ms"] for r in records) / len(records)
                avg_combine = sum(r["combine_ms"] for r in records) / len(records)
                total = avg_dispatch + avg_expert + avg_combine
                serial_comm = avg_dispatch + avg_combine
                # Theoretical gain if A2A is fully hidden behind expert GEMM
                overlap_potential = max(0.0, serial_comm - avg_expert)
                pct = (overlap_potential / total * 100) if total > 0 else 0
                log_rank_0(
                    f"[MoETiming] step={step:4d} layer={layer_num:3d} "
                    f"dispatch={avg_dispatch:6.2f}ms  expert={avg_expert:6.2f}ms  "
                    f"combine={avg_combine:6.2f}ms  total={total:6.2f}ms  "
                    f"overlap_potential={overlap_potential:.2f}ms ({pct:.1f}%)"
                )


def _make_timed_moe_forward(original_forward, layer_number: int):
    """Wrap MoELayer.forward to insert CUDA event timers around dispatch/combine."""

    def timed_forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        timer = _LayerTimer()

        # We need to intercept dispatch and combine within the custom_forward closure.
        # Save original methods and wrap them.
        orig_dispatch = self.dispatch
        orig_routed_experts_compute = self.routed_experts_compute
        orig_combine = self.combine

        def timed_dispatch(hidden_states_d, probs_d):
            timer.dispatch_start.record()
            result = orig_dispatch(hidden_states_d, probs_d)
            timer.dispatch_end.record()
            return result

        def timed_routed_experts_compute(dispatched_input, probs_d):
            timer.expert_start.record()
            result = orig_routed_experts_compute(dispatched_input, probs_d)
            timer.expert_end.record()
            return result

        def timed_combine(output_d):
            timer.combine_start.record()
            result = orig_combine(output_d)
            timer.combine_end.record()
            return result

        self.dispatch = timed_dispatch
        self.routed_experts_compute = timed_routed_experts_compute
        self.combine = timed_combine

        try:
            output = original_forward(self, hidden_states, intermediate_tensors, padding_mask)
        finally:
            # Restore originals
            self.dispatch = orig_dispatch
            self.routed_experts_compute = orig_routed_experts_compute
            self.combine = orig_combine

        # Schedule an async callback to read timings after GPU sync
        def _collect():
            torch.cuda.synchronize()
            try:
                dispatch_ms = timer.dispatch_start.elapsed_time(timer.dispatch_end)
                expert_ms = timer.expert_start.elapsed_time(timer.expert_end)
                combine_ms = timer.combine_start.elapsed_time(timer.combine_end)
                MoETimingStats.record(layer_number, dispatch_ms, expert_ms, combine_ms)
            except RuntimeError:
                pass  # Events may not have been recorded if layer was skipped

        # Register as a backward hook on the output so timing is collected
        # after the step completes (avoids blocking the forward pass).
        if isinstance(output, tuple):
            out_tensor = output[0]
        else:
            out_tensor = output
        if out_tensor is not None and out_tensor.requires_grad:
            out_tensor.register_hook(lambda _: _collect())
        else:
            # Fallback: collect inline (minor overhead, but correct)
            _collect()

        return output

    return timed_forward


@register_patch(
    "megatron.moe.timing",
    backend="megatron",
    phase="before_train",
    description="Instrument MoELayer with CUDA event timers to measure A2A vs expert GEMM time",
    condition=lambda ctx: getattr(get_args(ctx), "moe_timing_enabled", False),
)
def patch_moe_timing(ctx: PatchContext):
    """
    Patch MoELayer.forward to insert CUDA event timers.

    Records per-layer timing for:
    - dispatch A2A (comm)
    - expert GEMM (compute)
    - combine A2A (comm)

    The overlap potential = max(0, dispatch_ms + combine_ms - expert_ms).
    A large overlap potential means the model will benefit significantly from
    overlap_moe_expert_parallel_comm.
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer

    original_forward = MoELayer.forward
    log_rank_0("[Patch:megatron.moe.timing] Patching MoELayer.forward with timing instrumentation")

    def patched_forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        layer_num = getattr(self, "layer_number", -1)
        timed = _make_timed_moe_forward(original_forward, layer_num)
        return timed(self, hidden_states, intermediate_tensors, padding_mask)

    MoELayer.forward = patched_forward
    log_rank_0("[Patch:megatron.moe.timing]   MoELayer.forward patched")

    # Also patch Megatron training loop to log summary each step
    import megatron.training.training as training

    original_train_step = training.train_step

    def timed_train_step(*args, **kwargs):
        result = original_train_step(*args, **kwargs)
        # result[0] is loss; result[1] is skipped_iter; result[2] is grad_norm; etc.
        # We infer the step from MoETimingStats._step
        step = MoETimingStats._step
        if step % max(1, getattr(get_args(ctx), "log_interval", 10)) == 0:
            MoETimingStats.log_summary(step)
        MoETimingStats.reset_step()
        return result

    training.train_step = timed_train_step
    log_rank_0("[Patch:megatron.moe.timing]   train_step wrapped for per-step timing summary")
