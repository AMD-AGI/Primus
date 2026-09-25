###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Zero only the part of the DDP gradient buffer that no Turbo wgrad overwrites.

Megatron zeroes the whole ``_ParamAndGradBuffer.grad_data`` every iteration so
that the beta=1 gradient accumulation each producer performs starts from zero.
Turbo's MXFP4 expert wgrad no longer accumulates on the step's first write: it
runs its epilogue at beta=0 and replaces the whole ``main_grad`` view. Zeroing
those slices first is then pure bandwidth. On GPT-OSS 20B the expert weights are
93% of a 20,488,420,224-element buffer, so almost all of the clear is dead work.

The patch wraps three Megatron methods and modifies none:

``DistributedDataParallel.zero_grad_buffer``
    Rotates :mod:`primus_turbo.pytorch.core.grad_ownership`'s log once per
    iteration. A reset abandoned before backward/communication (for example,
    synthetic-warmup cleanup) is discarded by the next rotation.

``_ParamAndGradBuffer.reset``
    Zeroes the complement of the owned slices instead of the whole buffer.

``_ParamAndGradBucketGroup.start_grad_sync``
    Rejects any skipped slice that the current iteration did not overwrite,
    before its reduce-scatter can consume stale data.

What makes the skip safe:

- A slice is owned only if the producer logged a beta=0 write to it during the
  *previous* iteration, and the log is rebuilt from scratch every iteration. The
  producer selects and logs the beta=0 writer in actual backward order, so
  activation recompute and staged forwards cannot leave a forward-time claim.
- An empty log means "zero everything". Iteration 0 has nothing logged, as do
  configurations where no overwrite-capable producer runs. The previous log
  is also discarded unless the current schedule has exactly one microbatch.
- CUDA-graph configurations do not install this consumer: Python ownership
  recording is not replayed with captured backward kernels.
- A claim is honoured only when the parameter's ``main_grad`` is exactly the
  contiguous ``[start, end)`` the buffer's own index map assigns it, matched by
  address, element count, and dtype. Inter-parameter alignment padding and the padding
  at the end of each bucket therefore always get zeroed: the reduce-scatter
  reads them.
- ``PRIMUS_TURBO_GRAD_OWNERSHIP_POISON=1`` fills the skipped slices with NaN
  instead of leaving them, so anything that is not genuinely overwritten reaches
  the reduce-scatter as NaN rather than as a plausible stale gradient. A finite
  sentinel is not usable here: real wgrad output collides with one, while finite
  operands cannot produce NaN.

Set ``PRIMUS_TURBO_GRAD_OWNERSHIP=1`` to opt in. It defaults off, so workloads
that have not explicitly validated the producer/consumer contract continue to
zero the whole buffer.
"""

from __future__ import annotations

import os
from itertools import pairwise

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

Slice = tuple[int, int, object]

_DISABLED = os.environ.get("PRIMUS_TURBO_GRAD_OWNERSHIP", "0") != "1"
_POISON = os.environ.get("PRIMUS_TURBO_GRAD_OWNERSHIP_POISON", "0") == "1"

# Rebuilt every iteration by the wrapped ``zero_grad_buffer``; consumed by the
# wrapped ``reset`` of every buffer belonging to that iteration.
_state: dict[str, object] = {"owned": frozenset(), "seen": set(), "logged": False}


def _rotate_ownership(grad_ownership, *, discard: bool) -> frozenset[Slice]:
    """Start a producer epoch, optionally discarding the completed epoch.

    Pipeline warmup runs with ``args.curr_iteration == -1``. Its beta=0 writes
    exercise kernels but must never authorize skipped clearing in iteration 0.
    Rotating and discarding here clears the producer log without carrying those
    synthetic writes into real training.
    """
    owned = grad_ownership.begin_step()
    return frozenset() if discard else owned


def _get_num_microbatches() -> int:
    """Read Megatron's current schedule at reset time, not patch-install time."""
    from megatron.core.num_microbatches_calculator import get_num_microbatches

    return get_num_microbatches()


def _discard_ownership_for_step(args) -> bool:
    """Whether this step must retain Megatron's full gradient-buffer clear."""
    if getattr(args, "curr_iteration", None) == -1:
        return True
    try:
        return _get_num_microbatches() != 1
    except Exception:  # noqa: BLE001 -- any uncertainty must fail closed.
        # If the calculator is unavailable or not initialized, ownership for
        # the current schedule is not proven. Fail closed with a full clear.
        return True


def _is_enabled(ctx: PatchContext) -> bool:
    if _DISABLED:
        return False
    args = get_args(ctx)

    # Ownership recording happens in Python after the beta=0 custom op and is
    # therefore not replayed by CUDA graphs. Keep full framework clearing for
    # any configured graph mode. Megatron may expose the graph implementation
    # directly on args or on a nested model configuration, depending on the
    # trainer/bridge path.
    graph_configs = [
        args,
        getattr(args, "model_cfg", None),
        getattr(args, "model_config", None),
        getattr(args, "model", None),
    ]
    for config in graph_configs:
        if config is None:
            continue
        if bool(getattr(config, "enable_cuda_graph", False)) or bool(
            getattr(config, "external_cuda_graph", False)
        ):
            return False
        graph_impl = getattr(config, "cuda_graph_impl", None)
        graph_impl = getattr(graph_impl, "value", graph_impl)
        if graph_impl is not None and str(graph_impl).lower() not in {
            "",
            "none",
            "false",
        }:
            return False

    # The claim this patch trusts is only ever recorded by the fused wgrad path.
    if not bool(getattr(args, "gradient_accumulation_fusion", False)):
        return False
    return is_primus_turbo_can_patch(ctx)


def _owned_slices(buffer) -> list[tuple[int, int, Slice]]:
    """Buffer offsets whose parameter was fully overwritten last iteration.

    Returns ``(start, end, key)`` triples, where ``key`` is the producer's
    ``(data_ptr, numel, dtype)`` log entry, so the caller can report back exactly what
    it skipped.
    """
    owned: frozenset[Slice] = _state["owned"]
    grad = getattr(buffer, "grad_data", None)
    index_map = getattr(buffer, "param_index_map", None)
    if not owned or grad is None or index_map is None or grad.numel() == 0:
        return []

    base = grad.data_ptr()
    itemsize = grad.element_size()
    found = []
    for param, entry in index_map.items():
        start, end, _bucket_id = entry
        main_grad = getattr(param, "main_grad", None)
        if main_grad is None or main_grad.dtype != grad.dtype:
            continue
        if not main_grad.is_contiguous() or main_grad.numel() != (end - start):
            continue
        if main_grad.data_ptr() != base + start * itemsize:
            continue
        key = (main_grad.data_ptr(), main_grad.numel(), main_grad.dtype)
        if key in owned:
            found.append((start, end, key))

    found.sort()
    for (_s0, end0, _k0), (start1, _e1, _k1) in pairwise(found):
        if start1 < end0:
            raise RuntimeError(f"overlapping owned gradient slices ending {end0} and starting {start1}")
    return found


def _reset_complement(buffer) -> bool:
    """Zero everything in ``buffer`` except the owned slices. False if it declined."""
    slices = _owned_slices(buffer)
    if not slices:
        return False

    grad = buffer.grad_data
    numel = grad.numel()
    cursor = 0
    for start, end, _key in slices:
        if start > cursor:
            grad[cursor:start].zero_()
        cursor = end
    if cursor < numel:
        grad[cursor:numel].zero_()

    # Megatron's native reset also clears the detached higher-precision local
    # accumulation buffers. The optimized path bypasses native reset, so it
    # must preserve that side effect explicitly.
    for extra_grad in getattr(buffer, "extra_main_grads", ()):
        extra_grad.zero_()

    if _POISON:
        for start, end, _key in slices:
            grad[start:end].fill_(float("nan"))

    from primus_turbo.pytorch.core import grad_ownership

    grad_ownership.note_skipped(key for _start, _end, key in slices)

    if not _state["logged"]:
        skipped = sum(end - start for start, end, _key in slices)
        log_rank_0(
            f"[Patch:megatron.turbo.grad_buffer_ownership] buffer numel={numel} owned slices="
            f"{len(slices)} skipped={skipped} ({100.0 * skipped / numel:.2f}%)"
            + ("  POISON=NaN" if _POISON else "")
        )
        _state["logged"] = True
    return True


def _validate_bucket_group_overwrites(bucket_group, grad_ownership) -> None:
    """Validate this bucket group's skipped slices before its collective reads them."""
    slices = []
    for bucket in bucket_group.buckets:
        params = getattr(bucket, "params_list", getattr(bucket, "params", ()))
        for param in params:
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                slices.append((main_grad.data_ptr(), main_grad.numel(), main_grad.dtype))
    grad_ownership.validate_overwritten(slices)


@register_patch(
    "megatron.core.distributed.grad_buffer_ownership",
    backend="megatron",
    phase="before_train",
    description="Zero only the complement of the gradient-buffer slices that Primus-Turbo's beta=0 wgrad epilogue fully overwrites, instead of the whole buffer.",
    condition=_is_enabled,
    priority=70,
)
def patch_grad_buffer_ownership(ctx: PatchContext) -> None:
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.param_and_grad_buffer import (
        _ParamAndGradBucketGroup,
        _ParamAndGradBuffer,
    )
    from primus_turbo.pytorch.core import grad_ownership

    if getattr(_ParamAndGradBuffer.reset, "_primus_grad_ownership", False):
        return

    original_zero_grad_buffer = DistributedDataParallel.zero_grad_buffer
    original_reset = _ParamAndGradBuffer.reset
    original_start_grad_sync = _ParamAndGradBucketGroup.start_grad_sync
    args = get_args(ctx)

    def zero_grad_buffer(self):
        seen = _state["seen"]
        if id(self) in seen or not seen:
            _state["owned"] = _rotate_ownership(
                grad_ownership,
                discard=_discard_ownership_for_step(args),
            )
            seen.clear()
        seen.add(id(self))
        return original_zero_grad_buffer(self)

    def reset(self):
        if not _reset_complement(self):
            original_reset(self)

    def start_grad_sync(self, *args, **kwargs):
        # The first-batch duplicate-dispatch call is a native no-op; do not
        # validate unrelated later buckets after communication already began.
        if not (self.is_first_batch and self.grad_reduce_handle is not None):
            _validate_bucket_group_overwrites(self, grad_ownership)
        return original_start_grad_sync(self, *args, **kwargs)

    reset._primus_grad_ownership = True
    DistributedDataParallel.zero_grad_buffer = zero_grad_buffer
    _ParamAndGradBuffer.reset = reset
    _ParamAndGradBucketGroup.start_grad_sync = start_grad_sync

    log_rank_0(
        "[Patch:megatron.turbo.grad_buffer_ownership] Wrapped zero_grad_buffer() and "
        "_ParamAndGradBuffer.reset(), with pre-collective overwrite validation; owned "
        "slices are reported on the first iteration that has any."
        + ("  POISON MODE: skipped slices filled with NaN." if _POISON else "")
    )
