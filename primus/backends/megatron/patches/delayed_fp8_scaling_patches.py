###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron train_step patch for delayed FP8 scaling updates and preamble
optimization.

Patch 1 -- Delayed FP8 scale update:
    Wraps train_step to call the scale update preamble on every FP8 delayed
    module before the original train_step runs.  For most_recent +
    history_len=1 (production config), a fast path stages per-module amaxes
    into the registry-batched ``staged_amaxes_3n`` / ``scales_3n`` tensors,
    computes new scales with a few fused ops, and scatters them back to the
    per-module scalar ``m.scale_*`` buffers.  Per-module scalars (rather
    than views into a shared ``(N,)`` tensor) are required so that each
    module's buffer storage is independent for torch.compile version
    tracking.

Patch 2 -- Grad-zero stream overlap + data HtoD prefetch:
    Dispatches DDP grad-buffer zeroing (grad_data.zero_()) on a secondary
    CUDA stream so it overlaps with HtoD data transfer, saving ~4.6 ms/iter.
    Additionally, prefetches the next data batch to GPU on a dedicated HtoD
    stream.  The prefetch iterator is injected at the forward_backward_func
    level (not by replacing data_iterator in train_step) to preserve
    compatibility with Megatron's RerunDataIterator type assertion.
"""

import torch
from megatron.core.enums import Fp8Recipe

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

# Module-level handle so the MLPerf warmup hook (or any external code) can
# reach the CudaPrefetchIterator state owned by
# ``patch_grad_zero_and_data_prefetch`` (which would otherwise live solely
# in that patch's closure and be unreachable from outside).
#
# The warmup hook needs this so it can invalidate the cached prefetch
# iterator at the end of the warmup epilogue.  Without that invalidation,
# the prefetcher stays bound to the synthetic-data iterator that was passed
# in for warmup step 1, and every subsequent real training step silently
# reads from the cycling synthetic dataset (because
# ``MegatronDataloaderWrapper`` is cyclic and never raises ``StopIteration``).
_PREFETCH_HANDLE: dict = {"state": None}

# Shared state for async amax allreduce between Patch 1 (scale update) and
# Patch 2 (fwd/bwd wrapper).  Patch 2 launches the async allreduce after
# forward_backward_func returns; Patch 1 waits on the handle at the start
# of the next train_step before computing new scales.
#
# Realistic overlap window: the time between forward_backward_func returning
# (in step N) and the Patch 1 wait/compute at the start of step N+1.  In
# practice that's roughly ``optimizer.step + grad_zero`` -- not the full
# train_step.  The overlap is still useful but smaller than the all-reduce
# cost in most configs, so this is best-effort latency hiding rather than a
# free win.
_ASYNC_AMAX_HANDLE: dict = {"handle": None, "registry": None}


def _reset_async_amax_state():
    """Drop any pending async amax allreduce handle.

    Idempotent best-effort cleanup invoked from both call sites in this
    module (Patch 1's wait path and Patch 2's launch path).  Leaving a
    stale handle in ``_ASYNC_AMAX_HANDLE["handle"]`` would cause the next
    train_step to try to wait on an already-consumed or never-launched
    work object and either deadlock or raise.
    """
    _ASYNC_AMAX_HANDLE["handle"] = None


def get_prefetch_state():
    """Return the closure-shared ``_prefetch_state`` dict, or ``None`` if
    ``patch_grad_zero_and_data_prefetch`` has not been installed yet."""
    return _PREFETCH_HANDLE.get("state")


def reset_prefetch_state():
    """Drop the cached ``CudaPrefetchIterator`` so the next ``train_step``
    rebuilds it around its current ``data_iterator`` argument.

    Required after MLPerf warmup, because warmup step 1 is the first call
    into ``_patched_train_step``, so the prefetcher gets bound to the
    synthetic iterator.  Without invalidation, every real training step
    afterwards substitutes the cached prefetcher (still wrapping synthetic
    data) for the real ``data_iterator`` argument inside the
    ``_synced_prefetch_fwd_bwd`` wrapper.

    Returns the evicted iterator (or ``None`` if no cache existed) so
    callers can log what was dropped.
    """
    state = _PREFETCH_HANDLE.get("state")
    if state is None:
        return None
    return state.pop("iter", None)


def _needs_delayed_scaling(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if args is None or not bool(getattr(args, "fp8", False)):
        return False
    return (
        getattr(args, "fp8_scaling_strategy", "dynamic") == "delayed"
        or getattr(args, "fp8_recipe", None) == Fp8Recipe.delayed
    ) and not getattr(args, "disable_delayed_scaling_patches", False)


@register_patch(
    "megatron.fp8.delayed_scaling_update",
    backend="megatron",
    phase="before_train",
    description="Wrap train_step to update delayed FP8 scales before each step.",
    priority=40,
    condition=_needs_delayed_scaling,
)
def patch_delayed_fp8_update(ctx: PatchContext):
    import megatron.training.training as megatron_training

    from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
        _DelayedScalingRegistry,
        _fast_update_scales,
        _fast_update_scales_with_history,
        _wait_and_compute_scales,
    )
    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    _PATCH_KEY = "megatron.fp8.delayed_scaling_update"
    if is_patched(megatron_training, _PATCH_KEY):
        log_rank_0("[Patch:delayed_scaling_update] Already applied; skipping re-wrap.")
        return

    _original_train_step = megatron_training.train_step
    _cached_delayed_modules = []
    _registry = None

    def _patched_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        nonlocal _registry
        if not _cached_delayed_modules:
            _cached_delayed_modules.extend(
                m
                for model_chunk in model
                for m in model_chunk.modules()
                if getattr(m, "_use_delayed_scaling", False)
            )
        if _cached_delayed_modules:
            if _registry is None:
                _registry = _DelayedScalingRegistry(_cached_delayed_modules)
                _ASYNC_AMAX_HANDLE["registry"] = _registry

            pending_handle = _ASYNC_AMAX_HANDLE.get("handle")
            if pending_handle is not None:
                try:
                    _wait_and_compute_scales(_registry, pending_handle)
                finally:
                    # Always drop the handle, even if wait / compute raised:
                    # leaving a consumed handle here would cause the next
                    # train_step to try to wait on it again.
                    _reset_async_amax_state()
            else:
                if _registry.algo == "most_recent" and _registry.history_len == 1:
                    _fast_update_scales(_registry)
                else:
                    _fast_update_scales_with_history(_registry)
        return _original_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
            iteration=iteration,
        )

    megatron_training.train_step = _patched_train_step
    mark_patched(megatron_training, _PATCH_KEY)
    log_rank_0(
        "[Patch:delayed_scaling_update] "
        "Wrapped train_step with async amax allreduce support. "
        "First step uses synchronous fallback; subsequent steps wait on "
        "async handle launched by post-fwd_bwd hook."
    )


# ---------------------------------------------------------------------------
# Patch 2: Grad-zero stream overlap + data HtoD prefetch
# ---------------------------------------------------------------------------


@register_patch(
    "megatron.grad_zero_and_data_prefetch",
    backend="megatron",
    phase="before_train",
    description="Overlap grad buffer zeroing and data HtoD transfer via secondary CUDA streams.",
    priority=41,
    condition=_needs_delayed_scaling,
)
def patch_grad_zero_and_data_prefetch(ctx: PatchContext):
    args = get_args(ctx)
    if getattr(args, "reuse_grad_buf_for_mxfp8_param_ag", False):
        log_rank_0(
            "[Patch:grad_zero_and_data_prefetch] SKIPPED — "
            "reuse_grad_buf_for_mxfp8_param_ag is set (shared param/grad buffer)."
        )
        return

    import megatron.training.training as megatron_training
    from megatron.core.distributed import DistributedDataParallel

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    _PATCH_KEY = "megatron.grad_zero_and_data_prefetch"
    if is_patched(megatron_training, _PATCH_KEY):
        log_rank_0("[Patch:grad_zero_and_data_prefetch] Already applied; skipping re-wrap.")
        return

    tp_size = getattr(args, "tensor_model_parallel_size", 1)

    _original_train_step = megatron_training.train_step
    _zero_stream = torch.cuda.Stream()
    _prefetch_state: dict = {}
    # Expose the closure-local prefetch state so the MLPerf warmup hook can
    # invalidate the cached prefetch iterator at the end of warmup; see
    # ``reset_prefetch_state`` above for the rationale.
    _PREFETCH_HANDLE["state"] = _prefetch_state

    def _stream_zero_grad_buffer(self):
        """CPU metadata on main thread; GPU grad_data.zero_() on secondary stream."""
        if getattr(self.config, "cuda_graph_impl", "none") != "transformer_engine":
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
        with torch.cuda.stream(_zero_stream):
            for buffer in self.buffers + self.expert_parallel_buffers:
                buffer.reset()
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.reset()

    DistributedDataParallel.zero_grad_buffer = _stream_zero_grad_buffer

    def _patched_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        if "iter" not in _prefetch_state and tp_size == 1:
            if isinstance(data_iterator, (list, tuple)):
                # Virtual pipeline parallel passes a list of per-chunk iterators,
                # which the single-stream prefetcher cannot wrap. Skip prefetch
                # (a pure HtoD-overlap optimization) and let the original iterator
                # flow through unchanged.
                if not _prefetch_state.get("vpp_skip_logged"):
                    log_rank_0(
                        "[Patch:grad_zero_and_data_prefetch] "
                        "Skipping CudaPrefetchIterator: data_iterator is a list "
                        "(virtual pipeline parallel)."
                    )
                    _prefetch_state["vpp_skip_logged"] = True
            else:
                from primus.backends.megatron.data.cuda_prefetch import (
                    CudaPrefetchIterator,
                )

                compute_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16
                _prefetch_state["iter"] = CudaPrefetchIterator(
                    data_iterator,
                    compute_dtype=compute_dtype,
                )
                log_rank_0(
                    "[Patch:grad_zero_and_data_prefetch] "
                    f"Created CudaPrefetchIterator (dtype={compute_dtype})."
                )

        _pf = _prefetch_state.get("iter")

        def _synced_prefetch_fwd_bwd(*fwd_args, **fwd_kwargs):
            torch.cuda.current_stream().wait_stream(_zero_stream)
            if _pf is not None:
                fwd_kwargs["data_iterator"] = _pf
            result = forward_backward_func(*fwd_args, **fwd_kwargs)

            registry = _ASYNC_AMAX_HANDLE.get("registry")
            if registry is not None:
                from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
                    _stage_and_launch_async_allreduce,
                )

                try:
                    _ASYNC_AMAX_HANDLE["handle"] = _stage_and_launch_async_allreduce(registry)
                except Exception:
                    # Launch failed: clear so the next train_step takes the
                    # synchronous fallback path instead of trying to wait on
                    # a non-existent handle.
                    _reset_async_amax_state()
                    raise

            return result

        return _original_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            _synced_prefetch_fwd_bwd,
            iteration=iteration,
        )

    megatron_training.train_step = _patched_train_step
    mark_patched(megatron_training, _PATCH_KEY)
    log_rank_0(
        "[Patch:grad_zero_and_data_prefetch] "
        "DDP grad_data.zero_() on secondary stream; "
        "data HtoD prefetch on secondary stream; "
        "secondary streams synced before forward_backward_func; "
        "async amax allreduce launched after forward_backward_func "
        "(awaited at the start of the next train_step)."
    )
