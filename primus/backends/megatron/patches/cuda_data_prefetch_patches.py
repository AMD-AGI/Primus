###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Prefetch the next batch to the GPU on a side stream, on its own.

The FP8 delayed-scaling path already does this (``grad_zero_and_data_prefetch`` in
``delayed_fp8_scaling_patches.py``), but only there, and bundled with a stream-zeroed
``DDP.zero_grad_buffer`` that would undo ``defer_grad_buffer_zero``. Other recipes copy
the batch to the device inside the forward step, on the compute stream, with the GPU
idle: in the MXFP6 Flux MBS=32 step that is ~1.3 ms of exposed HtoD per step.

This installs only the prefetch half: the same ``CudaPrefetchIterator``, injected at the
same place (the ``data_iterator`` handed to ``forward_backward_func``, which keeps
Megatron's ``RerunDataIterator`` type check happy), sharing ``_PREFETCH_HANDLE`` so the
MLPerf warmup reset evicts it the same way.

The iterator casts floating-point tensors to the compute dtype while copying. The
diffusion forward step does exactly that cast itself (``.to(dtype=compute_dtype,
device="cuda")``), so batches are bit-identical and the step's own cast becomes a no-op.

Off by default. Enable with ``cuda_data_prefetch: true`` in the module config. Skipped
when the FP8 path already installs a prefetcher, and under TP > 1 or virtual pipeline
parallelism, which the single-stream prefetcher does not handle.
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.args.cuda_data_prefetch",
    backend="megatron",
    phase="setup",
    description="Expose --cuda-data-prefetch on Megatron's training argparse group.",
)
def patch_cuda_data_prefetch_arg(ctx: PatchContext) -> None:
    try:
        import megatron.training.arguments as margs
    except ImportError:
        return

    orig = getattr(margs, "_add_training_args", None)
    if getattr(orig, "_primus_cuda_data_prefetch", False):
        return
    if orig is None:
        raise RuntimeError(
            "megatron.training.arguments._add_training_args is missing, so "
            "--cuda-data-prefetch cannot be registered; update the patch rather than "
            "letting cuda_data_prefetch silently do nothing."
        )

    def _add_training_args(parser):
        parser = orig(parser)
        group = parser.add_argument_group(title="training")
        group.add_argument(
            "--cuda-data-prefetch",
            action="store_true",
            default=False,
            help="Copy the next batch to the GPU on a side stream during the current step.",
        )
        return parser

    _add_training_args._primus_cuda_data_prefetch = True
    margs._add_training_args = _add_training_args
    log_rank_0("[Patch:megatron.args.cuda_data_prefetch] added --cuda-data-prefetch")


@register_patch(
    "megatron.cuda_data_prefetch",
    backend="megatron",
    phase="before_train",
    description="Prefetch the next batch to the GPU on a side stream.",
    priority=42,  # after grad_zero_and_data_prefetch (41), so its guard below can see it
    condition=lambda ctx: bool(getattr(get_args(ctx), "cuda_data_prefetch", False)),
)
def patch_cuda_data_prefetch(ctx: PatchContext) -> None:
    import megatron.training.training as megatron_training

    from primus.backends.megatron.data.cuda_prefetch import CudaPrefetchIterator
    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
    from primus.backends.megatron.patches.delayed_fp8_scaling_patches import (
        _PREFETCH_HANDLE,
    )

    if is_patched(megatron_training, "megatron.grad_zero_and_data_prefetch"):
        log_rank_0("[Patch:cuda_data_prefetch] SKIPPED -- the FP8 path already prefetches.")
        return
    _PATCH_KEY = "megatron.cuda_data_prefetch"
    if is_patched(megatron_training, _PATCH_KEY):
        return

    args = get_args(ctx)
    if getattr(args, "tensor_model_parallel_size", 1) != 1:
        log_rank_0("[Patch:cuda_data_prefetch] SKIPPED -- TP > 1 broadcasts batches from rank 0.")
        return
    compute_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16

    _original_train_step = megatron_training.train_step
    _state: dict = {}
    _PREFETCH_HANDLE["state"] = _state

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
        if "iter" not in _state and data_iterator is not None:
            if isinstance(data_iterator, (list, tuple)):
                if not _state.get("vpp_skip_logged"):
                    log_rank_0("[Patch:cuda_data_prefetch] not prefetching: virtual pipeline parallel")
                    _state["vpp_skip_logged"] = True
            else:
                _state["iter"] = CudaPrefetchIterator(data_iterator, compute_dtype=compute_dtype)
                log_rank_0(f"[Patch:cuda_data_prefetch] prefetching batches (dtype={compute_dtype})")
        pf = _state.get("iter")

        def _prefetching_fwd_bwd(*fwd_args, **fwd_kwargs):
            if pf is not None:
                fwd_kwargs["data_iterator"] = pf
            return forward_backward_func(*fwd_args, **fwd_kwargs)

        return _original_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            _prefetching_fwd_bwd,
            iteration=iteration,
        )

    megatron_training.train_step = _patched_train_step
    mark_patched(megatron_training, _PATCH_KEY)
    log_rank_0("[Patch:megatron.cuda_data_prefetch] next-batch HtoD prefetch enabled")
