###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check a step's fatal NaN / Inf validations with one host sync, before the step's update.

``check_for_nan_in_loss_and_grad`` (on by default) validates every micro-batch's loss and every
gradient bucket's norm through ``RerunStateMachine.validate_result``. In the default ``disabled``
rerun mode that evaluates ``torch.isnan`` / ``torch.isinf`` on the host at once, a sync each, and the
GPU then waits for the host to queue work again, for every micro-batch and every bucket of a step.

With this patch those calls (fatal, ``torch.isnan`` / ``torch.isinf``, a one-element CUDA result)
are queued with their verdict computed on the device, and one sync checks the queue:

- in ``DistributedDataParallel.finish_grad_sync``, which runs once per step after the last
  backward, at its start and end (the end covers buckets it starts itself);
- at the entry of the optimizers' ``step``, and at the end of ``evaluate``, as backstops;
- whenever the queue reaches ``_MAX_PENDING`` entries.

A rejected entry is replayed through the original ``validate_result``, in queue order, so the error,
its message and the validation log are the ones upstream raises, in the same iteration and before
any parameter is updated. What changes is the moment: the step runs to the end of its backward
first. Other rerun modes, non-fatal checks and other rejection functions are untouched.

``defer_nan_check`` (trainer config) is null by default, which defers on gfx1250 only, where it was
validated; true / false turn it on / off on every architecture.

The gradient checks keep one sync per bucket group (a flush right after ``check_grads``) unless
``grad_checks_deferrable``. With MegaMoE at EP > 1 and overlapped gradient reduction, the
reduce-scatter a bucket group starts must not queue behind the next layers' Mega kernels: their
workgroups spin until the other EP ranks reach the same kernel, and there are more of them than fit
on the GPU at once, so their dispatch does not finish while they wait. ROCm runs a process's streams
on ``GPU_MAX_HW_QUEUES`` hardware queues and dispatches a queue in order; with 2 (Primus' default)
a reduce-scatter, or the work it waits for, can land in a Mega kernel's queue behind it, and the
ranks deadlock (high-priority streams for the reduce-scatters do not help). With 4 queues it runs.
The group sync instead puts each reduce-scatter in its queue ahead of those kernels.
"""

import os
from functools import wraps

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.gpu_arch import is_gfx1250
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.rerun.deferred_nan_check"
_MAX_PENDING = 4096
_MIN_HW_QUEUES = 4
_PENDING: list = []
_ORIGINAL = None
_GRADS_DEFERRABLE = None


def _enabled(ctx: PatchContext) -> bool:
    # null (the default): gfx1250 only, where it was validated; true / false: every architecture
    enabled = getattr(get_args(ctx), "defer_nan_check", None)
    return is_gfx1250() if enabled is None else bool(enabled)


def flush() -> None:
    """Check every queued validation with one sync; replay them in order if any is rejected."""
    if not _PENDING:
        return
    pending = list(_PENDING)
    _PENDING.clear()
    if not bool(torch.cat([entry[0] for entry in pending]).any()):
        return
    for _, machine, args in pending:
        _ORIGINAL(machine, *args)


def _deferrable(machine, result, rejection_func, fatal) -> bool:
    from megatron.core.rerun_state_machine import RerunMode

    return (
        machine.mode == RerunMode.DISABLED
        and fatal
        and (rejection_func is torch.isnan or rejection_func is torch.isinf)
        and isinstance(result, torch.Tensor)
        and result.is_cuda
        and result.numel() == 1
    )


def _flushing(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        flush()
        out = fn(*args, **kwargs)
        flush()
        return out

    return wrapper


def _grad_checks_deferrable():
    try:
        from megatron.training import global_vars

        args = global_vars.get_args()
    except (ImportError, AssertionError):
        return True, "no Megatron training arguments"
    if not getattr(args, "use_turbo_mega_moe", False) or args.expert_model_parallel_size <= 1:
        return True, "no MegaMoE expert parallelism"
    if not getattr(args, "overlap_grad_reduce", False):
        return True, "the gradient reduce-scatters do not overlap the backward"
    queues = os.environ.get("GPU_MAX_HW_QUEUES") or "4"  # unset: ROCm's default, 4
    if int(queues) < _MIN_HW_QUEUES:
        return False, f"MegaMoE at EP > 1 with GPU_MAX_HW_QUEUES={queues} (needs {_MIN_HW_QUEUES})"
    return True, f"GPU_MAX_HW_QUEUES={queues}"


def grad_checks_deferrable() -> bool:
    """Whether the gradient checks can go without their per-bucket-group sync (module docstring)."""
    global _GRADS_DEFERRABLE
    if _GRADS_DEFERRABLE is None:
        _GRADS_DEFERRABLE, why = _grad_checks_deferrable()
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] gradient checks "
            f"{'deferred' if _GRADS_DEFERRABLE else 'keep one sync per bucket group'}: {why}"
        )
    return _GRADS_DEFERRABLE


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="One host sync per step for the fatal NaN/Inf checks of losses and gradient norms",
    condition=_enabled,
    # after megatron.ddp.fused_grad_check_scaling (50): the guard below wraps its check_grads
    priority=60,
)
def patch_deferred_nan_check(ctx: PatchContext) -> None:
    global _ORIGINAL
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer,
        FP32Optimizer,
        MixedPrecisionOptimizer,
    )
    from megatron.core.rerun_state_machine import RerunStateMachine

    # imported here so that loading this module stays light (its CPU tests load it on its own)
    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    if is_patched(RerunStateMachine, _PATCH_KEY):
        return
    _ORIGINAL = RerunStateMachine.validate_result

    @wraps(_ORIGINAL)
    def validate_result(
        self, result, rejection_func, message, comparison_func=None, tolerance=0.0, fatal=True
    ):
        if not _deferrable(self, result, rejection_func, fatal):
            return _ORIGINAL(self, result, rejection_func, message, comparison_func, tolerance, fatal)
        result = result.detach()
        _PENDING.append(
            (
                rejection_func(result).reshape(1),
                self,
                (result, rejection_func, message, comparison_func, tolerance, fatal),
            )
        )
        if len(_PENDING) >= _MAX_PENDING:
            flush()

    # read by megatron.ddp.fused_grad_check_scaling: its calls to validate_result only queue
    validate_result._primus_deferred = True
    RerunStateMachine.validate_result = validate_result

    flushed = []
    # the optimizers' own step too, for an optimizer that steps them in place of ChainedOptimizer.step
    for owner, name in (
        (DistributedDataParallel, "finish_grad_sync"),
        (MixedPrecisionOptimizer, "step"),
        (FP32Optimizer, "step"),
        (ChainedOptimizer, "step"),
    ):
        setattr(owner, name, _flushing(getattr(owner, name)))
        flushed.append(f"{owner.__name__}.{name}")
    check_grads = _ParamAndGradBucketGroup.check_grads

    @wraps(check_grads)
    def guarded_check_grads(self, *args, **kwargs):
        out = check_grads(self, *args, **kwargs)
        if not grad_checks_deferrable():
            flush()  # one sync per bucket group, as without the deferral
        return out

    _ParamAndGradBucketGroup.check_grads = guarded_check_grads
    try:
        from megatron.training import training

        training.evaluate = _flushing(training.evaluate)
        flushed.append("training.evaluate")
    except ImportError:
        pass
    mark_patched(RerunStateMachine, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] fatal NaN/Inf validations are checked with one sync at: {', '.join(flushed)}"
    )
