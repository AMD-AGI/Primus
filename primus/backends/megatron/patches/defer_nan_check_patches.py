###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check the NaN / Inf validations of a training step once, before the optimizer step.

With ``check_for_nan_in_loss_and_grad`` (the default) Megatron validates every micro-batch's loss (NaN, then
Inf, in ``pretrain_gpt.loss_func``) and, in the last micro-batch's backward, every gradient bucket's norm
(``_ParamAndGradBucketGroup.check_grads``). With reruns disabled each ``validate_result`` turns a device
tensor into a Python bool, i.e. a host sync that drains the GPU queue: about 140 syncs per step on the Kimi K3
proxy (8 micro-batches), and the GPU idles while the CPU refills the queue after each one.

``defer_nan_check: true``: in that mode a fatal ``torch.isnan`` / ``torch.isinf`` validation of a CUDA
tensor only records the rejection flag on the device. The flags are read with one sync after
``forward_backward_func`` returns (all micro-batches, gradient reduction included) and before
``optimizer.step()``, and after ``evaluate_and_print_results``. A rejected result is then re-validated by the original method,
which logs it and raises the same ``RuntimeError``, so a NaN / Inf still stops the job before the parameters
are updated; it is reported at the end of the step instead of at the micro-batch. Other modes, non-fatal
checks and other rejection functions run as before.
"""

import functools
import inspect

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# flags held at most this long before a flush (evaluation loops without a train step)
_MAX_PENDING = 4096


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "defer_nan_check", False))


@register_patch(
    "megatron.training.defer_nan_check",
    backend="megatron",
    phase="before_train",
    description="Read the per-step NaN / Inf loss and grad validations with one host sync",
    priority=40,
    condition=_enabled,
)
def patch_defer_nan_check(ctx: PatchContext):
    import megatron.training.training as megatron_training
    from megatron.core.rerun_state_machine import RerunMode, RerunStateMachine

    orig_validate = RerunStateMachine.validate_result
    pending = []  # (flag, state machine, result, rejection_func, message)

    def flush():
        if not pending:
            return
        items = list(pending)
        pending.clear()
        if not bool(torch.stack([flag for flag, *_ in items]).any()):
            return
        for flag, machine, result, rejection_func, message in items:
            if bool(flag):
                orig_validate(machine, result, rejection_func, message, fatal=True)

    @functools.wraps(orig_validate)
    def validate_result(
        self, result, rejection_func, message, comparison_func=None, tolerance=0.0, fatal=True
    ):
        if (
            self.mode == RerunMode.DISABLED
            and fatal
            and rejection_func in (torch.isnan, torch.isinf)
            and isinstance(result, torch.Tensor)
            and result.is_cuda
        ):
            flag = rejection_func(result)
            flag = flag.reshape(()) if flag.numel() == 1 else flag.any()
            pending.append((flag, self, result, rejection_func, message))
            if len(pending) >= _MAX_PENDING:
                flush()
            return
        return orig_validate(self, result, rejection_func, message, comparison_func, tolerance, fatal)

    def flushing(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            flush()
            return out

        return wrapped

    orig_train_step = megatron_training.train_step
    # forward_backward_func is passed by position at the call in train()
    params = list(inspect.signature(inspect.unwrap(orig_train_step)).parameters)
    if "forward_backward_func" not in params:
        raise RuntimeError(f"defer_nan_check: train_step has no forward_backward_func parameter: {params}")
    fwd_bwd_index = params.index("forward_backward_func")

    @functools.wraps(orig_train_step)
    def train_step(*args, **kwargs):
        if "forward_backward_func" in kwargs:
            kwargs["forward_backward_func"] = flushing(kwargs["forward_backward_func"])
        else:
            args = list(args)
            args[fwd_bwd_index] = flushing(args[fwd_bwd_index])
        return orig_train_step(*args, **kwargs)

    RerunStateMachine.validate_result = validate_result
    megatron_training.train_step = train_step
    # not evaluate itself: the megatron.training.evaluate patch may replace it after this one
    megatron_training.evaluate_and_print_results = flushing(megatron_training.evaluate_and_print_results)
    log_rank_0("[Patch:defer_nan_check] NaN / Inf loss and grad validations read once per step")
