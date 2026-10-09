###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gradient-norm clipping on the device (``--device-grad-clip``).

Megatron reads the global gradient norm back to the host before it can clip (``get_grad_norm_fp32`` ends in
``total_norm.item()``). The all-reduce of the squared norm is the last thing the backward tail does, so the host is
blocked for the whole tail and then launches the clip and the optimizer while the GPU waits.

With this gate:

* ``get_grad_norm_fp32`` (2-norm) returns a :class:`DeviceGradNorm` -- the same all-reduced squared norm, left on the
  device;
* ``clip_grad_by_total_norm_fp32`` computes the coefficient on the device (fp64, cast to fp32) and always scales the
  gradients by it (1.0 when the norm is within ``clip_grad``, which leaves them unchanged), coalescing contiguous
  runs as ``flat_grad_clip`` does;
* the norm reaches the host lazily: ``training_log`` gets a float only on the steps that print or write it, and
  ``--light-log-sync-events`` reads it back asynchronously.

Losses, gradient norms and parameters are bitwise identical to the host path; see ``device_grad_clip`` for the
numerics. ``foreach`` is the only implementation; ``te`` (scaling inside the fused Adam kernel) is reserved and
rejected until a Transformer Engine build with a device-side gradient scale is available.

Off by default (``device_grad_clip: off``).
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

DEVICE_GRAD_CLIP_CHOICES = ("off", "foreach", "te")


@register_patch(
    "megatron.args.device_grad_clip",
    backend="megatron",
    phase="setup",
    description="Expose --device-grad-clip on Megatron's regularization argparse group.",
)
def patch_device_grad_clip_arg(ctx: PatchContext) -> None:
    try:
        import megatron.training.arguments as margs
    except ImportError:
        return
    orig = getattr(margs, "_add_regularization_args", None)
    if getattr(orig, "_primus_device_grad_clip", False):
        return
    if orig is None:
        raise RuntimeError(
            "megatron.training.arguments._add_regularization_args is missing, so --device-grad-clip cannot be "
            "registered; update the patch rather than letting the gate silently do nothing."
        )

    def _add_regularization_args(parser):
        parser = orig(parser)
        group = parser.add_argument_group(title="regularization")
        group.add_argument(
            "--device-grad-clip",
            type=str,
            default="off",
            choices=DEVICE_GRAD_CLIP_CHOICES,
            help="Compute the clip coefficient on the device and scale without a host sync (foreach).",
        )
        return parser

    _add_regularization_args._primus_device_grad_clip = True
    margs._add_regularization_args = _add_regularization_args
    log_rank_0("[Patch:megatron.args.device_grad_clip] added --device-grad-clip")


def _device_grad_clip_mode(ctx: PatchContext) -> str:
    mode = getattr(get_args(ctx), "device_grad_clip", "off")
    if mode in (None, False):
        return "off"
    return str(mode)


def _consumes_grad_norm(args, iteration, is_first_iteration, has_writer) -> bool:
    """Whether Megatron's training_log reads grad_norm this call (prints it, or writes it to TensorBoard)."""
    if is_first_iteration or iteration % args.log_interval == 0:
        return True
    return has_writer and iteration % args.tensorboard_log_interval == 0


@register_patch(
    "megatron.optimizer.device_grad_clip",
    backend="megatron",
    phase="before_train",
    description="Gradient norm and clip coefficient on the device; the gradients are scaled without a host sync.",
    # After flat_grad_clip (50), whose clip function this one delegates to for host-side norms.
    priority=60,
    condition=lambda ctx: _device_grad_clip_mode(ctx) != "off",
)
def patch_device_grad_clip(ctx: PatchContext) -> None:
    import megatron.training.training as megatron_training
    from megatron.core.optimizer import clip_grads as _clip_grads
    from megatron.core.optimizer import optimizer as _optimizer
    from megatron.core.utils import to_local_if_dtensor

    from primus.backends.megatron.core.optimizer.device_grad_clip import (
        DeviceGradNorm,
        clip_coefficient,
        scale_grads_,
    )
    from primus.backends.megatron.patches.optimizer_flat_grad_clip_patches import (
        _coalesce,
    )

    mode = _device_grad_clip_mode(ctx)
    if mode == "te":
        raise ValueError(
            "device_grad_clip: 'te' needs a Transformer Engine build whose fused Adam takes a device-side gradient "
            "scale; it is not available here. Use 'foreach'."
        )
    args = get_args(ctx)
    if getattr(args, "check_for_nan_in_loss_and_grad", False):
        raise ValueError(
            "device_grad_clip is incompatible with check_for_nan_in_loss_and_grad, whose per-step check reads the "
            "norm on the host anyway."
        )

    orig_norm = _clip_grads.get_grad_norm_fp32
    if getattr(orig_norm, "_primus_device_grad_clip", False):
        return
    orig_clip = _optimizer.clip_grad_by_total_norm_fp32
    flat_runs = bool(getattr(args, "flat_grad_clip", False))

    def get_grad_norm_fp32(grads_for_norm, norm_type=2, grad_stats_parallel_group=None):
        # Same reduction as Megatron's 2-norm path, stopping short of the .item().
        if float(norm_type) != 2.0:
            return orig_norm(grads_for_norm, norm_type, grad_stats_parallel_group)
        if isinstance(grads_for_norm, torch.Tensor):
            grads_for_norm = [grads_for_norm]
        data_parallel_group = None
        for grad in grads_for_norm:
            data_parallel_group = _clip_grads.get_data_parallel_group_if_dtensor(grad, data_parallel_group)
        grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]
        if grads_for_norm:
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device="cuda")
            grad_norm, _ = _clip_grads.multi_tensor_applier(
                _clip_grads.l2_norm_impl, dummy_overflow_buf, [grads_for_norm], False
            )
        else:
            grad_norm = torch.zeros(1, dtype=torch.float, device="cuda")
        total_norm = grad_norm**2.0
        if data_parallel_group:
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group)
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )
        return DeviceGradNorm(total_norm)

    def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if not isinstance(total_norm, DeviceGradNorm):
            return orig_clip(parameters, max_norm, total_norm, use_decoupled_grad=use_decoupled_grad)
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = []
        for param in parameters:
            if use_decoupled_grad:
                g = getattr(param, "decoupled_grad", None)
                if g is not None:
                    assert g.dtype in (torch.float32, torch.bfloat16)
                    grads.append(to_local_if_dtensor(g).detach())
            elif param.grad is not None:
                assert param.grad.type() == "torch.cuda.FloatTensor"
                grads.append(to_local_if_dtensor(param.grad).detach())
        if not grads:
            return
        coeff = clip_coefficient(total_norm, max_norm)
        runs = _coalesce(grads) if flat_runs else None
        scale_grads_(grads, coeff, runs=runs)

    get_grad_norm_fp32._primus_device_grad_clip = True
    clip_grad_by_total_norm_fp32._primus_device_grad_clip = True
    # optimizer.py binds both names at import time, so the module attributes alone are not enough.
    _clip_grads.get_grad_norm_fp32 = get_grad_norm_fp32
    _optimizer.get_grad_norm_fp32 = get_grad_norm_fp32
    _clip_grads.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    _optimizer.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32

    # train_step passes grad_norm through reduce_max_stat_across_model_parallel_group. Under pure DP Primus replaces
    # that with a pass-through that calls .item() on tensors; a DeviceGradNorm is not a tensor and passes as is.
    # With a model-parallel group the original builds a tensor from float(grad_norm): correct, one sync.
    orig_reduce = megatron_training.reduce_max_stat_across_model_parallel_group

    def reduce_max_stat_across_model_parallel_group(stat):
        if isinstance(stat, DeviceGradNorm) and _is_pure_dp(args):
            return stat
        return orig_reduce(stat)

    megatron_training.reduce_max_stat_across_model_parallel_group = reduce_max_stat_across_model_parallel_group

    # training_log consumes the norm only on the steps it prints or writes it; hand it a float there.
    from primus.backends.megatron.patches.training_log.training_log_args import (
        bind_training_log_args,
    )

    orig_training_log = megatron_training.training_log

    def training_log(*a, **k):
        bound = bind_training_log_args(a, k)
        if bound is not None and isinstance(bound.arguments.get("grad_norm"), DeviceGradNorm):
            if _consumes_grad_norm(
                args,
                bound.arguments["iteration"],
                bound.arguments.get("is_first_iteration", False),
                # Through the module attribute: MLPerf mode replaces it with a stub returning None.
                megatron_training.get_tensorboard_writer() is not None,
            ):
                bound.arguments["grad_norm"] = float(bound.arguments["grad_norm"])
                return orig_training_log(*bound.args, **bound.kwargs)
        return orig_training_log(*a, **k)

    megatron_training.training_log = training_log
    log_rank_0(f"[Patch:megatron.optimizer.device_grad_clip] gradient clipping on the device ({mode})")


def _is_pure_dp(args) -> bool:
    return getattr(args, "tensor_model_parallel_size", 1) == 1 and getattr(args, "pipeline_model_parallel_size", 1) == 1
