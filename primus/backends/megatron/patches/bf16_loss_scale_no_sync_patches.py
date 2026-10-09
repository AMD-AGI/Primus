###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Read the loss scale without a device-to-host sync when there is no grad scaler (``--bf16-loss-scale-no-sync``).

Megatron's training loop logs ``optimizer.get_loss_scale().item()`` after every step. Without a grad scaler (bf16)
the loss scale is the constant ``MixedPrecisionOptimizer._scale_one``, a one-element CUDA tensor holding 1.0, so the
``.item()`` copies a constant back to the host -- and because the copy is ordered on the compute stream, the host
waits for the whole optimizer step to finish before it can log, run the next step's data preparation or launch the
next forward.

With this patch ``get_loss_scale()`` returns that same tensor, viewed as a subclass whose ``item()`` returns the
host-side constant. Everything else about the tensor is unchanged (``__torch_function__`` is disabled, so arithmetic
on it -- ``scale_loss`` multiplies the loss by it -- returns ordinary tensors with the same values). The value is
verified against the device once, on first use.

Off by default. Enable with ``bf16_loss_scale_no_sync: true`` in the module config. A grad scaler (fp16) keeps the
original path.
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


class _HostKnownScale(torch.Tensor):
    """A view of a constant one-element CUDA tensor whose value the host already knows."""

    __torch_function__ = torch._C._disabled_torch_function_impl

    def item(self):
        return self._primus_host_value


@register_patch(
    "megatron.args.bf16_loss_scale_no_sync",
    backend="megatron",
    phase="setup",
    description="Expose --bf16-loss-scale-no-sync on Megatron's mixed precision argparse group.",
)
def patch_bf16_loss_scale_no_sync_arg(ctx: PatchContext) -> None:
    try:
        import megatron.training.arguments as margs
    except ImportError:
        return
    orig = getattr(margs, "_add_mixed_precision_args", None)
    if getattr(orig, "_primus_bf16_loss_scale_no_sync", False):
        return
    if orig is None:
        raise RuntimeError(
            "megatron.training.arguments._add_mixed_precision_args is missing, so --bf16-loss-scale-no-sync "
            "cannot be registered; update the patch rather than letting the gate silently do nothing."
        )

    def _add_mixed_precision_args(parser):
        parser = orig(parser)
        group = parser.add_argument_group(title="mixed precision")
        group.add_argument(
            "--bf16-loss-scale-no-sync",
            action="store_true",
            default=False,
            help="Without a grad scaler, return the constant loss scale with a host-side item() (no device sync).",
        )
        return parser

    _add_mixed_precision_args._primus_bf16_loss_scale_no_sync = True
    margs._add_mixed_precision_args = _add_mixed_precision_args
    log_rank_0("[Patch:megatron.args.bf16_loss_scale_no_sync] added --bf16-loss-scale-no-sync")


@register_patch(
    "megatron.optimizer.bf16_loss_scale_no_sync",
    backend="megatron",
    phase="before_train",
    description="Constant loss scale (no grad scaler) read on the host without a device sync.",
    condition=lambda ctx: bool(getattr(get_args(ctx), "bf16_loss_scale_no_sync", False)),
)
def patch_bf16_loss_scale_no_sync(ctx: PatchContext) -> None:
    from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer

    orig = MixedPrecisionOptimizer.get_loss_scale
    if getattr(orig, "_primus_bf16_loss_scale_no_sync", False):
        return

    def get_loss_scale(self):
        if self.grad_scaler is not None:
            return orig(self)
        scale = self.__dict__.get("_primus_host_known_scale")
        if scale is None:
            # One sync, once: the constant is created by Megatron as 1.0 and never written, but check it rather
            # than assume it.
            value = self._scale_one.item()
            if self._scale_one.numel() != 1 or value != 1.0:
                raise RuntimeError(f"bf16_loss_scale_no_sync: expected a constant loss scale of 1.0, found {value}")
            scale = self._scale_one.as_subclass(_HostKnownScale)
            scale._primus_host_value = value
            self._primus_host_known_scale = scale
        return scale

    get_loss_scale._primus_bf16_loss_scale_no_sync = True
    MixedPrecisionOptimizer.get_loss_scale = get_loss_scale
    log_rank_0("[Patch:megatron.optimizer.bf16_loss_scale_no_sync] constant loss scale read without a device sync")
