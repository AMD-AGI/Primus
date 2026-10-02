###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fuse global gradient clipping into Transformer Engine FusedAdam.

This is an opt-in runtime patch for experiments. Set
``PRIMUS_FUSED_ADAM_CLIP=1`` to keep Megatron's global-norm calculation but
defer application of the resulting clip coefficient to TE's Adam kernel.

The installed TE already exposes its capturable Adam entry point, which reads
an inverse loss scale from a device scalar and multiplies every gradient while
it is resident in registers. Reusing that entry point lets this patch remove
the standalone ``multi_tensor_scale`` pass without rebuilding TE. The current
phase intentionally leaves Megatron's norm ``.item()`` in place; removing that
host synchronization is a separate change once the fused path is validated.
"""

from __future__ import annotations

import os
from typing import Dict

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.te_fused_adam_clip"


def _enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_ADAM_CLIP", "0") == "1"


def _clip_loss_scale(total_norm: float, max_norm: float) -> float:
    """Return the synthetic loss scale whose reciprocal is the clip factor."""

    clip_coeff = min(1.0, float(max_norm) / (float(total_norm) + 1.0e-6))
    return 1.0 / clip_coeff


class _ClipGradScaler:
    """Small adapter for TE's capturable Adam inverse-scale interface."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer

    def _get_scale_async(self) -> torch.Tensor:
        return self.optimizer._primus_clip_loss_scale

    def _check_inf_per_device(self, optimizer) -> Dict[torch.device, torch.Tensor]:
        del optimizer
        found_inf = self.optimizer._primus_clip_found_inf
        devices = {
            param.device
            for group in self.optimizer.param_groups
            for param in group["params"]
        }
        return {device: found_inf for device in devices}


def _install_patch() -> None:
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from transformer_engine.pytorch.optimizers import FusedAdam

    if is_patched(FusedAdam, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] already installed; skipping.")
        return

    original_adam_init = FusedAdam.__init__
    original_adam_step = FusedAdam.step
    original_scheduler_step = OptimizerParamScheduler.step

    def patched_adam_init(self, *args, **kwargs):
        requested_capturable = kwargs.get("capturable", False)
        if requested_capturable is not False:
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP owns TE FusedAdam capturable mode; "
                f"unexpected capturable={requested_capturable!r}"
            )
        kwargs["capturable"] = True
        original_adam_init(self, *args, **kwargs)
        device = next(
            param.device
            for group in self.param_groups
            for param in group["params"]
        )
        self._primus_clip_loss_scale = torch.ones(1, dtype=torch.float32, device=device)
        self._primus_clip_found_inf = torch.zeros(1, dtype=torch.float32, device=device)
        self._primus_clip_grad_scaler = _ClipGradScaler(self)

    def patched_adam_step(self, closure=None, grad_scaler=None):
        if grad_scaler is not None:
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP does not currently compose with an AMP GradScaler"
            )
        return original_adam_step(
            self,
            closure=closure,
            grad_scaler=self._primus_clip_grad_scaler,
        )

    def patched_clip_grad_norm(self, clip_grad: float) -> float:
        # Preserve Megatron's exact norm calculation and collective behavior,
        # but do not launch clip_grad_by_total_norm_fp32's scale pass.
        grad_norm = self.get_grad_norm()
        inner_optimizer = self.optimizer
        if not isinstance(inner_optimizer, FusedAdam):
            raise RuntimeError(
                "PRIMUS_FUSED_ADAM_CLIP requires Transformer Engine FusedAdam, got "
                f"{type(inner_optimizer).__module__}.{type(inner_optimizer).__name__}"
            )
        loss_scale = _clip_loss_scale(grad_norm, clip_grad)
        inner_optimizer._primus_clip_loss_scale.fill_(loss_scale)
        return grad_norm

    def patched_scheduler_step(self, increment: int) -> None:
        # Megatron normally replaces param_group['lr'] with a Python float on
        # every scheduler update. Capturable Adam requires the constructor's
        # persistent device tensor, so retain it and update its value in place.
        lr_tensors = [
            group.get("lr") if torch.is_tensor(group.get("lr")) else None
            for group in self.optimizer.param_groups
        ]
        original_scheduler_step(self, increment)
        for group, lr_tensor in zip(self.optimizer.param_groups, lr_tensors):
            if lr_tensor is not None:
                lr_tensor.fill_(float(group["lr"]))
                group["lr"] = lr_tensor

    FusedAdam.__init__ = patched_adam_init
    FusedAdam.step = patched_adam_step
    MegatronOptimizer.clip_grad_norm = patched_clip_grad_norm
    OptimizerParamScheduler.step = patched_scheduler_step

    mark_patched(FusedAdam, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] enabled: global clip coefficient is applied inside "
        "TE capturable FusedAdam; standalone multi_tensor_scale is disabled."
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Fuse global gradient clipping into Transformer Engine FusedAdam",
    priority=39,
    condition=lambda ctx: _enabled(),
)
def patch_te_fused_adam_clip(ctx: PatchContext) -> None:
    del ctx
    _install_patch()
