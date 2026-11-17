###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Scale clipped gradients with one fused multiply instead of one kernel per tensor.

WHY:
  AutoModel computes the global gradient norm in FP64 and then calls
  ``torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach=True)``
  once per sharding group. Two things defeat the foreach path there:

    * the gradients are FSDP2 DTensors, and
    * the clip coefficient inherits the norm's float64 dtype, while the
      gradients are BF16.

  ``torch._foreach_mul_`` then falls back to one non-vectorized ``mul_`` per
  tensor: one launch per parameter shard, far above the bandwidth floor of the
  multiply, and as many launches for the CPU to enqueue.

WHAT:
  Rebinds ``torch.nn.utils.clip_grads_with_norm_`` -- the name AutoModel looks up
  at call time -- to an equivalent that multiplies the gradients' local shards
  with a single ``torch._foreach_mul_`` per (device, dtype), passing the clamped
  coefficient cast to the gradients' dtype on the GPU. Scaling the local shard is exact for
  every DTensor placement: a Shard or Replicate local tensor is scaled like the
  whole, and a Partial one sums to the scaled whole. The formula is torch's,
  ``min(max_norm / (total_norm + 1e-6), 1)``, kept on the device so there is no
  host sync.

  The binding keeps torch's function and calls it whenever the setting is off, so
  a later job in the same process that turns the setting off gets torch's; it
  also calls it for ``foreach=False`` and for a DTensor ``total_norm``.

  The rebinding is process-wide (``torch.nn.utils`` is shared), but it only
  reaches callers that look the name up on ``torch.nn.utils`` at call time, as
  AutoModel's sharding-aware clip does (DTensor gradients, pipeline or expert
  parallelism). ``torch.nn.utils.clip_grad_norm_``, which AutoModel uses when no
  gradient is a DTensor, calls torch's private implementation, so DDP and
  single-GPU runs are unaffected, as is a name imported before the patch.
  The cause is PyTorch's per-tensor foreach fallback for DTensors; this patch can
  be removed once PyTorch handles that case.

  Model-agnostic: it touches no model and no recipe method. No AutoModel or
  PyTorch source is edited. Off by default; idempotent.

NUMERICS:
  Bit-identical to torch's result, clipped or not. On the GPU torch's per-tensor
  ``mul_`` rounds the 0-dim float64 coefficient to the gradient dtype, which is what
  this does explicitly; on the CPU ``mul_`` multiplies by it in FP32, so CPU
  gradients get the coefficient as torch passes it
  (``tests/unit_tests/backends/automodel/test_fast_grad_clip.py``).

Setting (``primus_train:`` section, off by default):
    fast_grad_clip: false      apply this patch
"""
from __future__ import annotations

import logging

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_INSTALLED_FLAG = "_primus_fast_grad_clip"
_SECTION = "primus_train"


def is_enabled() -> bool:
    """Whether the fused clip scaling should be applied."""
    return options.flag(f"{_SECTION}.fast_grad_clip")


def clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=None) -> None:
    """``torch.nn.utils.clip_grads_with_norm_`` with one fused multiply per (device, dtype).

    ``total_norm`` must be a plain tensor. On a device without foreach support it raises for
    ``foreach=True`` and otherwise multiplies each gradient on its own, as torch does.
    """
    import torch
    from torch.distributed.tensor import DTensor
    from torch.utils._foreach_utils import _device_has_foreach_support

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    with torch.no_grad():
        clip_coef = torch.clamp(float(max_norm) / (total_norm + 1e-6), max=1.0)
        groups: dict = {}
        for g in grads:
            local = g.to_local() if isinstance(g, DTensor) else g
            if local.numel():
                groups.setdefault((local.device, local.dtype), []).append(local)
        for (device, dtype), tensors in groups.items():
            # torch's mul_ reads a CPU coefficient as an FP32 scalar but casts a GPU one to the gradient
            # dtype; the fused GPU path needs the coefficient in the gradient dtype.
            coef = clip_coef.to(device) if device.type == "cpu" else clip_coef.to(device=device, dtype=dtype)
            if _device_has_foreach_support(device):
                torch._foreach_mul_(tensors, coef)
            elif foreach:
                raise RuntimeError(
                    f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
                )
            else:
                for t in tensors:
                    t.mul_(coef)


def install() -> bool:
    """Rebind torch.nn.utils.clip_grads_with_norm_ to the fused version while the setting is on."""
    import torch
    from torch.distributed.tensor import DTensor

    if getattr(torch.nn.utils, _INSTALLED_FLAG, False):
        return True
    original = torch.nn.utils.clip_grads_with_norm_

    def fast_clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=None):
        if not is_enabled() or foreach is False or isinstance(total_norm, DTensor):
            return original(parameters, max_norm, total_norm, foreach)
        return clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)

    torch.nn.utils.clip_grads_with_norm_ = fast_clip_grads_with_norm_
    setattr(torch.nn.utils, _INSTALLED_FLAG, True)
    logger.info(
        "[PrimusFastGradClip] gradient clipping now scales local shards with one fused "
        "multiply per dtype, coefficient in the gradient dtype."
    )
    return True
