###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Keep selected parameters in FP32 across the framework's half-precision cast.

``Float16Module`` wraps the model with a blanket ``module.bfloat16()`` (or
``.half()``), which walks every floating-point parameter. Two DeepSeek-V4
parameters are FP32 in the released checkpoint and both feed a softmax directly:
the compressor's ``ape`` (an additive bias on the pooling scores) and the
attention's ``attn_sink`` (an extra logit in the denominator). This module can
hold them at FP32 through the cast: mark the parameter, mix
:class:`KeepInFp32Mixin` into its owner, and the marked entries are restored to
FP32 after any ``_apply`` (which is what ``.bfloat16()``, ``.half()``, ``.to()``
and ``.cuda()`` all funnel through).

**Off by default**, because holding a parameter at a dtype the rest of the model
does not use is not free downstream. ``ParamAndGradBuffer`` does key its buffers
on ``(param_dtype, grad_dtype)`` and allocate one per distinct combination, but
the distributed optimizer on top of it does not follow: ``distrib_optimizer.py``
carries five ``assert len(gbuf_range_maps) == 1, "single dtype supported, for
now."`` guards. Enabling this on a 4-node / PP4 / EP8 run with
``use_precision_aware_optimizer`` + ``store_param_remainders`` aborted every GPU
with ``Memory access fault`` on the first training step, while the same build
with the mechanism off completed cleanly. Single-node PP1 does not reproduce it,
so this cannot be validated by the unit tests or a one-node soak.

Leaving it off costs very little. ``store_param_remainders`` keeps FP32 master
params in the optimizer either way, so update precision is unchanged; every
consumer of these two parameters already promotes at the use site
(``sink.float()`` in the sparse-MLA adapter and the eager reference,
``score.float()`` before the pooling softmax), so forward precision is unchanged
too. What the mark actually buys is the stored resolution matching the released
checkpoint -- relevant to checkpoint parity, not to training correctness.

Set ``PRIMUS_V4_KEEP_FP32=1`` to turn it on, but only with PP1 or after
confirming the optimizer path tolerates two parameter dtypes.
"""

from __future__ import annotations

import os

import torch

__all__ = [
    "ENABLE_ENV_VAR",
    "KeepInFp32Mixin",
    "is_enabled",
    "is_marked_keep_in_fp32",
    "mark_keep_in_fp32",
    "unmark_keep_in_fp32",
]

ENABLE_ENV_VAR = "PRIMUS_V4_KEEP_FP32"

_MARK = "_primus_keep_in_fp32"


def is_enabled() -> bool:
    """Whether the keep-in-FP32 contract is active. Off unless opted in.

    ``PRIMUS_V4_KEEP_FP32=1`` enables it; see the module docstring for why the
    default is off and what the two parameters lose by following the model dtype
    (stored resolution only -- not update or forward precision).
    """
    return os.environ.get(ENABLE_ENV_VAR, "0") == "1"


def mark_keep_in_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Mark ``tensor`` so its owning :class:`KeepInFp32Mixin` keeps it FP32."""
    if not is_enabled():
        return tensor
    setattr(tensor, _MARK, True)
    return tensor


def unmark_keep_in_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Drop the mark, letting ``tensor`` follow the model dtype again."""
    if hasattr(tensor, _MARK):
        delattr(tensor, _MARK)
    return tensor


def is_marked_keep_in_fp32(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is pinned to FP32."""
    return bool(getattr(tensor, _MARK, False))


class KeepInFp32Mixin:
    """Restore marked parameters to FP32 after any ``_apply``.

    Must come before ``nn.Module`` in the MRO so ``super()._apply`` reaches the
    normal implementation. ``_apply`` runs once per module per conversion (and
    the tensors involved here are tiny), so the save/restore cost is noise.
    """

    def _apply(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        pinned = {
            name: param.detach().to(torch.float32).clone()
            for name, param in self._parameters.items()  # type: ignore[attr-defined]
            if param is not None and is_marked_keep_in_fp32(param)
        }

        module = super()._apply(fn, *args, **kwargs)  # type: ignore[misc]

        for name, original in pinned.items():
            param = module._parameters.get(name)
            if param is None:
                continue
            if param.dtype != torch.float32:
                # ``fn`` may have produced a fresh Parameter, so restore from
                # the saved FP32 copy rather than casting the downgraded values
                # back (which would keep the BF16 rounding).
                param.data = original.to(device=param.device)
            # ``_apply`` can replace the Parameter object outright, dropping
            # custom attributes; re-mark so the next conversion is protected.
            mark_keep_in_fp32(param)

        return module
