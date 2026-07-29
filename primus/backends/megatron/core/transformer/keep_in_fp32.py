###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Keep selected parameters in FP32 across the framework's half-precision cast.

``Float16Module`` wraps the model with a blanket ``module.bfloat16()`` (or
``.half()``), which walks every floating-point parameter. A few DeepSeek-V4
parameters must not follow: the compressor's ``ape`` and the attention's
``attn_sink`` are FP32 in the released checkpoint, and both feed a softmax
directly -- ``ape`` as an additive bias on the pooling scores, ``attn_sink`` as
an extra logit in the denominator -- so storing them at BF16 resolution throws
away trainable precision exactly where it is least affordable.

Newer upstream releases solve this with a ``mark_keep_in_fp32`` helper plus
support inside ``Float16Module``. The version pinned here has neither, so the
same contract is implemented from the module side: mark the parameter, mix
:class:`KeepInFp32Mixin` into its owner, and the marked entries are restored to
FP32 after any ``_apply`` (which is what ``.bfloat16()``, ``.half()``, ``.to()``
and ``.cuda()`` all funnel through).

Mixed parameter dtypes are fine downstream: ``ParamAndGradBuffer`` keys its
buffers on ``param_dtype`` and allocates one per distinct dtype, so the FP32
stragglers simply land in their own bucket.
"""

from __future__ import annotations

import torch

__all__ = [
    "KeepInFp32Mixin",
    "is_marked_keep_in_fp32",
    "mark_keep_in_fp32",
    "unmark_keep_in_fp32",
]

_MARK = "_primus_keep_in_fp32"


def mark_keep_in_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Mark ``tensor`` so its owning :class:`KeepInFp32Mixin` keeps it FP32."""
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
