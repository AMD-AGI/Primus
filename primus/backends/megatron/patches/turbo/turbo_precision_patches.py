###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo-only precision (``turbo_precision: mxfp8``).

Megatron runs without ``--fp8`` in this mode and the Primus-Turbo linears quantize themselves (see
the ``turbo_precision`` block in ``core/extensions/primus_turbo.py``). With more than one
microbatch per step those linears cache their quantized weight on the first microbatch and reuse it
for the rest. Megatron re-arms that cache through ``MegatronModule.set_is_first_microbatch()``, which
does nothing unless ``config.fp8`` / ``config.fp4`` is set, so without this patch every step after
the first would reuse the weights quantized at step 1.
"""

import os

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _turbo_precision_value(ctx: PatchContext):
    value = os.environ.get("PRIMUS_TURBO_PRECISION")
    if value is None:
        value = getattr(get_args(ctx), "turbo_precision", None)
    if value is None:
        return None
    value = str(value).strip().lower()
    return None if value in ("", "none", "null", "bf16") else value


def _is_turbo_precision_enabled(ctx: PatchContext) -> bool:
    return _turbo_precision_value(ctx) is not None and is_primus_turbo_can_patch(ctx)


@register_patch(
    "megatron.turbo.turbo_precision_first_microbatch",
    backend="megatron",
    phase="before_train",
    description="Re-arm the Turbo quantized-weight cache every step when Megatron runs without --fp8",
    condition=_is_turbo_precision_enabled,
)
def patch_turbo_precision_first_microbatch(ctx: PatchContext):
    from megatron.core.transformer.module import MegatronModule

    original = MegatronModule.set_is_first_microbatch

    def set_is_first_microbatch(self):
        original(self)
        # Same as Megatron's fp8 branch: flag every module that keeps a first-microbatch cache.
        modules = getattr(self, "_turbo_modules_with_is_first_microbatch", None)
        if modules is None:
            modules = [m for m in self.modules() if hasattr(m, "is_first_microbatch")]
            self._turbo_modules_with_is_first_microbatch = modules
        for m in modules:
            m.is_first_microbatch = True

    MegatronModule.set_is_first_microbatch = set_is_first_microbatch
    log_rank_0(
        f"[Patch:megatron.turbo.turbo_precision_first_microbatch] turbo_precision="
        f"{_turbo_precision_value(ctx)}: set_is_first_microbatch() now re-arms the Turbo weight cache "
        "without --fp8"
    )
