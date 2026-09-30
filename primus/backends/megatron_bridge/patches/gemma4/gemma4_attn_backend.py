###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-backend attention selection for Gemma 4, beyond what ``AttnBackend`` allows.

``config.attention_backend`` is not a per-layer setting. megatron-core
implements it in ``LanguageModule._set_attention_backend`` by writing three
process-global environment variables once at model init::

    local    -> NVTE_FLASH_ATTN=0  NVTE_FUSED_ATTN=0  NVTE_UNFUSED_ATTN=0
    flash    -> 1 0 0
    fused    -> 0 1 0
    unfused  -> 0 0 1
    auto     -> 1 1 1

TransformerEngine then picks a backend per invocation from the actual tensor
shapes, subject to those flags. That has two consequences specific to Gemma 4,
whose layers interleave two attention geometries 5:1:

  * sliding layers -- head_dim 256 -- are eligible for flash.
  * global layers  -- head_dim 512 -- are not. FlashAttention 2 and 3 both cap
    ``head_dim_qk`` at 256, so the global layers can only run fused or unfused.

So ``attention_backend=flash`` does not accelerate Gemma 4, it **breaks** it: it
sets ``NVTE_UNFUSED_ATTN=0`` and leaves the global layers with no legal backend.
And ``auto`` already gives the desired hybrid -- flash on the sliding 5/6,
fallback on the global 1/6 -- so there is nothing to gain by being more
specific.

What the enum *cannot* express is any combination of two flags. The one worth
having is::

    NVTE_FLASH_ATTN=1  NVTE_FUSED_ATTN=0  NVTE_UNFUSED_ATTN=1

which keeps flash on the sliding layers while excluding the fused path
entirely. That matters here because ``auto`` (all three enabled) hung the GPU on
its first training step, and since flash is ineligible for the head_dim-512
global layers, the fused backend is the prime suspect for what ``auto`` selected
there. This combination isolates it while giving up nothing on the 5/6 of layers
that can use flash.

Usage::

    PRIMUS_GEMMA4_NVTE_ATTN=flash,unfused     # the hybrid described above
    PRIMUS_GEMMA4_NVTE_ATTN=unfused           # equivalent to attention_backend=unfused
    PRIMUS_GEMMA4_NVTE_ATTN=flash,fused,unfused

The values are applied *after* ``_set_attention_backend`` runs, so megatron's
own assertion (which rejects pre-set environment variables that disagree with
``attention_backend``) is not tripped.
"""

from __future__ import annotations

import os
from typing import Optional

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_ENV = "PRIMUS_GEMMA4_NVTE_ATTN"
_PATCHED_ATTR = "_primus_gemma4_nvte_attn_patched"

_FLAGS = {
    "flash": "NVTE_FLASH_ATTN",
    "fused": "NVTE_FUSED_ATTN",
    "unfused": "NVTE_UNFUSED_ATTN",
}


def _parse(spec: str) -> Optional[dict]:
    wanted = {name.strip().lower() for name in spec.split(",") if name.strip()}
    unknown = wanted - set(_FLAGS)
    if unknown:
        log_rank_0(
            f"[Patch:gemma4.attn.nvte_backend] Ignoring {_ENV}={spec!r}: "
            f"unknown backend(s) {sorted(unknown)}; valid are {sorted(_FLAGS)}"
        )
        return None
    if not wanted:
        return None
    return {var: ("1" if key in wanted else "0") for key, var in _FLAGS.items()}


@register_patch(
    "gemma4.attn.nvte_backend",
    backend="megatron_bridge",
    phase="setup",
    description="Opt-in: set the NVTE attention flags directly, for combinations AttnBackend cannot express",
)
def patch_gemma4_nvte_attn_backend(ctx: PatchContext) -> None:
    spec = os.environ.get(_ENV, "").strip()
    if not spec:
        return

    desired = _parse(spec)
    if desired is None:
        return

    try:
        from megatron.core.models.common.language_module.language_module import LanguageModule
    except Exception:
        log_rank_0("[Patch:gemma4.attn.nvte_backend] megatron-core LanguageModule unavailable")
        return

    original = LanguageModule._set_attention_backend
    if getattr(original, _PATCHED_ATTR, False):
        return

    def _set_attention_backend(self):
        # Let megatron do its own thing first (and keep its assertion meaningful),
        # then overwrite. TE reads these at backend-selection time, not here, so a
        # later write still takes effect.
        original(self)
        for var, value in desired.items():
            os.environ[var] = value
        log_rank_0(
            "[Patch:gemma4.attn.nvte_backend] "
            + " ".join(f"{var}={value}" for var, value in sorted(desired.items()))
            + f"  (from {_ENV}={spec})"
        )

    setattr(_set_attention_backend, _PATCHED_ATTR, True)
    LanguageModule._set_attention_backend = _set_attention_backend
    log_rank_0(f"[Patch:gemma4.attn.nvte_backend] Will force NVTE attention flags from {_ENV}={spec}")
