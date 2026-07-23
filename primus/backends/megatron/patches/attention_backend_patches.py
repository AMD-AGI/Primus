###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Make megatron-core's attention-backend selection ROCm-safe.

``LanguageModule._set_attention_backend()`` reconciles the chosen
``attention_backend`` with the ``NVTE_*_ATTN`` env vars by *asserting* they are
unset-or-equal to what the backend wants, then setting them. Primus ROCm images
intentionally bake ``NVTE_FLASH_ATTN=0`` / ``NVTE_FUSED_ATTN=1`` (TE flash attn
is unavailable on ROCm; the fused/CK path is used instead), so stock megatron
assert-crashes before training starts on any model that goes through this probe
(Mamba, hybrid, non-Turbo GPT, ...):

* ``auto`` wants all three = 1, but the baked ``NVTE_FLASH_ATTN=0`` trips it.
* an explicit backend (e.g. ``unfused``) wants ``NVTE_FUSED_ATTN=0``, but the
  baked ``NVTE_FUSED_ATTN=1`` trips it.

On ROCm the baked vars are image defaults, not user intent, so the selected
backend should *win* over them rather than assert against them:

* ``auto`` -> enable every backend the platform hasn't explicitly disabled
  (fill only the *unset* vars, so the baked ``NVTE_FLASH_ATTN=0`` is respected).
* an explicit backend -> force exactly its ``NVTE_*_ATTN`` combination,
  overriding whatever the image baked in.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_NVTE_ATTN_ENVS = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")


def _is_rocm(ctx: PatchContext) -> bool:
    import torch

    return getattr(torch.version, "hip", None) is not None


@register_patch(
    "megatron.attention_backend.rocm_safe",
    backend="megatron",
    phase="before_train",
    description="Let attention_backend override ROCm's baked NVTE_*_ATTN instead of asserting",
    condition=_is_rocm,
)
def patch_attention_backend(ctx: PatchContext):
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )
    from megatron.core.transformer.enums import AttnBackend

    if getattr(LanguageModule, "_primus_attention_backend_patched", False):
        return

    original_set_attention_backend = LanguageModule._set_attention_backend

    # (NVTE_FLASH_ATTN, NVTE_FUSED_ATTN, NVTE_UNFUSED_ATTN) for each explicit backend.
    explicit_envs = {
        AttnBackend.flash: ("1", "0", "0"),
        AttnBackend.fused: ("0", "1", "0"),
        AttnBackend.unfused: ("0", "0", "1"),
        AttnBackend.local: ("0", "0", "0"),
    }

    def _set_attention_backend(self):
        backend = self.config.attention_backend
        if backend == AttnBackend.auto:
            for name in _NVTE_ATTN_ENVS:
                os.environ.setdefault(name, "1")
            return
        values = explicit_envs.get(backend)
        if values is None:
            original_set_attention_backend(self)
            return
        for name, value in zip(_NVTE_ATTN_ENVS, values):
            os.environ[name] = value

    LanguageModule._set_attention_backend = _set_attention_backend
    LanguageModule._primus_attention_backend_patched = True
    log_rank_0(
        "[Patch:megatron.attention_backend.rocm_safe] attention_backend now overrides baked NVTE_*_ATTN"
    )
