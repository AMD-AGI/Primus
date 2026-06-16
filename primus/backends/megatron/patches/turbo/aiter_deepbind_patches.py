###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
gfx942 hd128 backward crash fix via scoped RTLD_DEEPBIND (ROCm/aiter#1332).

rocm/primus:v26.3's transformer_engine bundles a STALE vendored ``libmha_bwd.so``
(old aiter f299f579) that, via ``libck_fused_attn.so``'s DT_NEEDED, interposes the
global symbol ``aiter::mha_bwd`` over the freshly pinned aiter (b5e03ed1) -> Turbo's
hd128 backward mis-selects the swa variant and launches with gridDim.x=0 -> 8-rank
SIGABRT. The ``mha_bwd_args`` ABI differs between the two aiter revisions, so a
*global* fix (RTLD_GLOBAL preload / .so overwrite) repairs Turbo but breaks the TE
path (TE's libck calls the stale libmha by the old ABI) -- isolation must be scoped.

Fix: wrap ``importlib.import_module`` (aiter loads its mha extensions through it,
honouring ``sys.setdlopenflags()``) and OR in ``RTLD_DEEPBIND`` only for the pinned
aiter mha modules (``module_fmha_v3_bwd`` / ``mha_bwd_bf16_*``), so they bind their
own fresh ``aiter::mha_bwd`` while TE keeps the stale one. DEEPBIND is scoped to these
few extensions, leaving torch/c10 untouched. Installed by the ``before_train`` patch
below, gated on Turbo + gfx942 (TE path never touched); ``before_train`` runs well
before the first backward that JIT-imports these modules.

TEMPORARY: delete this file (and its turbo/__init__.py entry) once the base image's
TE is built against the updated aiter.
"""

import importlib
import os
import sys

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

_LOG_PREFIX = "[Patch:megatron.turbo.aiter_deepbind]"
_ENV_SWITCH = "PRIMUS_AITER_DEEPBIND"
_TARGET_MODULE_SUBSTRINGS = ("module_fmha_v3_bwd", "mha_bwd_bf16")
# Marker so the import_module wrap is idempotent.
_WRAPPED_ATTR = "_primus_aiter_deepbind_wrapped"
# Process-wide guard so the per-import confirmation is logged at most once.
_HIT_LOGGED = False


def _enabled() -> bool:
    """ON by default; set PRIMUS_AITER_DEEPBIND=0 to disable."""
    return os.environ.get(_ENV_SWITCH, "1").strip().lower() not in ("0", "false", "no", "off")


def _is_gfx942() -> bool:
    """True only on gfx942 (MI300X/MI325X), where the stale-libmha crash reproduces."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return "gfx942" in (torch.cuda.get_device_properties(0).gcnArchName or "")
    except Exception:  # pragma: no cover - defensive, never block training
        return False


def _install_deepbind_import_hook() -> bool:
    """
    Idempotently wrap ``importlib.import_module`` to OR in RTLD_DEEPBIND for the pinned
    aiter mha extensions only (torch and every other import are untouched).

    Returns True if the wrapper is in place (installed now or already), False if it is
    a no-op (disabled via env / RTLD_DEEPBIND unsupported) so the caller logs honestly.
    """
    if not _enabled():
        return False

    original_import_module = importlib.import_module
    if getattr(original_import_module, _WRAPPED_ATTR, False):
        return True

    deepbind_flag = getattr(os, "RTLD_DEEPBIND", 0)
    if not deepbind_flag:  # non-Linux / no DEEPBIND support
        return False

    def _import_module_with_deepbind(name, package=None):
        if not any(sub in name for sub in _TARGET_MODULE_SUBSTRINGS):
            return original_import_module(name, package)

        prev_flags = sys.getdlopenflags()
        sys.setdlopenflags(prev_flags | deepbind_flag)
        try:
            return original_import_module(name, package)
        finally:
            sys.setdlopenflags(prev_flags)
            global _HIT_LOGGED
            if not _HIT_LOGGED:
                _HIT_LOGGED = True
                log_rank_0(
                    f"{_LOG_PREFIX} RTLD_DEEPBIND applied for '{name}' "
                    f"(flags 0x{prev_flags:x} -> 0x{prev_flags | deepbind_flag:x}); isolates "
                    "pinned aiter::mha_bwd from TE's stale libmha (ROCm/aiter#1332)."
                )

    setattr(_import_module_with_deepbind, _WRAPPED_ATTR, True)
    importlib.import_module = _import_module_with_deepbind
    return True


def _can_install_aiter_deepbind(ctx: PatchContext) -> bool:
    """Install only on gfx942 with the Turbo attention path active."""
    args = get_args(ctx)
    if not bool(getattr(args, "use_turbo_attention", False)):
        return False
    if not is_primus_turbo_can_patch(ctx):
        return False
    if not _is_gfx942():
        log_rank_0(f"{_LOG_PREFIX} device is not gfx942; aiter DEEPBIND isolation not needed.")
        return False
    return True


@register_patch(
    "megatron.turbo.aiter_deepbind",
    backend="megatron",
    phase="before_train",
    description=(
        "On gfx942, install the aiter mha RTLD_DEEPBIND import hook so the Turbo attention "
        "backward binds the pinned aiter::mha_bwd instead of TE's stale vendored libmha. "
        "Gated on Turbo attention (TE path untouched). Temporary workaround (ROCm/aiter#1332)."
    ),
    condition=_can_install_aiter_deepbind,
)
def patch_install_aiter_deepbind(ctx: PatchContext) -> None:
    """Install the import-time RTLD_DEEPBIND wrapper for the pinned aiter mha extensions."""
    if _install_deepbind_import_hook():
        log_rank_0(
            f"{_LOG_PREFIX} aiter DEEPBIND import hook installed (PRIMUS_AITER_DEEPBIND=0 to disable)."
        )
    else:
        log_rank_0(f"{_LOG_PREFIX} aiter DEEPBIND import hook NOT installed (disabled or unsupported).")
