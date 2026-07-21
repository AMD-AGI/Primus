###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
hd128 backward crash fix via scoped RTLD_DEEPBIND (ROCm/aiter#1332).

Affects gfx942 (MI300X/MI325X) and gfx950 (MI350X/MI355X).

rocm/primus:v26.3's transformer_engine bundles a STALE vendored ``libmha_bwd.so``
(old aiter f299f579) that, via ``libck_fused_attn.so``'s DT_NEEDED, interposes the
global symbol ``aiter::mha_bwd`` over the freshly pinned aiter (b5e03ed1) -> Turbo's
hd128 backward mis-selects the swa variant and launches with an invalid grid config
(gridDim.x=0 on gfx942 / "invalid configuration argument" on gfx950) -> 8-rank crash.
The ``mha_bwd_args`` ABI differs between the two aiter revisions, so a
*global* fix (RTLD_GLOBAL preload / .so overwrite) repairs Turbo but breaks the TE
path (TE's libck calls the stale libmha by the old ABI) -- isolation must be scoped.

Fix: wrap ``importlib.import_module`` (aiter loads its mha extensions through it,
honouring ``sys.setdlopenflags()``) and OR in ``RTLD_DEEPBIND`` only for the pinned
aiter mha modules (``module_fmha_v3_bwd`` / ``mha_bwd_bf16_*``), so they bind their
own fresh ``aiter::mha_bwd`` while TE keeps the stale one. DEEPBIND is scoped to these
few extensions, leaving torch/c10 untouched. Installed by the ``before_train`` patch
below, gated on Turbo + affected arch (TE path never touched); ``before_train`` runs
well before the first backward that JIT-imports these modules.

Self-detection: skipped once ``transformer_engine.__version__`` >= 2.14, which renamed
the vendored library and stopped exporting the colliding ``aiter::mha_bwd`` symbol --
confirmed fixed in rocm/primus:v26.4. Older images (e.g. v26.3) still get the patch.

TEMPORARY: delete this file (and its turbo/__init__.py entry) once every base image
still in support ships a TE built against the updated aiter.
"""

import importlib
import os
import re
import sys

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

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


# GPU arches where the stale-libmha hd128 backward crash reproduces:
#   gfx942 -> MI300X / MI325X
#   gfx950 -> MI350X / MI355X
_AFFECTED_ARCHS = ("gfx942", "gfx950")


def _is_affected_arch() -> bool:
    """True on the arches where the stale-libmha crash reproduces (gfx942 / gfx950)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        arch = torch.cuda.get_device_properties(0).gcnArchName or ""
        return any(affected in arch for affected in _AFFECTED_ARCHS)
    except Exception:  # pragma: no cover - defensive, never block training
        return False


# First TE version that dropped the stale vendored libmha_bwd.so (see module docstring).
_TE_FIX_VERSION = (2, 14, 0)
_TE_FIX_VERSION_STR = ".".join(map(str, _TE_FIX_VERSION))


def _te_version_tuple():
    """Best-effort (major, minor, patch) from ``transformer_engine.__version__``, or
    None if TE isn't importable / the version string can't be parsed."""
    try:
        import transformer_engine

        match = re.match(r"(\d+)\.(\d+)\.(\d+)", getattr(transformer_engine, "__version__", "") or "")
        return tuple(int(part) for part in match.groups()) if match else None
    except Exception:  # pragma: no cover - defensive, never block training
        return None


def _te_already_fixed() -> bool:
    """True if the installed TE no longer vendors the stale libmha_bwd.so.

    Fails safe towards False (unfixed) on an unknown/unparseable version: applying
    the hook on an already-fixed TE is a harmless no-op, but skipping it on an
    unfixed TE reintroduces the hd128 backward crash.
    """
    version = _te_version_tuple()
    return version is not None and version >= _TE_FIX_VERSION


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
    """Install only on affected arches (gfx942/gfx950) with an unfixed TE and Turbo attention active."""
    args = get_args(ctx)
    if not bool(getattr(args, "use_turbo_attention", False)):
        return False
    if not is_primus_turbo_can_patch(ctx):
        return False
    if not _is_affected_arch():
        log_rank_0(
            f"{_LOG_PREFIX} device is not an affected arch ({', '.join(_AFFECTED_ARCHS)}); "
            "aiter DEEPBIND isolation not needed."
        )
        return False
    if _te_already_fixed():
        log_rank_0(
            f"{_LOG_PREFIX} transformer_engine >= {_TE_FIX_VERSION_STR} detected "
            "(no vendored stale libmha_bwd.so exported as aiter::mha_bwd); aiter DEEPBIND "
            "isolation not needed on this image."
        )
        return False
    return True


@register_patch(
    "megatron.turbo.aiter_deepbind",
    backend="megatron",
    phase="before_train",
    description=(
        "On gfx942/gfx950, install the aiter mha RTLD_DEEPBIND import hook so the Turbo "
        "attention backward binds the pinned aiter::mha_bwd instead of TE's stale vendored "
        "libmha. Gated on Turbo attention and an unfixed TE (auto-skips once "
        f"transformer_engine >= {_TE_FIX_VERSION_STR}). Temporary workaround (ROCm/aiter#1332)."
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
