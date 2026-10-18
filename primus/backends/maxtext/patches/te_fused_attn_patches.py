###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TransformerEngine fused-attention backward attribute patch

Some ROCm TransformerEngine builds (seen with 2.19.0.dev0+rocm10.2) read a ``return_max_logit`` attribute in the
C++ backward fused-attention FFI handler, but the Python lowering of ``FusedAttnBwdPrimitive`` only passes it to the
forward call. The handler then falls back to ``False`` and prints a four-line
"Failure in getting attribute value of 'return_max_logit'" warning on every call: once per attention layer, per GPU,
per step, which is most of a MaxText training log.

This patch adds ``return_max_logit=False`` to the backward FFI lowering when it is missing, which is the value the
handler already falls back to, so numerics are unchanged. It only activates when the installed TE's backward
lowering does not pass the attribute, so it becomes a no-op once TE is fixed.

MaxText imports TransformerEngine lazily, and TE's import order matters on this stack, so the patch does not import
TE itself: it installs an import hook and patches the module when MaxText first imports it.
"""

import importlib.abc
import inspect
import sys

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_TARGET = "transformer_engine.jax.cpp_extensions.attention"
_BWD_FFI = "te_fused_attn_backward_ffi"
_ATTR = "return_max_logit"
_PREFIX = "[Patch:maxtext.te_fused_attn_bwd_attr]"


def _needs_patch(module) -> bool:
    prim = getattr(module, "FusedAttnBwdPrimitive", None)
    if prim is None or getattr(prim, "name", None) != _BWD_FFI:
        return False
    try:
        return _ATTR not in inspect.getsource(prim.lowering)
    except (OSError, TypeError):
        return False


class _FfiShim:
    """Forwards to ``jax.ffi``; adds the missing attribute to the backward fused-attention lowering."""

    def __init__(self, ffi):
        self._ffi = ffi

    def __getattr__(self, name):
        return getattr(self._ffi, name)

    def ffi_lowering(self, call_target_name, *args, **kwargs):
        rule = self._ffi.ffi_lowering(call_target_name, *args, **kwargs)
        if call_target_name != _BWD_FFI:
            return rule

        def lowering(ctx, *operands, **attrs):
            attrs.setdefault(_ATTR, False)
            return rule(ctx, *operands, **attrs)

        return lowering


def _patch_module(module) -> None:
    if isinstance(getattr(module, "ffi", None), _FfiShim):
        return
    if not _needs_patch(module):
        log_rank_0(f"{_PREFIX} TE's backward fused-attention lowering already passes '{_ATTR}'; not patched.")
        return
    module.ffi = _FfiShim(module.ffi)
    log_rank_0(f"{_PREFIX} Added '{_ATTR}=False' to TE's backward fused-attention FFI lowering.")


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Patches ``_TARGET`` right after it is executed, without changing when it is imported."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                _patch_module(module)
            except Exception as exc:  # noqa: BLE001 - a failed cosmetic patch must not abort training
                warning_rank_0(f"{_PREFIX} Failed to patch {_TARGET}: {exc!r}")

        loader.exec_module = exec_and_patch
        return spec


@register_patch(
    patch_id="maxtext.te_fused_attn_bwd_attr",
    backend="maxtext",
    phase="setup",
    description="Pass return_max_logit to TE's backward fused-attention FFI (silences a per-call warning)",
    condition=lambda ctx: True,
)
def patch_te_fused_attn_bwd_attr(ctx: PatchContext) -> None:
    del ctx
    module = sys.modules.get(_TARGET)
    if module is not None:
        _patch_module(module)
        return
    if not any(isinstance(f, _PostImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportHook())
    log_rank_0(f"{_PREFIX} Import hook installed for {_TARGET}.")
