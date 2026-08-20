###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Triton buffer-store WAR miscompile workaround
=============================================

The AMD Triton backend can emit a ``buffer_store_dwordx4`` whose data VGPRs are
redefined by a later instruction with no intervening ``s_waitcnt vmcnt``, so the
store writes the clobbering value instead of the computed one.  The corruption
is silent: it drops Inf/NaN on TorchTitan and stays inside the normal numeric
range on Megatron, so a finite loss does not prove a clean run.

``AMDGCN_USE_BUFFER_OPS=0`` avoids it globally but costs ~18% on MoE recipes,
because the kernels that spill without buffer addressing (the Primus-Turbo
grouped-GEMM ones) are not the kernels that miscompile.  So compile every
kernel normally, look for the hazard in the emitted AMDGCN, and recompile only
the affected kernels with buffer ops off.

Keying on the machine code means neither a kernel nor an architecture allowlist
has to be maintained, and it follows the toolchain: every Triton build tried so
far (upstream 3.7.0, ROCm 3.7.0/3.7.1/3.8.0) still emits the pattern on at least
one production kernel.
"""

import functools
import os
import re
import sys

from primus.core.patches.context import PatchContext
from primus.core.patches.patch_registry import register_patch
from primus.core.utils.module_utils import log_rank_0

# Bump whenever the detection below changes, so that torch.compile caches
# populated by an older guard are not reused.  See _break_inductor_caches.
_GUARD_VERSION = "1"

_STORE = re.compile(r"^\s*buffer_store_dwordx4\s+v\[?(\d+)(?::(\d+))?\]?")
_WRITE = re.compile(r"^\s*[a-z_0-9]+\s+v\[?(\d+)(?::(\d+))?\]?\s*,")
_WAIT = re.compile(r"^\s*s_waitcnt\b")
_VMCNT = re.compile(r"vmcnt\((\d+)\)")
_STOP = re.compile(r"^\s*s_(endpgm|branch|cbranch|setpc|swappc)\b")

_stats = {"scanned": 0, "patched": 0, "residual": 0, "names": {}}

_PREFIX = "[Patch:triton.buffer_store_war]"


def _log(msg: str) -> None:
    """Log without ever raising: this runs inside the compiler, which can be
    reached before the Primus logger exists."""
    try:
        log_rank_0(f"{_PREFIX} {msg}")
    except Exception:
        print(f"{_PREFIX} {msg}", file=sys.stderr)


def has_hazard(asm: str) -> bool:
    """True if a buffer_store_dwordx4 can have its data VGPRs clobbered.

    Walks forward from each store to the first instruction that either waits on
    vmcnt (the store's registers are safe from there on) or redefines a
    register the store is reading (the miscompile).
    """
    if "buffer_store_dwordx4" not in asm:
        return False
    lines = asm.splitlines()
    for i, line in enumerate(lines):
        m = _STORE.match(line)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        for j in range(i + 1, len(lines)):
            nxt = lines[j]
            if not nxt.strip() or nxt.lstrip().startswith((";", ".", "//")):
                continue
            if _STOP.match(nxt):
                break
            if _WAIT.match(nxt):
                if _VMCNT.search(nxt):
                    break
                continue
            w = _WRITE.match(nxt)
            if w:
                wlo = int(w.group(1))
                whi = int(w.group(2)) if w.group(2) else wlo
                # Both operands are contiguous VGPR ranges, so they overlap iff
                # neither ends before the other begins.
                if wlo <= hi and lo <= whi:
                    return True
                break
    return False


def _on_rocm() -> bool:
    """True on any ROCm GPU.

    Deliberately not an architecture allowlist.  Wrong values were measured on
    gfx942; gfx950 emits the same pattern but was measured to tolerate it.  Since
    the compiler is omitting a required ``s_waitcnt`` on both, which chips happen
    to tolerate that is not a safe thing to encode -- and an allowlist would have
    to be widened for every new architecture, leaving it unprotected until
    somebody remembers.  Scanning decides instead, at a substring test per
    compiled kernel.
    """
    try:
        import torch

        return torch.cuda.is_available() and torch.version.hip is not None
    except Exception:
        return False


def _break_inductor_caches() -> None:
    """Make torch.compile caches from before this guard unreachable.

    On an FX graph cache hit inductor never calls ``triton.compile``, so a cache
    filled before the guard existed keeps serving hazardous binaries -- and does
    so invisibly, since ``_stats`` only counts kernels that reach the compiler.
    """
    tag = f"primus-bufops-war-v{_GUARD_VERSION}"
    existing = os.environ.get("TORCH_COMPILE_CACHE_KEY_TAG", "")
    if tag in existing.split(","):
        return
    combined = f"{existing},{tag}" if existing else tag
    os.environ["TORCH_COMPILE_CACHE_KEY_TAG"] = combined
    try:
        from torch.compiler import config as compiler_config

        # The env var is only read when torch.compiler.config is first imported,
        # which may already have happened.
        compiler_config.cache_key_tag = combined
    except Exception as exc:
        _log(
            f"could not set cache_key_tag ({exc}); wipe the torch.compile cache "
            f"directory before trusting this run"
        )


@register_patch(
    "triton.buffer_store_war",
    backend=None,  # both Megatron and TorchTitan compile Triton kernels
    phase="build_args",  # earliest phase, well before the first kernel compile
    condition=lambda ctx: _on_rocm(),
)
def patch_triton_buffer_store_war(ctx: PatchContext) -> None:
    """Disable buffer ops for the individual kernels that hit the store WAR bug."""
    import triton.knobs as knobs
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.compiler import compiler as _cc

    if getattr(_cc.compile, "_primus_bufops_war", False):
        return

    _break_inductor_caches()
    orig = _cc.compile

    @functools.wraps(orig)
    def guarded(src, target=None, options=None, _env_vars=None):
        kernel = orig(src, target=target, options=options, _env_vars=_env_vars)
        try:
            if kernel.metadata.target.backend != "hip":
                return kernel
            asm = kernel.asm["amdgcn"]
        except Exception:
            return kernel

        _stats["scanned"] += 1
        if not has_hazard(asm):
            return kernel

        name = getattr(kernel, "name", "?")
        # Record the override in the cache key so flipping the policy off cannot
        # resurrect an artifact compiled under the other setting.
        env = dict(get_cache_invalidating_env_vars() if _env_vars is None else _env_vars)
        env["AMDGCN_USE_BUFFER_OPS"] = "0"
        with knobs.amd.scope():
            knobs.amd.use_buffer_ops = False
            try:
                fixed = orig(src, target=target, options=options, _env_vars=env)
            except Exception as exc:  # a broken workaround must not break the build
                _stats["residual"] += 1
                _log(f"{name}: recompile failed ({exc}); KEEPING HAZARDOUS KERNEL")
                return kernel

        _stats["patched"] += 1
        _stats["names"][name] = _stats["names"].get(name, 0) + 1
        if has_hazard(fixed.asm["amdgcn"]):
            _stats["residual"] += 1
            _log(f"{name}: hazard survives with buffer ops off; the workaround does not cover it")
        return fixed

    guarded._primus_bufops_war = True
    _cc.compile = guarded
    # JITFunction.create_binder resolves `from ..compiler import compile` lazily,
    # so the package attribute is the binding that actually reaches the JIT.
    for modname in ("triton.compiler", "triton", "triton.runtime.jit"):
        mod = sys.modules.get(modname)
        if mod is not None and getattr(mod, "compile", None) is orig:
            mod.compile = guarded

    _log("installed")


def get_stats() -> dict:
    """Kernels seen, recompiled, and still hazardous afterwards.

    ``residual`` must be zero.  It only counts kernels that reached the
    compiler, not ones served from a torch.compile cache.
    """
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _stats.items()}
