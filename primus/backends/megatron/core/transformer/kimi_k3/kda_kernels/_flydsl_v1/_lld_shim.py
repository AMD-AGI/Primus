###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Repair the ``ld.lld`` invocation FlyDSL's JIT needs, when it is broken.

Without this, **no** FlyDSL kernel compiles on ``rocm/primus:v26.4``. Every
``@flyc.jit`` launcher dies inside the ``gpu-module-to-binary`` pass with::

    could not find path component of main program: 'ld.lld'
    error: "-":2:3: lld invocation failed

The cause is a defect in the ``_rocm_sdk_devel`` wheel that ships this image's
toolchain, not in FlyDSL and not in the kernel:

* ``$ROCM_PATH/llvm/bin/ld.lld`` is a 26 KB *relocation trampoline*, not the
  linker — the real ``lld`` beside it is 8.2 MB. The trampoline finds the real
  binary **relative to its own** ``argv[0]``.
* MLIR's ROCDL target resolves the tool's path but spawns it with
  ``argv[0] = "ld.lld"``, a bare name with no path component, so the trampoline
  gives up. Run by absolute path the same binary works (``AMD LLD 23.0.0``,
  rc=0); run as a bare ``ld.lld`` it exits 1.
* A symlink does not help: it preserves ``argv[0]`` and, worse, points the
  trampoline at a directory holding no ``lld`` at all. (``$ROCM_PATH/bin/amdlld``
  is broken for exactly that reason and says so:
  ``binary '.../bin/lld' does not exist``.)

So the repair is a tiny wrapper *script* named ``ld.lld`` — its own ``argv[0]``
is irrelevant because it hands off by absolute path to ``lld -flavor gnu``,
which is what ``ld.lld`` is.

Where the wrapper has to go
---------------------------
**Prepending it to ``PATH`` is not enough** — measured, not assumed. MLIR looks
for the tool inside the ROCm *toolkit* directory, so a wrapper that is only on
``PATH`` is never consulted and the compile fails with the original message.
The wrapper therefore goes into a **shadow toolkit**: a directory that symlinks
every entry of the real ``$ROCM_PATH`` (so ``amdgcn/bitcode``, ``lib`` and the
rest still resolve) with ``ld.lld`` replaced, and ``ROCM_PATH`` repointed at it.

:func:`ensure_usable_lld` probes for the defect the same way MLIR triggers it
and only then builds the shadow, so a healthy toolchain is left untouched.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from typing import Optional

__all__ = ["ensure_usable_lld"]

_SHADOW_DIR_NAME = "primus_kda_flydsl_rocm_shadow"
_probe_cache = {"probed": False, "shadow": None}  # once-per-process memoization


def _resolves_and_runs_by_bare_name() -> bool:
    """Does ``ld.lld`` work when resolved from ``PATH`` and spawned by name?

    That is how MLIR invokes the linker, and precisely what the trampoline
    cannot survive, so it is the condition worth testing rather than mere
    existence or executability.

    The probe goes through ``/bin/sh`` on purpose. Passing ``executable=`` to
    :func:`subprocess.run` looks equivalent but is not: it was measured to
    *succeed* against the very trampoline that fails from a shell, which made
    an earlier version of this check pass vacuously.
    """
    try:
        proc = subprocess.run(
            ["/bin/sh", "-c", "ld.lld --version"],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _toolkit_root() -> Optional[str]:
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME"), "/opt/rocm"):
        if root and os.path.isdir(root) and os.path.isdir(os.path.join(root, "llvm", "bin")):
            return root
    return None


def _real_lld(root: str) -> Optional[str]:
    """The generic ``lld`` driver the wrapper delegates to."""
    candidate = os.path.join(root, "llvm", "bin", "lld")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which("lld")


def _mirror(src: str, dst: str, skip: frozenset) -> None:
    """Symlink every entry of ``src`` into ``dst``, minus ``skip``."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name in skip:
            continue
        link = os.path.join(dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(src, name), link)


def _write_wrapper(path: str, real_lld: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        fh.write(f'#!/bin/sh\nexec "{real_lld}" -flavor gnu "$@"\n')
    os.chmod(tmp, os.stat(tmp).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, path)  # atomic: concurrent ranks may race here


def _build_shadow_toolkit(root: str, real_lld: str) -> str:
    shadow = os.path.join(tempfile.gettempdir(), _SHADOW_DIR_NAME)
    # `bin` and `llvm/bin` are mirrored entry by entry rather than symlinked
    # wholesale, because the wrapper has to be *inside* them.
    _mirror(root, shadow, frozenset({"llvm", "bin"}))
    _mirror(os.path.join(root, "llvm"), os.path.join(shadow, "llvm"), frozenset({"bin"}))
    for rel in ("bin", os.path.join("llvm", "bin")):
        src = os.path.join(root, rel)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(shadow, rel)
        _mirror(src, dst, frozenset({"ld.lld"}))
        _write_wrapper(os.path.join(dst, "ld.lld"), real_lld)
    return shadow


def ensure_usable_lld() -> Optional[str]:
    """Make ``ld.lld`` usable by MLIR, repairing the toolkit only if needed.

    Idempotent and cached: the probe runs at most once per process. Returns the
    shadow toolkit directory it installed, or ``None`` when the toolchain was
    already healthy.

    Raises:
        RuntimeError: when ``ld.lld`` is broken and cannot be repaired, since
            every subsequent kernel compile would otherwise fail with a message
            that does not name the real cause.
    """
    if _probe_cache["probed"]:
        return _probe_cache["shadow"]
    _probe_cache["probed"] = True

    if _resolves_and_runs_by_bare_name():
        return None

    root = _toolkit_root()
    real_lld = _real_lld(root) if root else None
    if root is None or real_lld is None:
        raise RuntimeError(
            "the FlyDSL KDA backend needs a working `ld.lld` to JIT-compile its kernels; "
            f"the one on PATH ({shutil.which('ld.lld')!r}) fails when spawned with a bare "
            "argv[0], and no ROCm toolkit with a generic `lld` driver was found to build a "
            "replacement from ($ROCM_PATH / $ROCM_HOME / /opt/rocm). Select a different "
            "kda_backend (eager | eager_recurrent | fla)."
        )

    shadow = _build_shadow_toolkit(root, real_lld)
    os.environ["ROCM_PATH"] = shadow
    os.environ["PATH"] = os.path.join(shadow, "llvm", "bin") + os.pathsep + os.environ.get("PATH", "")
    if not _resolves_and_runs_by_bare_name():
        raise RuntimeError(
            f"built a shadow ROCm toolkit at {shadow} with an `ld.lld` wrapper delegating to "
            f"{real_lld}, but `ld.lld` still does not run by name. Select a different "
            "kda_backend (eager | eager_recurrent | fla)."
        )
    _probe_cache["shadow"] = shadow
    return shadow
