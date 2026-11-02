###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Enlarge Transformer Engine's GEMM workspace (ROCm hipBLASLt split-K needs more than the default).

The one implementation of this ROCm workaround. Users:

* Muse Spark (``before_train``), through ``apply_rocm_te_gemm_workspace``, which raises
  on ROCm when nothing is patched. The size is the ``te_gemm_workspace_mb`` arg (512
  when unset) or an explicit ``mb``; both go through ``parse_workspace_mb``, so a value
  that is not a positive integer stops the run.
* GLM-5.3 (``glm5_next_patches``), through ``apply_rocm_te_gemm_workspace`` too, with
  ``required=False`` (best effort: logs a warning when nothing is patched and keeps TE's
  workspace) and the size from ``PRIMUS_TE_GEMM_WORKSPACE_MB`` (parsed by
  ``workspace_mb_from_env``: unset is 512, blank raises, an integer <= 0 is no change;
  ROCm only).

``enlarge_te_gemm_workspace`` is idempotent: a second call with the same or a smaller
size changes nothing.

TE builds keep the workspace size function, ``get_cublas_workspace_size_bytes``, in
different modules (``module.base`` in the TE of the Primus image; older builds use
``cpp_extensions.gemm``), and other TE modules may hold their own reference to it
(``from .base import get_cublas_workspace_size_bytes``). So the helper:

* imports the known modules, then visits every loaded ``transformer_engine.*`` module
  that has a ``get_cublas_workspace_size_bytes`` attribute and rebinds it;
* drops the cached workspaces in those modules: the ``lru_cache`` of
  ``get_cublas_workspace`` / ``get_workspace`` / ``get_multi_stream_cublas_workspace``,
  and the module globals ``_cublas_workspace`` / ``_multi_stream_cublas_workspace``,
  so the next GEMM allocates the larger buffer.

A reference captured in a closure or a default argument cannot be reached; no TE
version known to us does that. Only ``transformer_engine.*`` modules are scanned:
Primus's own ``primus.backends.transformer_engine.pytorch.module.base`` defines a
``get_cublas_workspace_size_bytes`` that returns 0, but it belongs to the Primus
``torch.mm`` GEMM path (no hipBLASLt workspace) and TE never calls it, so there is
nothing to enlarge there. ``enlarge_te_gemm_workspace`` is best effort and reports
errors and notes in its result; ``apply_rocm_te_gemm_workspace`` (``required=True``,
Muse Spark) raises on ROCm when nothing could be patched (fail fast instead of
"HIPBLASLT Error: 6" later).

Memory: the size is a lower bound per workspace buffer. TE allocates one buffer for
the default stream and, for grouped / multi-stream GEMMs, one per extra stream
(``get_multi_stream_cublas_workspace``), so the per-GPU cost is about
``size x (1 + number of GEMM streams)``, e.g. ~2.5 GiB at 512 MiB and 4 streams.
Lower the ``te_gemm_workspace_mb`` config / args value to trade memory for split-K headroom.
"""

import contextlib
import importlib
import os
import sys

from primus.core.utils.module_utils import log_rank_0, warning_rank_0
from primus.core.utils.number_utils import parse_positive_int

__all__ = [
    "enlarge_te_gemm_workspace",
    "apply_rocm_te_gemm_workspace",
    "workspace_mb_of",
    "parse_workspace_mb",
    "DEFAULT_MB",
    "ARG_MB",
]

ARG_MB = "te_gemm_workspace_mb"  # config / args key (muse_spark_base.yaml: 512)
DEFAULT_MB = 512
SIZE_FN = "get_cublas_workspace_size_bytes"
_KNOWN = (
    "transformer_engine.pytorch.module.base",
    "transformer_engine.pytorch.cpp_extensions.gemm",
)
_CACHED_FUNCS = ("get_cublas_workspace", "get_workspace", "get_multi_stream_cublas_workspace")
_CACHE_GLOBALS = ("_cublas_workspace", "_multi_stream_cublas_workspace")


def _size_fn(original, size_bytes):
    """``max(original(), size_bytes)``, evaluated per call, so TE's own size is kept when
    it is larger. When ``original()`` raises (TE's size is unknown, e.g. "No HIP GPUs are
    available" before the device is set), the call returns ``size_bytes``.

    Evaluating ``original`` on each call, not once at patch time, avoids guessing. The
    first such error is logged (once), so a broken TE size function is not hidden.
    """
    logged = []

    def get_cublas_workspace_size_bytes() -> int:
        try:
            return max(int(original()), size_bytes)
        except Exception as e:  # noqa: BLE001 - TE's size still unknown: use ours
            if not logged:
                logged.append(e)
                warning_rank_0(
                    f"[te_gemm_workspace] TE's {SIZE_FN}() raised {type(e).__name__}: {e}; using "
                    f"the Primus size {size_bytes} bytes (logged once)"
                )
            return size_bytes

    get_cublas_workspace_size_bytes._primus_workspace_bytes = size_bytes
    get_cublas_workspace_size_bytes._primus_original = original
    return get_cublas_workspace_size_bytes


def _drop_caches(mod):
    for fn in _CACHED_FUNCS:
        cache_clear = getattr(getattr(mod, fn, None), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    for g in _CACHE_GLOBALS:
        cur = getattr(mod, g, None)
        if isinstance(cur, (list, dict)):
            cur.clear()
        elif cur is not None:
            setattr(mod, g, None)


def enlarge_te_gemm_workspace(size_bytes: int):
    """Make every TE ``get_cublas_workspace_size_bytes`` report ``max(TE's size, size_bytes)``.

    Returns ``(enlarged, unchanged, errors, notes)``:

    * ``enlarged``: modules whose size function now returns at least ``size_bytes``
      (rebound to a ``max`` wrapper, cached workspaces dropped);
    * ``unchanged``: modules whose TE size already is >= ``size_bytes`` (or that were
      already patched to at least that size); nothing done;
    * ``errors``: ``"module: error"`` strings for real failures (a failed rebind, a
      broken import, or a missing known module when nothing was patched);
    * ``notes``: informational ``"module: ..."`` strings that do not need action: a
      size probe that raised (the module was still patched; the wrapper asks TE
      again on each call), or a known module that does not exist in this TE build
      (``ImportError``) while another module was patched.

    Best effort: this function does not raise.
    """
    errors, notes, missing = [], [], []
    for name in _KNOWN:
        try:
            importlib.import_module(name)
        except ImportError as e:  # this TE build does not have the module (or no TE at all)
            missing.append(f"{name}: {type(e).__name__}: {e}")
        except Exception as e:  # broken extension (OSError), no GPU, ...
            errors.append(f"{name}: {type(e).__name__}: {e}")
    enlarged, unchanged = [], []
    for name, mod in sorted(sys.modules.items()):
        if mod is None or not (name == "transformer_engine" or name.startswith("transformer_engine.")):
            continue
        try:
            get_size = mod.__dict__.get(SIZE_FN)
            if get_size is None:
                continue
            already = getattr(get_size, "_primus_workspace_bytes", None)
            if already is not None and already >= size_bytes:
                unchanged.append(name)
                continue
            original = getattr(get_size, "_primus_original", get_size)
            try:
                current = int(original())
            except Exception as e:  # e.g. "No HIP GPUs are available"
                notes.append(f"{name}: size probe failed ({type(e).__name__}: {e}); patched with max()")
                current = None
            if current is not None and current >= size_bytes:
                unchanged.append(name)
                continue
            setattr(mod, SIZE_FN, _size_fn(original, size_bytes))
            _drop_caches(mod)
            enlarged.append(name)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    (notes if enlarged or unchanged else errors).extend(missing)
    return enlarged, unchanged, errors, notes


def parse_workspace_mb(raw, name: str = ARG_MB) -> int:
    """``raw`` as a positive int (MiB); ``name`` is used in the error.

    DEFAULT_MB when unset (None or an empty string). Accepted: an int (also a numpy
    integer), an integral float (``512.0``) and a string of either (``"512"``, ``"512.0"``,
    ``"5e2"``). Anything else (a bool, a fraction, NaN / inf, <= 0, an unparsable string)
    raises ValueError: a typo must not silently run with another workspace size.
    """
    return parse_positive_int(
        raw, DEFAULT_MB, f"{name}={raw!r} is not a positive integer number of MiB (e.g. 512)"
    )


def workspace_mb_from_env(env_name: str, environ=None):
    """The workspace size (MiB) set by the environment variable ``env_name``, for a caller
    configured by env (GLM-5.3); None when the value is an integer <= 0 ("no enlargement",
    e.g. ``0`` or ``-1``).

    Unset is DEFAULT_MB. Every other value goes through :func:`parse_workspace_mb`, the
    parser of the ``te_gemm_workspace_mb`` arg, so the env and the YAML key accept the same
    sizes: a blank value, a non-integer (``-0.5``, ``0.5``), ``-inf`` / ``nan`` or text
    raise ValueError naming ``env_name``.
    """
    raw = (os.environ if environ is None else environ).get(env_name)
    if raw is None:
        return DEFAULT_MB
    if not raw.strip():  # set but blank is a config error: unset it for the default
        raise ValueError(f"{env_name}={raw!r} is blank; unset it for the {DEFAULT_MB} MiB default")
    with contextlib.suppress(ValueError):  # not an integer: parse_workspace_mb names the variable
        if int(raw.strip()) <= 0:  # only an integer opts out ("0", "-1"); "-0.5" / "-inf" raise
            return None
    return parse_workspace_mb(raw, env_name)


def workspace_mb_of(args) -> int:
    """``args.te_gemm_workspace_mb`` through :func:`parse_workspace_mb` (512 when unset)."""
    # ARG_MB, read as a literal so the config-schema scan sees the key in use
    return parse_workspace_mb(getattr(args, "te_gemm_workspace_mb", None))


def apply_rocm_te_gemm_workspace(
    tag: str, args=None, mb=None, mb_name: str = "mb", required: bool = True
) -> list:
    """The ``before_train`` patch body: on ROCm, TE GEMM workspace >= ``mb`` MiB.

    ``mb`` defaults to ``workspace_mb_of(args)`` (``args.te_gemm_workspace_mb``, 512); an
    explicit ``mb`` is validated the same way (:func:`parse_workspace_mb`: 0, -5, 1.5 or
    ``"abc"`` raise ValueError naming ``mb_name``; ``""`` is DEFAULT_MB; ``mb=None`` uses
    ``args.te_gemm_workspace_mb``, itself DEFAULT_MB when unset). Logs errors as warnings and
    notes as info. ``required`` (Muse Spark): raise RuntimeError on ROCm when no TE size
    function could be found or patched, since the run would otherwise fail later with
    "HIPBLASLT Error: 6". ``required=False`` (GLM-5.3, best effort) logs that as a
    warning and keeps TE's own workspace. Returns
    the enlarged modules. Off ROCm it only validates the value and returns ``[]``.
    """
    import torch

    # parsed on every platform: a bad value fails on CUDA too, not only later on ROCm
    mb = workspace_mb_of(args) if mb is None else parse_workspace_mb(mb, mb_name)
    if torch.version.hip is None:
        return []
    size = mb * 1024 * 1024
    enlarged, unchanged, errors, notes = enlarge_te_gemm_workspace(size)
    if not enlarged and not unchanged:
        msg = (
            f"{tag} no Transformer Engine {SIZE_FN} found or patched (errors: {errors or 'none'}); "
            "the default TE GEMM workspace is too small for hipBLASLt split-K (HIPBLASLT Error: 6)"
        )
        if required:
            raise RuntimeError(msg)
        warning_rank_0(msg + "; going on with TE's own workspace")
        return []
    if enlarged:
        log_rank_0(
            f"{tag} TE GEMM workspace -> max(TE default, {mb} MiB) in {', '.join(enlarged)} "
            f"(per GPU: one buffer, plus one per multi-stream GEMM stream)"
        )
    if unchanged:
        log_rank_0(f"{tag} TE GEMM workspace already >= {mb} MiB in {', '.join(unchanged)}")
    for n in notes:
        log_rank_0(f"{tag} {n}")
    for e in errors:
        warning_rank_0(f"{tag} {e}")
    return enlarged
