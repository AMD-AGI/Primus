###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Feature registry and capability gates of the DeepSeek-V4 fast path on gfx1250 (MI455X).

The fast path is a set of fused kernels and wiring changes that are on by default when the device
is gfx1250 and each feature's own preconditions hold (parallel sizes, shapes, kernel availability;
those guards live next to the code they protect). This module holds what is shared:

* ``gluon_ok()`` -- the one gate for every Gluon kernel: Triton >= 3.9 (or ``PRIMUS_GLUON=1``), a
  gfx1250 device, and ``PRIMUS_GLUON`` not 0. When it is false the Triton / hipBLASLt fallbacks run.
* ``apply_defaults()`` -- fills ``os.environ`` with the defaults of the fast path before the
  modules that read them are imported. Explicit user settings always win (``setdefault``).
* ``describe()`` -- one rank-0 log line listing the active gates.

Switches that remain user-facing:

``PRIMUS_DSV4_FAST=0``
    Master off switch of the Primus-side fast path (upstream Primus paths run instead). Primus
    side only: the Turbo kernels it falls back to are the default Turbo ones.
``PRIMUS_DSV4_BITWISE=1``
    Turns the fusions that change floating-point summation order back off, for numerics debugging.
``PRIMUS_GLUON=0``
    Forces the non-Gluon fallbacks (``1`` skips the Triton version check).
``PRIMUS_HC_FUSED`` / ``PRIMUS_OVERLAP`` / ``PRIMUS_FP8BLK_ADAM`` / ``PRIMUS_DSA_GFX1250_TURBO``
    Per-component kill switches (``0`` disables the component).
``PRIMUS_FP8BLK_ADAM_SKIP_BF16=0``
    Keeps the BF16 weights current for external readers or a mid-run checkpoint load.

Defaults are in two tiers. The model tier covers names that only DeepSeek-V4 code reads. The recipe
tier additionally requires ``PRIMUS_TURBO_FP8_BLOCK=1`` (the block-FP8 paper recipe), because those
names enable patches that touch every model (first-write wgrad, step-boundary logging, optimizer).
"""

from __future__ import annotations

import functools
import logging
import os
import re
import sys
from typing import Dict, Optional, Tuple

GLUON_MIN_TRITON: Tuple[int, int] = (3, 9)

# --- model tier: read only by DeepSeek-V4 code -------------------------------------------------
_MODEL_DEFAULTS: Dict[str, str] = {
    # attention glue
    "PRIMUS_DSA_GFX1250_TURBO": "1",
    "PRIMUS_V4_QKV_A_CONCAT": "1",
    "PRIMUS_V4_QKV_POST_FUSED": "1",
    "PRIMUS_V4_ROPE_BWD_INPLACE": "1",
    "PRIMUS_V4_OROPE_INPLACE": "1",
    "PRIMUS_V4_OA_MAIN_GRAD": "1",
    "PRIMUS_V4_OA_DGRAD_TN": "1",
    "PRIMUS_V4_INDEXER_TOPK_TRITON": "1",
    "PRIMUS_GLUE_RMSNORM": "1",
    "PRIMUS_GLUE_ROPE_NOCAST": "1",
    "PRIMUS_GLUE_COMPRESS_POOL": "1",
    "PRIMUS_MB_ROPE_TBL": "1",
    "PRIMUS_LAUNCH_ROPE_TBL": "1",
    "PRIMUS_LAUNCH_SINK_CAST": "1",
    "PRIMUS_LAUNCH_DSINK": "2",
    "PRIMUS_FUSE2_WPOOL_DAPE": "1",
    "PRIMUS_FUSE2_COMP_MG": "1",
    "PRIMUS_FUSE2_ATT_GADD": "1",
    "PRIMUS_GLUE_SMALLS": "1",
    "PRIMUS_MB_SMALLS": "1",
    # MoE routing
    "PRIMUS_V4_ROUTER_FUSED": "1",
    "PRIMUS_V4_ROUTER_DENSE_OUT": "1",
    "PRIMUS_FUSE_ROUTE": "1",
    "PRIMUS_FUSE_MOE_ADD": "1",
    "PRIMUS_PERSIST_ROUTER_BWD": "1",
    "PRIMUS_MB_ROUTER_FWD": "128,4",
    # BF16 GEMM routing (Gluon-gated inside the fusions)
    "PRIMUS_BF16G2_OA_DGRAD": "1",
    "PRIMUS_BF16G2_IDXQ": "1",
}

# --- recipe tier: patches that apply to any model, enabled together with the block-FP8 recipe ---
_RECIPE_DEFAULTS: Dict[str, str] = {
    # optimizer / grad norm
    "PRIMUS_FP8BLK_ADAM_QUANT": "1",
    "PRIMUS_FP8BLK_ADAM_MEGAMOE": "1",
    "PRIMUS_FP8BLK_ADAM_SKIP_BF16": "1",
    "PRIMUS_OPTB_DENSE_SKIP_BF16": "1",
    "PRIMUS_OPTB_EMB_SKIP_BF16": "1",
    "PRIMUS_GRAD_NORM_FROM_NAN_CHECK": "1",
    "PRIMUS_MB_GNORM": "1",
    # wgrad into main_grad
    "PRIMUS_WGRAD_BETA0": "1",
    "PRIMUS_TURBO_G90_WGRAD_FUSE": "1",
}

# Component kill switches: ``X=0`` turns every name in the group off (unless that name is set explicitly).
_KILL_GROUPS: Dict[str, Tuple[str, ...]] = {
    "PRIMUS_FP8BLK_ADAM": (
        "PRIMUS_FP8BLK_ADAM_QUANT",
        "PRIMUS_FP8BLK_ADAM_MEGAMOE",
        "PRIMUS_FP8BLK_ADAM_SKIP_BF16",
        "PRIMUS_OPTB_DENSE_SKIP_BF16",
        "PRIMUS_OPTB_EMB_SKIP_BF16",
    ),
    "PRIMUS_FP8BLK_ADAM_SKIP_BF16": ("PRIMUS_OPTB_DENSE_SKIP_BF16", "PRIMUS_OPTB_EMB_SKIP_BF16"),
}

# PRIMUS_DSV4_BITWISE=1: fusions that change the order of floating-point sums go back to the reference order.
_BITWISE_OFF: Dict[str, str] = {
    "PRIMUS_MB_GNORM": "0",
    "PRIMUS_TURBO_MOE_WGRAD_SUMSQ": "0",
    "PRIMUS_V4_QKV_A_CONCAT": "0",
    "PRIMUS_FUSE2_WPOOL_DAPE": "0",
    "PRIMUS_FUSE_ROUTE": "0",
    "PRIMUS_GLUE_RMSNORM": "0",
}


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


@functools.lru_cache(maxsize=None)
def gpu_arch() -> str:
    """Device architecture string (``gfx1250``), or ``""`` when no GPU is visible."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0].lower()
    except Exception:  # noqa: BLE001
        return ""


def is_gfx1250() -> bool:
    return gpu_arch() == "gfx1250"


@functools.lru_cache(maxsize=None)
def triton_version() -> Optional[Tuple[int, int]]:
    """``(major, minor)`` of the installed Triton, or None when it is not importable."""
    try:
        import triton

        m = re.match(r"(\d+)\.(\d+)", triton.__version__)
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=None)
def gluon_ok() -> bool:
    """Whether Gluon kernels may be used: Triton >= 3.9 on a gfx1250 device and ``PRIMUS_GLUON`` not 0.

    Triton 3.8 ships the ``cdna5`` Gluon modules but cannot compile the kernels, so an import test is not a
    usable gate. ``PRIMUS_GLUON=1`` skips the version check (for a Triton build with a different version string).
    """
    setting = os.environ.get("PRIMUS_GLUON", "").strip().lower()
    if setting in ("0", "false", "no"):
        return False
    if not is_gfx1250():
        return False
    if setting in ("1", "true", "yes"):
        return True
    ver = triton_version()
    return ver is not None and ver >= GLUON_MIN_TRITON


def fast_enabled() -> bool:
    """``PRIMUS_DSV4_FAST`` master switch (default on)."""
    return os.environ.get("PRIMUS_DSV4_FAST", "1").strip().lower() not in ("0", "false", "no")


def recipe_enabled() -> bool:
    return _flag("PRIMUS_TURBO_FP8_BLOCK")


def bitwise() -> bool:
    return _flag("PRIMUS_DSV4_BITWISE")


_applied: Dict[str, str] = {}


def _set(name: str, value: str) -> None:
    if name not in os.environ:
        os.environ[name] = value
        _applied[name] = value


def guard_broken_fla() -> bool:
    """Make a broken ``fla`` (flash-linear-attention) look like a missing one. Returns True when it was masked.

    The image's ``fla`` raises a non-ImportError at import time under Triton >= 3.9 (an autotuner ``key`` that is not
    a kernel argument). Megatron imports it from modules that DeepSeek-V4 never uses, behind ``except ImportError``,
    so the error escapes and aborts start-up. Probing the import once and masking ``fla`` with ``None`` in
    ``sys.modules`` turns every later ``import fla...`` into an ImportError. Only done for Triton >= 3.9, where the
    breakage occurs, and only if ``fla`` is installed.
    """
    ver = triton_version()
    if ver is None or ver < GLUON_MIN_TRITON or sys.modules.get("fla", False) is None:
        return False
    import importlib

    try:
        importlib.import_module("fla")
        return False
    except ImportError:
        return False
    except Exception as exc:  # noqa: BLE001
        for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
            del sys.modules[name]
        sys.modules["fla"] = None
        logging.getLogger(__name__).warning(
            "fla failed to import under Triton %d.%d (%s: %s); treating it as not installed",
            ver[0],
            ver[1],
            type(exc).__name__,
            str(exc).split("\n")[0][:120],
        )
        return True


def apply_defaults() -> Dict[str, str]:
    """Fill ``os.environ`` with the fast-path defaults. Idempotent; returns what was set by this process.

    Nothing is set when ``PRIMUS_DSV4_FAST=0`` or the device is not gfx1250.
    """
    guard_broken_fla()
    if _applied or not fast_enabled() or not is_gfx1250():
        return dict(_applied)
    for group, names in _KILL_GROUPS.items():
        if os.environ.get(group, "1").strip().lower() in ("0", "false", "no"):
            for n in names:
                _set(n, "0")
    if bitwise():
        for n, v in _BITWISE_OFF.items():
            _set(n, v)
    for n, v in _MODEL_DEFAULTS.items():
        _set(n, v)
    if recipe_enabled():
        for n, v in _RECIPE_DEFAULTS.items():
            _set(n, v)
    return dict(_applied)


def describe() -> str:
    """One-line summary of the gates, for the rank-0 log."""
    ver = triton_version()
    return (
        f"[dsv4_features] arch={gpu_arch() or 'none'} fast={fast_enabled()} recipe={recipe_enabled()} "
        f"triton={'.'.join(map(str, ver)) if ver else 'none'} gluon={gluon_ok()} bitwise={bitwise()} "
        f"defaults_set={len(_applied)}"
    )
