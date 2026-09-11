###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FLA KDA Autotune-Space Narrowing Patch
======================================

Drops the ``num_stages > 2`` candidates from the Triton autotune space of
flash-linear-attention's KDA intra-chunk kernels
(``fla/ops/kda/chunk_intra.py``), in-process, before the first kernel launch
triggers autotuning.

On Triton 3.6 (ROCm 7.2) the ``num_stages=4`` candidates fail to compile
outright: AMD's loop-scheduling pass aborts the MLIR pass pipeline
(``PassManager::run failed``). Triton 3.7+ compiles the whole sweep and the
autotuner settles on ``num_stages=2`` anyway, so the defect belongs to one
toolchain generation rather than to the kernels. The ``[2, 3, 4]`` space is
still declared in FLA 0.5.2 (the pinned version) and in upstream ``main``, so
there is no fixed release to upgrade to instead. Delete this patch once the
supported toolchain floor reaches Triton >= 3.7.

The gate is the platform (ROCm) plus the run actually using FLA KDA, never the
GPU architecture: the pass at fault is chip-independent, so an architecture
allowlist would only record where the failure was first seen and would silently
stop protecting architectures added later. Applying it where it was not needed
costs ~1% of a single op at worst, against a hard compile failure whenever it is
missed. The KDA half of the gate exists to avoid importing ``fla`` into a run
that has no other use for it, not because narrowing an unused kernel would hurt.

Two paths can feed a config to a kernel, so both are covered: ``tuner.configs``
is filtered in place, and ``tuner.cache`` is clamped, since
``CachedAutotuner.maybe_load_cached_config`` writes a best-config JSON straight
into the cache without consulting ``self.configs``. Unsafe cache entries are
clamped rather than dropped because under ``FLA_CACHE_MODE=always`` the JSON is
reloaded before every launch: dropping the entry makes Triton re-benchmark and
the next reload reinstates the unsafe config, i.e. a full autotune per kernel
call. Clamping leaves a cache hit that is already safe, and touches only
``num_stages`` so the other fields the JSON recorded survive.

Note the ``.fn`` walk in :func:`_find_autotuner`: the decorated module attribute
is a Triton ``Heuristics`` wrapper, and the ``CachedAutotuner`` that owns
``.configs`` sits a hop below it.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.fla.kda_safe_autotune"

# AMD's loop-scheduling pass cannot build the num_stages=4 variants of these
# kernels on Triton 3.6; 2 keeps a margin below the depth that trips it.
_MAX_SAFE_NUM_STAGES = 2

_FLA_KDA_INTRA_MODULE = "fla.ops.kda.chunk_intra"
_FLA_KDA_INTRA_KERNELS = (
    "chunk_kda_bwd_kernel_intra",
    # Unreachable today: only fla's safe_gate=True path calls it, and Primus
    # never enables safe_gate. Narrowed anyway so the protection is already in
    # place, but re-measure before enabling safe_gate -- at those shapes this is
    # the one kernel where num_stages=4 has been seen to win.
    "chunk_kda_fwd_kernel_intra_sub_chunk",
)

_TRUTHY = {"1", "true", "yes", "on"}

# Config keys that select the FLA KDA backend, in resolution order. Mirrors
# KimiK3TransformerConfig.__post_init__, where a non-null
# use_kimi_k3_attention_backend supersedes the legacy kda_backend field.
_FLA_TRITON_KDA_KEY = "use_fla_triton_kda"
_KDA_BACKEND_KEY = "kda_backend"
_KDA_BACKEND_OVERRIDE_KEY = "use_kimi_k3_attention_backend"
_FLA_BACKEND = "fla"


# ─── Gating ──────────────────────────────────────────────────────────────────


def _is_rocm() -> bool:
    """Return True when running on an AMD ROCm platform."""
    return getattr(torch.version, "hip", None) is not None


def _uses_fla_kda(args) -> str | None:
    """Return the args field that selects FLA KDA for this run, else None.

    Both spellings resolve here because ``train_runtime`` merges the model YAML
    preset and the experiment ``overrides:`` block onto the same ``args``
    namespace before the ``before_train`` phase runs.
    """
    value = getattr(args, _FLA_TRITON_KDA_KEY, None)
    if value is True or (isinstance(value, str) and value.strip().lower() in _TRUTHY):
        return _FLA_TRITON_KDA_KEY

    override = getattr(args, _KDA_BACKEND_OVERRIDE_KEY, None)
    if override is not None:
        backend, key = override, _KDA_BACKEND_OVERRIDE_KEY
    else:
        backend, key = getattr(args, _KDA_BACKEND_KEY, None), _KDA_BACKEND_KEY
    if backend is not None and str(backend).strip().lower() == _FLA_BACKEND:
        return key

    return None


def _should_narrow_kda_autotune(ctx: PatchContext) -> bool:
    """Apply on ROCm, for runs that reach FLA's KDA kernels.

    The KDA half of this is about ``import fla``, not about the narrowing:
    trimming the config list of a kernel that never launches would be harmless,
    but importing FLA into a run that has no other use for it is not.
    """
    if not _is_rocm():
        return False
    return _uses_fla_kda(get_args(ctx)) is not None


# ─── Autotuner surgery ───────────────────────────────────────────────────────


def _find_autotuner(kernel: Any, max_depth: int = 8) -> Any | None:
    """Return the object in ``kernel``'s decorator chain that owns ``.configs``.

    Triton stacks decorators as wrappers linked by ``.fn``, so the autotuner may
    sit any number of hops below the module attribute.
    """
    current = kernel
    for _ in range(max_depth):
        if current is None:
            return None
        if isinstance(getattr(current, "configs", None), list):
            return current
        current = getattr(current, "fn", None)
    return None


def _install_cache_guard(tuner: Any) -> bool:
    """Clamp unsafe configs that FLA's JSON config cache writes past the filter.

    Returns True if the guard was installed. No-op (and cheap) when the tuner
    predates the cache mechanism, and never called at all while
    ``FLA_CACHE_MODE`` is ``DISABLED``.
    """
    original = getattr(tuner, "maybe_load_cached_config", None)
    if original is None:
        return False
    if is_patched(tuner, _PATCH_KEY):
        return False

    def guarded(autotune_key):
        original(autotune_key)
        cache = getattr(tuner, "cache", None)
        if not cache:
            return
        # Sweep the whole cache, not just the key this call installed: the guard
        # goes on at before_train, so entries can already be there, and clamping
        # them is cheap.
        clamped = 0
        for config in cache.values():
            if getattr(config, "num_stages", 0) > _MAX_SAFE_NUM_STAGES:
                # In place, so every other field survives. maybe_load_cached_config
                # builds a fresh triton.Config per call and hands it to nobody
                # else, so there is no shared object to corrupt here.
                config.num_stages = _MAX_SAFE_NUM_STAGES
                clamped += 1
        if clamped:
            log_rank_0(
                f"[Patch:{_PATCH_KEY}] clamped {clamped} cached config(s) to "
                f"num_stages={_MAX_SAFE_NUM_STAGES} for {getattr(tuner, 'kernel_name', '?')}; "
                "the rest of each config is kept as measured."
            )

    tuner.maybe_load_cached_config = guarded
    mark_patched(tuner, _PATCH_KEY)
    return True


def _narrow_kernel_autotune(module: Any, kernel_name: str) -> bool:
    """Filter one kernel's autotune space.

    Returns True once the space is known safe -- including when it already was,
    which is the expected outcome on an FLA release that fixes this upstream.
    """
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        log_rank_0(f"[Patch:{_PATCH_KEY}] {kernel_name} not found in this FLA version; skipping.")
        return False

    tuner = _find_autotuner(kernel)
    if tuner is None:
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] no autotune config list reachable from {kernel_name}; "
            "FLA's decorator layout changed -- skipping."
        )
        return False

    before = list(tuner.configs)
    kept = [config for config in before if getattr(config, "num_stages", 0) <= _MAX_SAFE_NUM_STAGES]
    if not kept:
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] every {kernel_name} config needs num_stages > "
            f"{_MAX_SAFE_NUM_STAGES}; leaving the autotune space untouched rather than "
            "emptying it."
        )
        return False

    tuner.configs = kept
    _install_cache_guard(tuner)

    removed = len(before) - len(kept)
    warps = sorted({getattr(config, "num_warps", None) for config in kept})
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] {kernel_name}: kept {len(kept)}/{len(before)} autotune "
        f"configs (dropped {removed} with num_stages > {_MAX_SAFE_NUM_STAGES}); "
        f"num_warps sweep still {warps}."
    )
    return True


def _install_kda_safe_autotune_patch() -> None:
    try:
        module = importlib.import_module(_FLA_KDA_INTRA_MODULE)
    except Exception as exc:  # noqa: BLE001 - FLA absent or KDA ops moved
        log_rank_0(f"[Patch:{_PATCH_KEY}] cannot import {_FLA_KDA_INTRA_MODULE} ({exc!r}); skipping.")
        return

    if is_patched(module, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] KDA intra autotune space already narrowed; skipping.")
        return

    secured = [name for name in _FLA_KDA_INTRA_KERNELS if _narrow_kernel_autotune(module, name)]

    mark_patched(module, _PATCH_KEY)
    if not secured:
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] could not secure any KDA intra kernel; on Triton 3.6 "
            "this run may still fail to compile during autotuning. Check the lines "
            "above for the reason."
        )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Drop num_stages >= 3 from the autotune space of FLA's KDA intra-chunk "
        "kernels, which AMD Triton 3.6 fails to compile."
    ),
    # Well before the first KDA kernel launch triggers autotuning.
    priority=50,
    condition=_should_narrow_kda_autotune,
    tags=["rocm", "kda", "fla", "triton"],
)
def patch_fla_kda_safe_autotune(ctx: PatchContext) -> None:
    _install_kda_safe_autotune_patch()
