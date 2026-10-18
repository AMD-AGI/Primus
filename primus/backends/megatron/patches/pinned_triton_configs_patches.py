###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in pinned Triton configs for the MLA YaRN RoPE and MoE permutation fusions
(``use_primus_pinned_triton_configs``).

``apply_rope_fusion`` (multi-latent attention) and ``moe_permute_fusion`` launch kernels wrapped in
``triton.autotune``: on the first launch per autotune key every candidate config is compiled and benchmarked
inside the training process. On gfx1250 (MI455X), benchmarking autotune candidates inside the training process
can hang; pinning removes it.

This patch replaces each autotuner's config list in place by one config picked from the stock candidates
(the groups below), so ``Autotuner.run`` never benchmarks. The kernel sources are untouched, and every
reference to the autotuner object (module globals, ``from ... import`` bindings) sees the pinned list. Only
the kernels of the enabled fusions are pinned. Every requested kernel is resolved before any autotuner
changes: if a required one cannot be pinned (Triton or Megatron layout changed), nothing is pinned and the run
aborts at ``before_train`` instead of letting the first launch autotune.
"""

from __future__ import annotations

import importlib
from typing import Any, NamedTuple

from primus.backends.megatron.core.fusions._notices import warn_if_not_gfx1250
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.triton.pinned_configs"
_ARG = "use_primus_pinned_triton_configs"


class _PinGroup(NamedTuple):
    module: str
    kernels: tuple[str, ...]
    meta: dict[str, int]
    num_warps: int
    num_stages: int
    optional: bool = False  # skipped with a log line when it cannot be pinned


# Picked on gfx1250 (MI455X) at DeepSeek-V3 shapes (128 heads, qk 128 + rope 64, v 128, hidden 7168, top-8
# routing) from the stock candidates. Any candidate is correct (the kernels mask by head / hidden size).
_ROPE_GROUP = _PinGroup(
    "megatron.core.fusions.fused_mla_yarn_rope_apply",
    ("rotary_fwd_q_kernel", "rotary_bwd_q_kernel", "rotary_fwd_kv_kernel", "rotary_bwd_kv_kernel"),
    {"BLOCK_H": 32},
    num_warps=4,
    num_stages=3,
)
# The permutation row-id-map kernels are not autotuned upstream and are left alone.
_PERMUTE_PIN = dict(
    kernels=(
        "_permute_kernel",
        "_unpermute_kernel",
        "_unpermute_bwd_with_merging_probs_kernel",
        "_sort_chunks_by_map_kernel",
    ),
    meta={"BLOCK_SIZE": 1024},
    num_warps=4,
    num_stages=3,
)
_PERMUTE_GROUPS = (
    # What moe_permute_fusion routes to (patch "megatron.moe.permute_fusion").
    _PinGroup("primus.backends.transformer_engine.pytorch.common.triton.permutation", **_PERMUTE_PIN),
    # Transformer Engine's own copy (same kernels), reached by any direct TE caller.
    _PinGroup("transformer_engine.common.triton.permutation", **_PERMUTE_PIN, optional=True),
)


def _wanted_groups(args) -> list[tuple[str, tuple[_PinGroup, ...]]]:
    """[(label, groups)] for every enabled fusion whose kernels get pinned."""
    groups = []
    if getattr(args, "multi_latent_attention", False) and getattr(args, "apply_rope_fusion", False):
        groups.append(("mla_rope", (_ROPE_GROUP,)))
    # The patches run before validate_args, which turns moe_permute_fusion on for turbo_sync_free_moe_stage
    # >= 1.
    if (
        getattr(args, "moe_permute_fusion", False)
        or (getattr(args, "turbo_sync_free_moe_stage", 0) or 0) >= 1
    ):
        groups.append(("moe_permute", _PERMUTE_GROUPS))
    return groups


def _find_autotuner(obj: Any, max_depth: int = 8) -> Any:
    """The object in a Triton decorator chain (linked by ``.fn``) that owns ``.configs``."""
    from triton.runtime.autotuner import Autotuner

    cur = obj
    for _ in range(max_depth):
        if cur is None:
            return None
        if isinstance(cur, Autotuner):
            return cur
        cur = getattr(cur, "fn", None)
    return None


def _resolve_group(group: _PinGroup) -> list[tuple[str, Any]]:
    """[(kernel name, autotuner)] of ``group``, checked for pinning; changes nothing.

    Kernels that are not autotuned in this version are left out (nothing tunes them in process). Raises if
    the module cannot be imported, a kernel is missing or an autotuner has no argument for a pinned
    meta-parameter.
    """
    mod = importlib.import_module(group.module)
    found = []
    for name in group.kernels:
        obj = getattr(mod, name, None)
        if obj is None:
            raise RuntimeError(f"{group.module}.{name} not found")
        tuner = _find_autotuner(obj)
        if tuner is None:
            continue
        missing = [k for k in group.meta if k not in tuner.arg_names]
        if missing:
            raise RuntimeError(f"{group.module}.{name}: no argument(s) {missing} to pin")
        found.append((name, tuner))
    return found


def _pin(group: _PinGroup, found: list[tuple[str, Any]]) -> dict[str, list[str]]:
    """Give every resolved autotuner of ``group`` its one config (in place) and drop its tuning state.
    Returns {kernel: [str(config)]} as installed."""
    import triton

    for _, tuner in found:
        tuner.configs = [
            triton.Config(dict(group.meta), num_warps=group.num_warps, num_stages=group.num_stages)
        ]
        tuner.cache.clear()
        if hasattr(tuner, "early_config_prune"):
            tuner.early_config_prune = None
        if hasattr(tuner, "perf_model"):
            tuner.perf_model = None
    return {name: [str(c) for c in tuner.configs] for name, tuner in found}


def _apply_pinned_configs(args) -> dict[str, dict[str, list[str]]]:
    """Pin the kernels of every enabled fusion; returns {module: {kernel: [config]}}.

    Every group is resolved first; if a required kernel cannot be pinned, RuntimeError is raised and no
    autotuner has been changed.
    """
    resolved = []
    for label, groups in _wanted_groups(args):
        for group in groups:
            try:
                resolved.append((label, group, _resolve_group(group)))
            except Exception as exc:  # noqa: BLE001 - optional groups are skipped, the others re-raised
                if not group.optional:
                    raise RuntimeError(
                        f"[Patch:{_PATCH_KEY}] cannot pin the {label} Triton kernels in {group.module} "
                        f"({exc!r}); their first launch would autotune in process. Turn the fusion or "
                        f"{_ARG} off."
                    ) from exc
                # moe_permute_fusion launches Primus' copy, so a missing or changed TE module is skipped.
                log_rank_0(f"[Patch:{_PATCH_KEY}] {group.module} not pinnable ({exc!r}); skipping it.")
    done = {}
    for label, group, found in resolved:
        done[group.module] = _pin(group, found)
        log_rank_0(f"[Patch:{_PATCH_KEY}] {label}: pinned in {group.module}: {done[group.module]}")
    return done


def _condition(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if not getattr(args, _ARG, False):
        return False
    if not _wanted_groups(args):
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] {_ARG} is set but neither apply_rope_fusion (with "
            "multi_latent_attention) nor moe_permute_fusion (or turbo_sync_free_moe_stage >= 1) is enabled: "
            "nothing to pin."
        )
        return False
    return True


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Pin the autotuned Triton kernels of the MLA YaRN RoPE fusion and the MoE permutation "
        "fusion to one config each (no in-process autotune)."
    ),
    condition=_condition,
    tags=["triton", "rope", "moe"],
)
def patch_pinned_triton_configs(ctx: PatchContext) -> None:
    warn_if_not_gfx1250(_PATCH_KEY, _ARG)
    # run_patches logs and swallows Exception, and an unpinned kernel would autotune in process on its first
    # launch, so abort the run instead. SystemExit is a BaseException and is not caught by run_patches.
    try:
        _apply_pinned_configs(get_args(ctx))
    except Exception as exc:
        log_rank_0(f"[Patch:{_PATCH_KEY}] FATAL: {exc}")
        raise SystemExit(f"[{_PATCH_KEY}] {exc}") from exc
