###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Pinned Triton configs for the MLA YaRN RoPE fusion and the MoE permutation fusion
=================================================================================

``apply_rope_fusion`` (multi-latent attention) and ``moe_permute_fusion`` launch
Triton kernels wrapped in ``triton.autotune``. On the first launch per autotune
key every candidate config is compiled and benchmarked inside the training
process: BLOCK_H 1..128 for the four MLA RoPE kernels
(``megatron/core/fusions/fused_mla_yarn_rope_apply.py``), BLOCK_SIZE 64..4096 for
the permutation kernels (Primus' copy of Transformer Engine's
``common/triton/permutation.py``, which ``moe_permute_fusion`` routes to, and TE's
own module). In-process autotune benchmarking is a known hang class on gfx1250
(MI455X).

With ``use_primus_pinned_triton_configs: true`` this patch replaces each
autotuner's config list IN PLACE by ONE config picked from the stock candidates,
so ``Autotuner.run`` takes its single-config branch: no benchmarking, no
``restore_value`` clones, no disk cache, one binary per kernel signature. The
kernel sources are untouched, and every reference to the autotuner object
(module globals, ``from ... import`` bindings in the launch wrappers) sees the
pinned list.

Pinned configs (picked on gfx1250 at DeepSeek-V3 shapes: 128 heads, qk 128 +
rope 64, v 128, hidden 7168, top-8 routing):

  MLA RoPE     rotary_{fwd,bwd}_{q,kv}_kernel        BLOCK_H 32      num_warps 4  num_stages 3
  permutation  _permute_kernel, _unpermute_kernel,  BLOCK_SIZE 1024 num_warps 4  num_stages 3
               _unpermute_bwd_with_merging_probs_kernel, _sort_chunks_by_map_kernel

The permutation row-id-map kernels are not autotuned upstream and are left alone.

Only the kernels of the fusions that are enabled (``multi_latent_attention`` +
``apply_rope_fusion``, ``moe_permute_fusion`` or ``turbo_sync_free_moe_stage``
>= 1, which turns ``moe_permute_fusion`` on during argument validation) are
pinned. Every requested kernel is resolved and checked before any autotuner is
changed: if one cannot be pinned (Triton or Megatron layout changed), nothing is
pinned and the run aborts (``SystemExit``) at ``before_train`` instead of
letting the first launch autotune. Transformer Engine's own permutation module
is optional and skipped when it is absent.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.triton.pinned_configs"
_ARG = "use_primus_pinned_triton_configs"

# (module, kernels, meta-parameters, num_warps, num_stages)
_ROPE_GROUP = (
    "megatron.core.fusions.fused_mla_yarn_rope_apply",
    ("rotary_fwd_q_kernel", "rotary_bwd_q_kernel", "rotary_fwd_kv_kernel", "rotary_bwd_kv_kernel"),
    {"BLOCK_H": 32},
    4,
    3,
)
_PERMUTE_KERNELS = (
    "_permute_kernel",
    "_unpermute_kernel",
    "_unpermute_bwd_with_merging_probs_kernel",
    "_sort_chunks_by_map_kernel",
)
_TE_PERMUTE_MODULE = "transformer_engine.common.triton.permutation"
_PERMUTE_GROUPS = (
    # What moe_permute_fusion routes to (patch "megatron.moe.permute_fusion").
    (
        "primus.backends.transformer_engine.pytorch.common.triton.permutation",
        _PERMUTE_KERNELS,
        {"BLOCK_SIZE": 1024},
        4,
        3,
    ),
    # Transformer Engine's own copy (same kernels), reached by any direct TE caller.
    (_TE_PERMUTE_MODULE, _PERMUTE_KERNELS, {"BLOCK_SIZE": 1024}, 4, 3),
)


def _moe_permute_fusion_enabled(args) -> bool:
    # The patches run before validate_args, which turns moe_permute_fusion on for turbo_sync_free_moe_stage >= 1.
    return (
        bool(getattr(args, "moe_permute_fusion", False))
        or (getattr(args, "turbo_sync_free_moe_stage", 0) or 0) >= 1
    )


def wanted_groups(args) -> List[Tuple[str, Tuple]]:
    """[(label, groups)] for every enabled fusion whose kernels get pinned."""
    groups = []
    if getattr(args, "multi_latent_attention", False) and getattr(args, "apply_rope_fusion", False):
        groups.append(("mla_rope", (_ROPE_GROUP,)))
    if _moe_permute_fusion_enabled(args):
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


def resolve_group(group: Tuple) -> List[Tuple[str, Any]]:
    """[(kernel name, autotuner)] of one (module, kernels, meta, num_warps, num_stages) group, checked for pinning.

    Kernels that are not autotuned in this version are left out (nothing tunes them in process). Raises if the
    module cannot be imported, a kernel is missing or an autotuner has no argument for a pinned meta-parameter.
    Changes nothing.
    """
    modname, kernels, meta, _, _ = group
    mod = importlib.import_module(modname)
    found = []
    for name in kernels:
        obj = getattr(mod, name, None)
        if obj is None:
            raise RuntimeError(f"{modname}.{name} not found")
        tuner = _find_autotuner(obj)
        if tuner is None:
            continue
        missing = [k for k in meta if k not in tuner.arg_names]
        if missing:
            raise RuntimeError(f"{modname}.{name}: no argument(s) {missing} to pin")
        found.append((name, tuner))
    return found


def pin_autotuner(tuner: Any, meta: Dict[str, int], num_warps: int, num_stages: int) -> None:
    """Replace ``tuner.configs`` by one ``triton.Config`` (in place) and drop tuning state."""
    import triton

    tuner.configs = [triton.Config(dict(meta), num_warps=num_warps, num_stages=num_stages)]
    tuner.cache.clear()
    if hasattr(tuner, "early_config_prune"):
        tuner.early_config_prune = None
    if hasattr(tuner, "perf_model"):
        tuner.perf_model = None


def pin_group(group: Tuple) -> List[str]:
    """Pin every autotuned kernel of one group (all checked first). Returns the pinned names."""
    _, _, meta, num_warps, num_stages = group
    found = resolve_group(group)
    for _, tuner in found:
        pin_autotuner(tuner, meta, num_warps, num_stages)
    return [name for name, _ in found]


def describe_pins(group: Tuple) -> Dict[str, List[str]]:
    """{kernel: [str(config), ...]} as installed (for logs and tests)."""
    modname, kernels, _, _, _ = group
    mod = importlib.import_module(modname)
    out = {}
    for name in kernels:
        tuner = _find_autotuner(getattr(mod, name, None))
        out[name] = [str(c) for c in tuner.configs] if tuner is not None else ["<not autotuned>"]
    return out


def apply_pinned_configs(args) -> Dict[str, Dict[str, List[str]]]:
    """Pin the kernels of every enabled fusion; returns {module: describe_pins(...)}.

    Every group is resolved and checked first; if one kernel cannot be pinned, RuntimeError is raised and no
    autotuner has been changed.
    """
    resolved = []
    for label, groups in wanted_groups(args):
        for group in groups:
            modname = group[0]
            try:
                found = resolve_group(group)
            except ModuleNotFoundError as exc:
                if modname == _TE_PERMUTE_MODULE:
                    # Optional: moe_permute_fusion goes through Primus' copy.
                    log_rank_0(f"[Patch:{_PATCH_KEY}] {modname} not importable ({exc}); skipping it.")
                    continue
                raise RuntimeError(f"[Patch:{_PATCH_KEY}] cannot pin {label}: {exc!r}") from exc
            except Exception as exc:  # noqa: BLE001 - re-raised with the reason
                raise RuntimeError(
                    f"[Patch:{_PATCH_KEY}] cannot pin the {label} Triton kernels in {modname} ({exc!r}); "
                    f"their first launch would autotune in process. Turn the fusion or {_ARG} off."
                ) from exc
            resolved.append((label, group, found))
    done: Dict[str, Dict[str, List[str]]] = {}
    for label, group, found in resolved:
        modname, _, meta, num_warps, num_stages = group
        for _, tuner in found:
            pin_autotuner(tuner, meta, num_warps, num_stages)
        done[modname] = describe_pins(group)
        pinned = [name for name, _ in found]
        log_rank_0(f"[Patch:{_PATCH_KEY}] {label}: pinned {pinned} in {modname} -> {done[modname]}")
    return done


def _condition(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if not getattr(args, _ARG, False):
        return False
    if not wanted_groups(args):
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
    # After megatron.moe.permute_fusion (default priority 50) has imported Primus' permutation copy.
    # Pinning mutates the autotuner objects in place, so the order is not load-bearing either way.
    priority=60,
    condition=_condition,
    tags=["triton", "rope", "moe"],
)
def patch_triton_pinned_configs(ctx: PatchContext) -> None:
    # run_patches logs and swallows Exception, and an unpinned kernel would autotune in process on its first
    # launch, so abort the run instead. SystemExit is a BaseException and is not caught by run_patches.
    try:
        apply_pinned_configs(get_args(ctx))
    except Exception as exc:
        log_rank_0(f"[Patch:{_PATCH_KEY}] FATAL: {exc}")
        raise SystemExit(f"[{_PATCH_KEY}] {exc}") from exc
