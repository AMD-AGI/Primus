###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MoE identity-sort short-circuit patch.

Migrated from the source patch ``megatron_moe_skip_identity_sort.patch``
(mpo branch).

In the all-to-all MoE token dispatcher, ``sort_chunks_by_idxs`` is called with
``sort_input_by_local_experts`` (dispatch) and ``restore_output_by_local_experts``
(combine). When the EP/TP topology yields an *identity* permutation
(``[0, 1, ..., N-1]``) -- the common ``EP=1`` / ``TP=1`` case -- the call
degrades into a full-tensor ``split`` + ``cat`` round-trip that produces an
output identical to its input. This patch short-circuits those calls.

Rather than editing ``third_party/Megatron-LM`` in place, we wrap the
``sort_chunks_by_idxs`` symbol bound inside the ``token_dispatcher`` module.
The identity decision is cached per index tensor (the topology indices are
created once per dispatcher and reused every step), so the ``torch.equal``
probe -- and the device sync it implies -- happens at most once per tensor.

Gate:
    ``MOE_SKIP_IDENTITY_SORT`` (default ``1`` / enabled; set ``0`` to disable).
"""

import os
import weakref

import torch

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0

# Cache the identity decision keyed on the index tensor object. The topology
# index tensors persist for the dispatcher's lifetime, so a WeakKeyDictionary
# evicts entries automatically once a dispatcher is freed.
_IDENTITY_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _skip_enabled(_ctx: PatchContext) -> bool:
    return os.environ.get("MOE_SKIP_IDENTITY_SORT", "1") != "0"


def _is_identity(idxs) -> bool:
    if idxs is None or not torch.is_tensor(idxs) or idxs.dim() != 1 or idxs.numel() == 0:
        return False
    cached = _IDENTITY_CACHE.get(idxs)
    if cached is not None:
        return cached
    result = bool(
        torch.equal(idxs, torch.arange(idxs.numel(), device=idxs.device, dtype=idxs.dtype))
    )
    try:
        _IDENTITY_CACHE[idxs] = result
    except TypeError:
        # Some tensors are not weak-referenceable; fall back to no caching.
        pass
    return result


def _make_wrapped_sort(orig_sort):
    def sort_chunks_by_idxs(input, split_sizes, sorted_idxs, probs=None, fused=False):
        # Identity permutation -> output == input, permuted_probs == probs.
        if _is_identity(sorted_idxs):
            return input, probs
        return orig_sort(input, split_sizes, sorted_idxs, probs=probs, fused=fused)

    return sort_chunks_by_idxs


@register_patch(
    "megatron.moe.skip_identity_sort",
    backend="megatron",
    phase="before_train",
    description=(
        "Short-circuit sort_chunks_by_idxs in the MoE token dispatcher when the "
        "local-expert permutation is identity (e.g. EP=1/TP=1); gated by "
        "MOE_SKIP_IDENTITY_SORT (default on)."
    ),
    condition=_skip_enabled,
)
def patch_moe_skip_identity_sort(ctx: PatchContext):
    del ctx

    try:
        from megatron.core.transformer.moe import token_dispatcher as td_mod
    except ImportError as exc:
        warning_rank_0(
            f"[Patch:megatron.moe.skip_identity_sort] token_dispatcher not "
            f"importable; skipping: {exc}"
        )
        return

    orig_sort = getattr(td_mod, "sort_chunks_by_idxs", None)
    if orig_sort is None:
        warning_rank_0(
            "[Patch:megatron.moe.skip_identity_sort] sort_chunks_by_idxs not bound "
            "in token_dispatcher; skipping."
        )
        return

    if getattr(orig_sort, "_primus_skip_identity_sort", False):
        return

    wrapped = _make_wrapped_sort(orig_sort)
    wrapped._primus_skip_identity_sort = True
    td_mod.sort_chunks_by_idxs = wrapped
    log_rank_0(
        "[Patch:megatron.moe.skip_identity_sort] Patched "
        "token_dispatcher.sort_chunks_by_idxs to skip identity permutations."
    )
