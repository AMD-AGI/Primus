###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Split tensors larger than 2^28 elements before Megatron's multi-tensor kernels.

PROBLEM
-------
Megatron computes the gradient norm and applies gradient clipping with the multi-tensor
kernels of Transformer Engine (``multi_tensor_l2norm``, ``multi_tensor_scale``) or apex
(``amp_C``). Their per-tensor sizes are 32-bit, so a tensor with more than 2^31 elements
is read or written out of bounds.

The distributed optimizer hands these kernels one flat view per gradient-buffer shard.
With expert data-parallel size 1 that shard holds every local expert's gradient: the
Kimi K3.1 Pro 4-layer proxy on one 4 x MI455X node (896 experts, EP 4) passes a
4,932,501,504-element BF16 tensor. The out-of-bounds access is a GPU page fault
(``Page not present``, UTCL2 client TCP or SDMA0) in iteration 1 when the stray address is
unmapped, and a silently wrong gradient norm when it is not. 224 experts (1.85 B elements
per tensor) is clean, 448 (3.7 B) faults.

SOLUTION
--------
Wrap the module-level kernel handles that Megatron calls through ``multi_tensor_applier``
so that every contiguous tensor above ``_MAX_NUMEL`` elements is passed as consecutive
views of at most ``_MAX_NUMEL`` elements, the same split applied to every tensor list.
2^28 keeps both the element count and the byte size (FP32) under 2^31. Tensor lists
without such a tensor go to the kernel unchanged.

The total L2 norm and in-place scaling are unchanged by the split. A per-tensor L2 norm
would not be, so that case raises instead of splitting; Megatron only asks for the total.

TURNING OFF
-----------
``PRIMUS_MULTI_TENSOR_SPLIT=0``.
"""

from __future__ import annotations

import os

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.optimizer.multi_tensor_large_tensor_split"
_MAX_NUMEL = 1 << 28

# (module, attribute) handles Megatron passes to multi_tensor_applier.
_TARGETS = (
    ("megatron.core.optimizer.clip_grads", "l2_norm_impl"),
    ("megatron.core.optimizer.clip_grads", "multi_tensor_scale_impl"),
    ("megatron.core.optimizer.clip_grads", "multi_tensor_scale_tensor_impl"),
    ("megatron.core.optimizer.optimizer", "multi_tensor_scale_impl"),
)


def _enabled(ctx: PatchContext) -> bool:
    return os.environ.get("PRIMUS_MULTI_TENSOR_SPLIT", "1") != "0"


def _needs_split(tensor) -> bool:
    return tensor.numel() > _MAX_NUMEL and tensor.is_contiguous()


def _split_tensor_lists(tensor_lists):
    """Split each over-limit tensor into views, position by position across the lists."""
    split_lists = [[] for _ in tensor_lists]
    for tensors in zip(*tensor_lists):
        if any(_needs_split(t) for t in tensors):
            numels = {t.numel() for t in tensors}
            if len(numels) != 1 or not all(t.is_contiguous() for t in tensors):
                raise RuntimeError(
                    "multi-tensor split: tensors at one position must be contiguous and the "
                    f"same size, got {[tuple(t.shape) for t in tensors]}"
                )
            for out, t in zip(split_lists, tensors):
                out.extend(t.view(-1).split(_MAX_NUMEL))
        else:
            for out, t in zip(split_lists, tensors):
                out.append(t)
    return split_lists


def make_large_tensor_safe(op, *, is_l2norm: bool = False):
    """Return ``op`` wrapped so that tensors above ``_MAX_NUMEL`` elements are split."""
    if op is None or getattr(op, "_primus_large_tensor_safe", False):
        return op

    def safe_op(chunk_size, noop_flag, tensor_lists, *args):
        if not any(_needs_split(t) for tensors in tensor_lists for t in tensors):
            return op(chunk_size, noop_flag, tensor_lists, *args)
        if is_l2norm and args and bool(args[0]):
            raise RuntimeError(
                "multi-tensor split: a per-tensor L2 norm of a tensor above "
                f"{_MAX_NUMEL} elements is not supported"
            )
        return op(chunk_size, noop_flag, _split_tensor_lists(tensor_lists), *args)

    safe_op._primus_large_tensor_safe = True
    safe_op.__wrapped__ = op
    return safe_op


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Split tensors above 2^28 elements before TE/apex multi-tensor L2-norm and scale "
        "kernels (32-bit sizes; out-of-bounds access on >2^31-element gradient shards)"
    ),
    condition=_enabled,
    tags=["optimizer", "rocm"],
)
def patch_multi_tensor_large_tensor_split(ctx: PatchContext) -> None:
    import importlib

    wrapped = []
    for module_name, attr in _TARGETS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        key = f"{_PATCH_KEY}:{attr}"
        if is_patched(module, key) or getattr(module, attr, None) is None:
            continue
        setattr(
            module,
            attr,
            make_large_tensor_safe(getattr(module, attr), is_l2norm=attr == "l2_norm_impl"),
        )
        mark_patched(module, key)
        wrapped.append(f"{module_name.rsplit('.', 1)[-1]}.{attr}")

    log_rank_0(
        f"[Patch:{_PATCH_KEY}] split tensors above {_MAX_NUMEL} elements in: "
        f"{', '.join(wrapped) or 'nothing (already applied)'}"
    )
