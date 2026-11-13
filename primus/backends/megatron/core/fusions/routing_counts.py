###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``routing_map.sum(dim=0)`` (tokens per expert) in at most one launch per routing map.

A MoE layer sums the same boolean routing map up to three times (aux loss via
``get_tokens_per_expert_and_token_count``, the expert-bias ``local_tokens_per_expert``
update, the token dispatcher's ``preprocess``), each an ATen fill + ``reduce_kernel``.
The counts are integers, so any summation order gives the same int64 values. Here:

* ``tokens_per_expert(rmap)`` returns a cached int64 ``[E]`` tensor for a map it has seen
  (same object, same ``_version``), otherwise one Triton launch computes it;
* ``register(rmap, counts)`` lets a producer that knows the counts (the even-routing
  kernel writes them in its own launch) hand them over without any launch.

Callers only read the result (as with ATen's fresh tensor, nothing writes into it).
``PRIMUS_FUSED_ROUTING_COUNTS=0`` keeps ATen's sums.
"""

from __future__ import annotations

import os
import weakref

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PRIMUS_FUSED_ROUTING_COUNTS", "1") != "0"

_CACHE = []  # [(weakref(map), version, counts)], most recent few


@triton.jit
def _count_kernel(MAP, OUT, N, E, BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    cm = cols < E
    acc = tl.zeros([BLOCK_E], dtype=tl.int32)
    for r0 in range(0, N, BLOCK_N):
        rows = r0 + tl.arange(0, BLOCK_N)
        m = (rows < N)[:, None] & cm[None, :]
        v = tl.load(MAP + rows[:, None].to(tl.int64) * E + cols[None, :], mask=m, other=0)
        acc += tl.sum((v != 0).to(tl.int32), axis=0)
    tl.store(OUT + cols, acc.to(tl.int64), mask=cm)


def _lookup(rmap):
    for ref, ver, cnt in _CACHE:
        if ref() is rmap and ver == rmap._version:
            return cnt
    return None


def register(rmap: torch.Tensor, counts: torch.Tensor) -> None:
    if not ENABLED:
        return
    _CACHE[:] = [e for e in _CACHE if e[0]() is not None][-3:]
    _CACHE.append((weakref.ref(rmap), rmap._version, counts))


def _eligible(rmap) -> bool:
    return (
        ENABLED
        and rmap.is_cuda
        and rmap.dtype == torch.bool
        and rmap.dim() == 2
        and rmap.is_contiguous()
        and rmap.shape[0] < 2**31
    )


def tokens_per_expert(rmap: torch.Tensor) -> torch.Tensor:
    """``rmap.sum(dim=0)`` (int64) for a 2-D boolean routing map."""
    if not _eligible(rmap):
        return rmap.sum(dim=0)
    cnt = _lookup(rmap)
    if cnt is not None:
        return cnt
    N, E = rmap.shape
    cnt = torch.empty(E, dtype=torch.int64, device=rmap.device)
    BLOCK_E = min(triton.next_power_of_2(E), 128)
    _count_kernel[(triton.cdiv(E, BLOCK_E),)](
        rmap.view(torch.uint8), cnt, N, E, BLOCK_N=64, BLOCK_E=BLOCK_E, num_warps=8
    )
    register(rmap, cnt)
    return cnt


def _patched_get_tokens_per_expert_and_token_count(orig):
    from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region

    def get_tokens_per_expert_and_token_count(routing_map, reduce_group, topk=None, with_padding_mask=False):
        if not _eligible(routing_map):
            return orig(routing_map, reduce_group, topk, with_padding_mask)
        local_tokens_per_expert = tokens_per_expert(routing_map)
        global_tokens_per_expert = reduce_from_tensor_model_parallel_region(local_tokens_per_expert, reduce_group)
        if with_padding_mask:
            local_num_tokens = local_tokens_per_expert.sum() / topk
            total_num_tokens = global_tokens_per_expert.sum() / topk
        else:
            local_num_tokens = routing_map.shape[0]
            total_num_tokens = local_num_tokens * reduce_group.size()
        return global_tokens_per_expert, local_num_tokens, total_num_tokens

    get_tokens_per_expert_and_token_count._primus_routing_counts = True
    return get_tokens_per_expert_and_token_count


def install() -> None:
    """Route Megatron's three per-layer ``routing_map.sum(dim=0)`` through ``tokens_per_expert``."""
    if not ENABLED:
        return False
    import inspect
    import textwrap

    import megatron.core.transformer.moe.router as mrouter
    import megatron.core.transformer.moe.token_dispatcher as mtd

    f = getattr(mrouter, "get_tokens_per_expert_and_token_count", None)
    if f is not None and not getattr(f, "_primus_routing_counts", False):
        mrouter.get_tokens_per_expert_and_token_count = _patched_get_tokens_per_expert_and_token_count(f)

    cls = mtd.MoEAlltoAllTokenDispatcher
    pre = cls.preprocess
    if getattr(pre, "_primus_routing_counts", False):
        return True
    old = "num_local_tokens_per_expert = routing_map.sum(dim=0).long()"
    try:
        src = textwrap.dedent(inspect.getsource(pre))
    except (OSError, TypeError):
        return False
    if src.count(old) != 1:
        return False
    src = src.replace(old, "num_local_tokens_per_expert = _primus_tokens_per_expert(routing_map)")
    g = dict(mtd.__dict__)
    g["_primus_tokens_per_expert"] = tokens_per_expert
    loc = {}
    exec(compile(src, inspect.getsourcefile(pre) or "<token_dispatcher>", "exec"), g, loc)
    new = loc["preprocess"]
    new._primus_routing_counts = True
    cls.preprocess = new
    return True
