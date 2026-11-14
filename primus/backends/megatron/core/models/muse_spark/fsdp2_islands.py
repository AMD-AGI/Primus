###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark under torch FSDP2: its fp32 "dtype islands" as their own FSDP2 groups.

Muse Spark keeps the router weight and the residual gates fp32 in a bf16 model (HF's
``_keep_in_fp32_modules``). Their classes set ``fsdp2_dtype_island = True``. The patch
``megatron.muse_spark.fsdp2_dtype_islands`` (Muse Spark + ``use_torch_fsdp2`` only) wraps
the ``fully_shard`` that Primus's FSDP2 wrapper calls, so each wrapped module first gets
its islands sharded as their own groups (:func:`shard_dtype_islands`). Other models and
the shared wrapper are unchanged.

Contract of an island class: its group uses ``param_dtype=None`` and
``cast_forward_inputs=False``, so its forward receives the activations in the enclosing
group's compute dtype (e.g. bf16 inputs to an fp32 router). The island's forward must cast
its inputs to its own parameter dtype itself (Muse Spark's router computes
``F.linear(x.float(), W.float())``; the residual gates combine in fp32 and cast
the result back to the residual's dtype), or the island mixes dtypes.

Islands are not in the wrapper's prefetch lists: FSDP2 unshards each in its own
pre-forward / pre-backward hook. Their all-gathers are tiny (a few KB-MB per layer).
"""

import torch

from primus.core.utils.module_utils import log_rank_0, warning_rank_0

__all__ = ["ISLAND_ATTR", "shard_dtype_islands", "with_dtype_islands"]

ISLAND_ATTR = "fsdp2_dtype_island"


try:
    from torch.distributed.fsdp import FSDPModule as _FSDPModule
    from torch.distributed.fsdp import MixedPrecisionPolicy
except ImportError:  # older torch
    from torch.distributed._composable.fsdp import FSDPModule as _FSDPModule
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy


def _is_fsdp_module(m) -> bool:
    return isinstance(m, _FSDPModule)


def _island_kwargs(fsdp_kwargs: dict, dtype: torch.dtype) -> dict:
    """fsdp kwargs for a group of ``dtype`` params kept apart from the model's policy.

    The group keeps its own dtype end to end: no param cast for the all-gather and a
    reduce-scatter in ``dtype`` (e.g. an fp32 router weight is neither cast to bf16
    for compute nor reduced in bf16, whatever the model's ``mp_policy`` is). Inputs are not
    cast (``cast_forward_inputs=False``): the island's forward casts them itself (the
    contract in this module's docstring).
    """
    kw = dict(fsdp_kwargs)
    kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=dtype, cast_forward_inputs=False)
    return kw


_ISLAND_WARNED_ATTR = "_primus_fsdp2_island_warned"  # set once its "no group" warning is logged
WRAPPED_ATTR = "_primus_dtype_islands"  # set on the fully_shard wrapper (patch idempotence)


def shard_dtype_islands(module: torch.nn.Module, fsdp_kwargs: dict, fully_shard) -> list:
    """Shard the opted-in dtype "islands" under ``module`` as their own FSDP2 groups.

    ``fully_shard`` is the FSDP2 sharding function (the one the wrapper calls).

    Opt-in only: a module class sets ``fsdp2_dtype_island = True`` when its *direct*
    parameters must keep their own dtype, whatever it is (e.g. Muse Spark's fp32 router
    weight and residual gates, kept fp32 like HF's ``_keep_in_fp32_modules``). Each such module
    is sharded on its own with ``_island_kwargs`` (its own param dtype for compute and
    reduce-scatter). This holds for every model policy: with ``params_dtype=fp32`` +
    bf16 compute (``mp_policy`` param/reduce bf16) the islands are still not cast or
    reduced in bf16.

    FSDP2 groups are per module, so a child of an island that has (not yet sharded)
    parameters, e.g. the bf16 norm inside the gated residual, is split off first with
    the model kwargs.

    No module opts in -> a no-op: nothing is checked or split, and the caller's
    ``fully_shard`` is the only sharding applied.

    An island's dtype is that of its direct *trainable* parameters; when it has any,
    its frozen direct parameters do not decide (or have to match) the dtype. A fully
    frozen island gets its own group only when its frozen
    parameters share one dtype. With mixed frozen dtypes it gets no island policy (never
    an error; FSDP2 does not check frozen dtypes): its parameters are sharded with the
    model's ``fsdp_kwargs``, as a child group of an enclosing island (the island itself
    or the direct child that holds it) or in the enclosing wrapped module.
    Raises ValueError when ``module`` itself opts in (it has to be a submodule) or an
    island's direct trainable parameters mix dtypes.
    Returns the new groups in creation order (children before parents).
    """
    # one walk: the islands with their names (the names are read only by the errors and
    # the warnings)
    pairs = [
        (name, m)
        for name, m in module.named_modules()
        if getattr(m, ISLAND_ATTR, False) and not _is_fsdp_module(m)
    ]
    if not pairs:
        return []
    if any(m is module for _, m in pairs):
        raise ValueError(f"FSDP2: {type(module).__name__} sets {ISLAND_ATTR} but is the module being wrapped")

    managed = set()
    for m in module.modules():
        if m is not module and _is_fsdp_module(m):
            managed.update(id(p) for p in m.parameters())

    def _unmanaged(m, recurse=True):
        return [p for p in m.parameters(recurse=recurse) if id(p) not in managed]

    # the root is never an island (rejected above), so every island has a non-empty name
    plan = []
    for name, m in pairs:  # module.modules() order: parents first
        own = _unmanaged(m, recurse=False)
        trainable_dtypes = {p.dtype for p in own if p.requires_grad}
        if len(trainable_dtypes) > 1:
            raise ValueError(
                f"FSDP2: island {name} directly owns trainable parameters of several dtypes "
                f"{sorted(str(d) for d in trainable_dtypes)}; FSDP2 needs one dtype per group"
            )
        if trainable_dtypes:
            plan.append((m, next(iter(trainable_dtypes))))
            continue
        # Fully frozen island (e.g. a frozen router in a partial finetune): keep it in its
        # own dtype when it has one. Otherwise the island opted in but gets no group of its
        # own: say so, as nothing else would show it.
        # This function runs once per wrapped submodule and again on the root, so an
        # island without a group is seen again: warn once per module.
        frozen_dtypes = {p.dtype for p in own}
        if len(frozen_dtypes) == 1:
            plan.append((m, next(iter(frozen_dtypes))))
        elif not getattr(m, _ISLAND_WARNED_ATTR, False):
            setattr(m, _ISLAND_WARNED_ATTR, True)
            if not own:
                warning_rank_0(
                    f"FSDP2: {ISLAND_ATTR} module {name} owns no unsharded parameters directly; "
                    "it gets no group of its own (its children follow their own wrapping)"
                )
            else:
                warning_rank_0(
                    f"FSDP2: frozen {ISLAND_ATTR} module {name} mixes dtypes "
                    f"{sorted(str(d) for d in frozen_dtypes)}; it gets no island dtype policy: its parameters "
                    "are sharded with the model's FSDP settings (as a child group of an enclosing "
                    "island, or in the enclosing wrapped module); FSDP2 does not check frozen dtypes"
                )

    groups = []

    def _shard(m, kw):
        fully_shard(m, **kw)
        managed.update(id(p) for p in m.parameters())
        groups.append(m)

    for m, dtype in reversed(plan):  # children before parents
        for child in m.children():
            if _unmanaged(child):
                _shard(child, fsdp_kwargs)  # split off with the model's settings
        _shard(m, _island_kwargs(fsdp_kwargs, dtype))
    return groups


def island_group_counts(groups) -> tuple:
    """(groups of ``fsdp2_dtype_island`` modules, other split-off child groups) among ``groups``
    from :func:`shard_dtype_islands`. An island module is counted as an island even when it is
    sharded with the model's settings (a frozen island with mixed dtypes gets no island policy)."""
    n_islands = sum(1 for m in groups if getattr(m, ISLAND_ATTR, False))
    return n_islands, len(groups) - n_islands


def with_dtype_islands(fully_shard):
    """``fully_shard`` that first shards the islands under the module as their own groups
    (the wrapper calls it per wrapped submodule and on the root)."""

    def fully_shard_with_islands(module, **kwargs):
        groups = shard_dtype_islands(module, kwargs, fully_shard)
        if groups:
            n_islands, n_children = island_group_counts(groups)
            log_rank_0(
                f"FSDP2: {n_islands} {ISLAND_ATTR} groups (own dtype unless frozen with mixed dtypes) + "
                f"{n_children} split-off children "
                f"(the model's FSDP settings) under {type(module).__name__}; not prefetched"
            )
        return fully_shard(module, **kwargs)

    setattr(fully_shard_with_islands, WRAPPED_ATTR, True)
    return fully_shard_with_islands
