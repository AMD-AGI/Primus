###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 hash-router input_ids PP broadcast (upfront/pre-loop).

V4's hash-routed MoE layers (the first ``num_hash_layers`` MoE layers)
look up a static ``tid2eid`` table and therefore need the raw
``input_ids`` on every PP stage that owns one. Megatron's
:func:`pretrain_gpt.get_batch` only loads tokens on the first / last PP
stage; middle stages short-circuit to ``return None, None, None, ...``.

Plan-2 P19 attempted two simpler hooks before this one:

1. **In-forward broadcast** inside :meth:`DeepseekV4Model.forward`.
2. **Per-call broadcast** wrapping :func:`pretrain_gpt.get_batch`.

Both work for the non-interleaved 1F1B schedule (``PP>1``,
``VPP=1``) but deadlock the interleaved-1F1B / VPP schedule. The reason:
the interleaved scheduler issues a single ``recv_forward`` *before* the
warm-up loop on every non-first PP rank (see
:func:`forward_backward_pipelining_with_interleaving` in
``megatron/core/pipeline_parallel/schedules.py:1363-1392``). PP rank > 0
therefore parks in that ``recv_forward.wait()`` until PP rank 0's first
``send_forward`` arrives. PP rank 0 meanwhile enters the warm-up loop,
calls ``forward_step`` -> ``get_batch`` -> ``dist.broadcast`` and gets
stuck waiting for PP rank > 0 to issue the matching broadcast — which
will never happen because PP rank > 0 is itself blocked on
``recv_forward``. The non-interleaved 1F1B schedule does not hit this
because its pre-loop ``recv_forward`` is *inside* each warm-up iter
(``schedules.py:2128-2156``), so PP rank > 0 reaches its broadcast call
on the same iteration that PP rank 0 issues the matching ``send_forward``
and the broadcast pairs up before either rank stalls.

This patch instead does **all** PP token broadcasts up-front, *before*
the schedule's first ``recv_forward``:

* It wraps :func:`megatron.core.pipeline_parallel.get_forward_backward_func`
  so that every schedule fetched for a train_step is replaced by a
  thin wrapper which:
    1. Pre-loads ``num_microbatches`` × ``num_chunks`` batches by calling
       the original ``pretrain_gpt.get_batch`` on the first / last PP
       stages, and allocating empty token buffers on middle PP stages.
    2. Runs one ``dist.broadcast`` per (chunk, microbatch) on the PP
       group, sourced from PP rank 0. All collectives fire before any
       ``send_forward`` / ``recv_forward`` runs, so they pair up
       deterministically across ranks — no deadlock.
    3. Caches the resulting tuples in a module-local store keyed by
       chunk and microbatch ordinal, then calls the underlying schedule
       with the original ``data_iterator``.
* It wraps :func:`pretrain_gpt.get_batch` so that, while the cache is
  active, calls return the pre-cached tuple for the corresponding
  (vp_stage, microbatch) — bypassing the data iterator. Outside a
  train_step (e.g. during eval not routed through the schedule) the
  wrapper falls back to the original ``get_batch``.
* The cache is reset after each schedule call (success or exception)
  so subsequent train_steps start clean.

Gating: ``model_type == "deepseek_v4"``, ``num_hash_layers > 0``, and
``pipeline_model_parallel_size > 1``. The patch is a strict no-op for
any other model.

Cost analysis (on the V4 BF16 smoke):

* Each pre-broadcast moves ``mbs * seq * 8 B`` per microbatch
  (~1 KiB for ``mbs=1, seq=128``). With ``num_microbatches=16,
  num_chunks=2`` that is ~32 KiB total, dwarfed by activation P2P
  (~32 MiB per microbatch).
* Pre-loading consumes the data_iterator on PP rank 0 / last during
  ``pre_broadcast`` rather than spread across the warm-up. This
  collapses ``batch-generator`` time into a single bursty phase.
"""

from typing import Any, Optional

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


# ---------------------------------------------------------------------------
# Module-local pre-broadcast cache.
#
# ``data[chunk_id][microbatch_id]`` holds the 6-tuple returned by
# :func:`pretrain_gpt.get_batch`:
#   (tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params)
#
# On middle PP stages only ``tokens`` is meaningful; the other fields stay
# ``None`` (matching what the original ``get_batch`` returns there).
# ``consumed[chunk_id]`` tracks how many microbatches have been popped from
# the cache during the current schedule call; it lets us return the right
# entry from the patched ``get_batch`` regardless of which interleaved
# microbatch ordinal is being processed (interleaved / 1F1B both consume
# microbatches in increasing order *per chunk*).
# ---------------------------------------------------------------------------
_V4_PP_TOKEN_CACHE: dict = {
    "active": False,
    "data": [],
    "consumed": [],
}


def _v4_reset_cache() -> None:
    _V4_PP_TOKEN_CACHE["active"] = False
    _V4_PP_TOKEN_CACHE["data"] = []
    _V4_PP_TOKEN_CACHE["consumed"] = []


def _v4_pre_broadcast_step(
    data_iterator: Any,
    num_microbatches: int,
    original_get_batch,
) -> None:
    """Pre-broadcast V4 ``input_ids`` for every (chunk, microbatch) of one step."""
    from megatron.core import parallel_state
    from megatron.training import get_args as _get_args

    args = _get_args()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    src_global = torch.distributed.get_global_rank(pp_group, 0)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_world = parallel_state.get_pipeline_model_parallel_world_size()

    cp_size = max(1, int(getattr(args, "context_parallel_size", 1) or 1))
    mbs = int(args.micro_batch_size)
    seq = int(args.seq_length) // cp_size
    device = torch.device("cuda", torch.cuda.current_device())

    # ``is_first_or_last_pp`` mirrors :func:`pretrain_gpt.get_batch`'s
    # gate: the original returns real tokens only on the first / last PP
    # stage. (MTP can also build the dataset on a middle stage; that
    # case is left to the original short-circuit because the smoke runs
    # ``mtp_num_layers=0`` and the broadcast logic still tolerates a
    # ``None``-tuple result by falling back to an empty token buffer.)
    is_first_or_last_pp = (pp_rank == 0) or (pp_rank == pp_world - 1)

    if isinstance(data_iterator, list):
        iter_list = data_iterator
        num_chunks = len(iter_list)
        # When VPP is enabled, each chunk has its own ``vp_stage`` index.
        chunk_vp_stages = list(range(num_chunks))
    else:
        iter_list = [data_iterator]
        num_chunks = 1
        # Non-VPP: ``vp_stage`` is ``None`` (matches what
        # ``pretrain_gpt.forward_step`` reads off the model attribute).
        chunk_vp_stages = [None]

    cache_data: list = []
    for chunk_id in range(num_chunks):
        chunk_iter = iter_list[chunk_id]
        chunk_vp_stage = chunk_vp_stages[chunk_id]
        chunk_cache: list = []
        for _mb_id in range(num_microbatches):
            if is_first_or_last_pp and chunk_iter is not None:
                tup = original_get_batch(chunk_iter, chunk_vp_stage)
                tokens = tup[0]
                if tokens is None:
                    # Should not happen on first / last PP for V4 smoke;
                    # fall back to empty buffer and let the broadcast
                    # populate. This keeps us robust against unusual
                    # configs (e.g. dataset-on-rank gating disabled).
                    tokens = torch.empty([mbs, seq], dtype=torch.long, device=device)
                cached = (tokens, tup[1], tup[2], tup[3], tup[4], tup[5])
            else:
                tokens = torch.empty([mbs, seq], dtype=torch.long, device=device)
                cached = (tokens, None, None, None, None, None)

            torch.distributed.broadcast(tokens, src=src_global, group=pp_group)
            chunk_cache.append(cached)
        cache_data.append(chunk_cache)

    _V4_PP_TOKEN_CACHE["data"] = cache_data
    _V4_PP_TOKEN_CACHE["consumed"] = [0] * num_chunks
    _V4_PP_TOKEN_CACHE["active"] = True


def _make_v4_get_batch(original_get_batch):
    """Return a ``get_batch`` wrapper that consumes the pre-broadcast cache."""

    def patched_get_batch(data_iterator: Any, vp_stage: Optional[int] = None):
        if not _V4_PP_TOKEN_CACHE["active"]:
            return original_get_batch(data_iterator, vp_stage)
        chunk_id = vp_stage if vp_stage is not None else 0
        counter = _V4_PP_TOKEN_CACHE["consumed"][chunk_id]
        cached = _V4_PP_TOKEN_CACHE["data"][chunk_id][counter]
        _V4_PP_TOKEN_CACHE["consumed"][chunk_id] += 1
        return cached

    patched_get_batch.__wrapped__ = original_get_batch
    patched_get_batch._v4_pp_get_batch_patched = True
    return patched_get_batch


def _make_v4_pre_broadcast_schedule(original_schedule, original_get_batch):
    """Wrap ``forward_backward_func`` to pre-broadcast V4 tokens up-front."""

    def patched_schedule(*args, **kwargs):
        data_iterator = kwargs.get("data_iterator")
        num_microbatches = int(kwargs.get("num_microbatches", 1) or 1)

        try:
            _v4_pre_broadcast_step(data_iterator, num_microbatches, original_get_batch)
            return original_schedule(*args, **kwargs)
        finally:
            _v4_reset_cache()

    patched_schedule._v4_pp_schedule_wrapped = True
    return patched_schedule


@register_patch(
    "megatron.deepseek_v4.pp_token_pre_broadcast",
    backend="megatron",
    phase="before_train",
    description=(
        "DeepSeek-V4: pre-broadcast input_ids from PP rank 0 across the PP "
        "group up-front in the forward_backward schedule wrapper, before the "
        "first recv_forward, so middle PP stages owning hash-routed MoE "
        "layers can read raw token IDs without deadlocking the interleaved "
        "1F1B / VPP schedule."
    ),
    condition=lambda ctx: (
        getattr(get_args(ctx), "model_type", None) == "deepseek_v4"
        and int(getattr(get_args(ctx), "num_hash_layers", 0) or 0) > 0
        and int(getattr(get_args(ctx), "pipeline_model_parallel_size", 1) or 1) > 1
    ),
    # Ordered after pp_dump_data so its schedule_wrapper does not double-wrap;
    # see ``pp_dump_data_patches.py`` for the priority=100 anchor.
    priority=60,
)
def patch_v4_pp_token_pre_broadcast(ctx: PatchContext):
    """Install the V4 PP token pre-broadcast hooks."""
    import megatron.core.pipeline_parallel as pp_module
    import megatron.training.training as training_module
    import pretrain_gpt

    original_get_batch = pretrain_gpt.get_batch
    if getattr(original_get_batch, "_v4_pp_get_batch_patched", False):
        log_rank_0(
            "[Patch:megatron.deepseek_v4.pp_token_pre_broadcast] get_batch "
            "already patched, skip"
        )
        return

    # Hook 1: replace pretrain_gpt.get_batch with a cache-consuming wrapper.
    # We capture ``original_get_batch`` here so the schedule wrapper can call
    # the *unpatched* implementation during the pre-broadcast phase (the
    # patched version would just hit an empty cache and recurse).
    pretrain_gpt.get_batch = _make_v4_get_batch(original_get_batch)

    # Hook 2: replace get_forward_backward_func so every schedule fetched
    # by ``train_step`` is wrapped with the pre-broadcast.
    original_get_fbf = pp_module.get_forward_backward_func

    def wrapped_get_fbf():
        original_schedule = original_get_fbf()
        if getattr(original_schedule, "_v4_pp_schedule_wrapped", False):
            return original_schedule
        return _make_v4_pre_broadcast_schedule(original_schedule, original_get_batch)

    pp_module.get_forward_backward_func = wrapped_get_fbf
    training_module.get_forward_backward_func = wrapped_get_fbf

    log_rank_0(
        "[Patch:megatron.deepseek_v4.pp_token_pre_broadcast] wrapped "
        "pretrain_gpt.get_batch + get_forward_backward_func; PP rank 0 "
        "broadcasts input_ids up-front (once per microbatch × chunk per "
        "train_step) so middle PP stages owning hash-routed MoE layers "
        "see real token IDs without per-(chunk, microbatch) collectives "
        "racing the interleaved 1F1B P2P sends."
    )
