"""bf16 EC output: cast the *unique* rows, not the expanded ones.

TorchRec's index dedup hands
``torchrec.modules.utils.construct_jagged_tensors`` a **unique-row** embedding
tensor plus ``reverse_indices``, and expands it to one row per token *inside*
the call::

    expanded_indices = reverse_indices.unsqueeze(1).expand(-1, embeddings.size(-1))
    embeddings = torch.gather(embeddings, 0, expanded_indices)

At this model's yambda-5b shape that is 69,256 unique rows against 7,108,550
token rows -- **102.6x**. Everything downstream of the expand on the three big
UIH features runs in bf16 anyway (``modules/dlrm_hstu.py:645`` casts
``item_id`` explicitly; autocast casts the ``cat`` of ``artist_id`` and
``album_id`` at ``_additional_embedding_mlp``), so the fp32 rows are produced
only to be rounded one op later.

What the hoist removes, measured on a both-arms trace
(steady steps 53-55, per rank per
step; the join keys ``(rank, External id)`` because External id is a
per-process namespace):

=========================================================  =========  =========
component                                                     today     hoisted
=========================================================  =========  =========
``aten::gather`` -- the expand                              3.838 ms    ~1.98
``embedding.to(bf16)``  [2.37M, 512]   (dlrm_hstu.py:645)   1.207 ms    deleted
autocast cast [2.37M, 1024] at ``_additional_embedding_mlp``2.521 ms    deleted
``aten::cat`` dim=1 of the two additional features          3.189 ms    ~1.77
new: cast of the unique rows [69k, 512]                          --     0.04 ms
``aten::cat`` of 11 keys under ``SplitWithSizesBackward0``  5.797 ms    ~2.9
the two ``ToCopyBackward0`` partners (1.248 + 2.521)        3.769 ms    deleted
new: one bf16->fp32 upcast of the [T,512] gradient               --     ~3.7 ms
``aten::scatter_add_``                                     18.394 ms  unchanged
=========================================================  =========  =========

**The forward is bitwise exact.** ``gather`` is a pure copy, so
``gather(x, i).to(bf16)`` and ``gather(x.to(bf16), i)`` produce identical bit
patterns, and the same commutation holds for the ``cat`` on the
additional-feature path. ``_bf16ec_verify.py`` checks both as raw int16 bit
patterns, and checks the patched ``construct_jagged_tensors`` against the
unpatched one at production shape.

**The backward is deliberately NOT changed.** Letting the expand's backward run
in bf16 would accumulate ~103 duplicates per unique row in bf16 instead of
fp32 -- a real numerics change needing a convergence A/B, not an SNR check.
:class:`_CastExpandFp32Bwd` therefore owns both halves: the forward casts and
gathers in bf16, and the backward **upcasts the incoming bf16 gradient to fp32
and runs ``scatter_add_`` in fp32**. The accumulator dtype, the accumulator
shape and the set of atomic adds are exactly what they are today. That upcast
is relocated, not added: today the same 7.1M x 512 of rows are upcast anyway,
as the ``ToCopyBackward0`` partners of the two deleted forward casts.

**Only large key-segments are hoisted.** ``construct_jagged_tensors`` splits the
expanded tensor by ``length_per_key``, and not every key's consumer is in bf16:
``DlrmHSTU._item_forward`` runs ``_item_embedding_mlp`` on the candidate-side
features **outside any autocast**, so handing it bf16 would raise on the fp32
weight. Segments below ``min_rows`` are therefore re-expanded from the original
fp32 tensor and substituted back, which is bitwise identical to today for those
keys. At this model's shape that is 8 segments of 1024 rows against 3 of
~2.37M -- 0.1% of the rows, and the re-gather is free.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import gin
import torch

logger: logging.Logger = logging.getLogger(__name__)

_PATCHED: bool = False
_LOGGED: bool = False
# Set on the first hoisted call to the dtype the EC really emits, so
# DlrmHSTU._user_forward can still restore the transducer output to it.
_HOISTED_FROM: Optional[torch.dtype] = None

# Rows per backward chunk; the backward needs an fp32 copy of the gradient and
# slicing bounds that temporary. 0 disables chunking.
_BWD_CHUNK_ROWS: int = 0
# A key-segment smaller than this keeps its original dtype (see the docstring).
_MIN_ROWS: int = 100_000

# Replace the backward's ``scatter_add_`` with the sort-then-segment-reduce
# kernel in ``ec_segment_reduce``.  Default OFF: this is the arm of a ladder,
# not a default, until the ladder says otherwise.  Standalone at the production
# shape it is 16.763 -> 2.668 ms (6.28x) sort included; see that module.
#
# Set by ``enable_segment_reduce()``, which ``make_model`` calls when the gin
# param ``make_model.ec_segment_reduce`` is true -- NOT by an import-time read
# of ``os.environ``.  An import-time constant cannot be reported in gin's
# operative config, so a run could not say what it had been built with; every
# other arm in this stack is a gin param and this one now matches.
_SEGMENT_REDUCE: bool = False
# One-shot dump of the REAL reverse_indices segment-length distribution.  The
# kernel's standalone numbers were taken on a synthetic uniform index draw, and
# both the sort cost and the load balance depend on the real shape, so this is
# how that assumption gets checked against the training path rather than argued.
_SEG_HIST: bool = False
_SEG_ARMED_LOGGED: bool = False
# The histogram is a RUNNING MAX on T, not a one-shot.  A "log the first call"
# guard samples whichever call happens to come first, and at step 0 of window 0
# that is the SMALLEST call in the run -- the first probe reported T=12288
# U=4076 dup=3.01x, an early short-history window, when the call this
# optimization exists for is ~580x larger.  Logging only on a new maximum means
# the LAST line printed describes the dominant call.
_SEG_HIST_MAX_T: int = 0


# ---------------------------------------------------------------------------
# UIH+candidate merge folded into the expand (make_model.ec_merge_uih_candidate).
#
# DlrmHSTU.main_forward merges each (uih, candidate) embedding pair with
# concat_2D_jagged: a full [T_uih + B, 512] bf16 copy per pair in the forward
# (3 pairs, ~2.4M rows each at the yambda-5b shape) plus a split_2D_jagged
# (and a zeros() of the left gradient) per pair in the backward. The rows it
# copies were themselves written one op earlier by this module's gather. A
# concat of gathered rows is a gather with a permuted index, so the merge can
# be done on the 8-byte index instead of the 1 KiB row: build the merged
# per-sample order [uih_b..., cand_b...] of ``reverse_indices`` and let the
# ONE gather write the merged layout directly. Its backward (scatter_add_ /
# segment-reduce in fp32) then consumes the merged gradient with no split.
#
# Each expanded key also gets its OWN _CastExpandFp32Bwd, instead of one
# expand followed by torch.split. That deletes the SplitWithSizesBackward0
# ``aten::cat`` that re-assembled the per-key gradients into one [T, 512]
# buffer for the single backward; each key's reduction reads its own gradient.
#
# Numerics: forward bitwise identical (the concat is a pure copy and cast;
# the candidate rows are rounded fp32->bf16 by the gather's up-front cast
# rather than by the concat's store -- both RNE; eval/eval_ec_merge.py checks
# raw bits). Backward: the same fp32 addends in fp32 accumulators, summed in
# a different order (per key instead of one pass; candidate-row merged
# gradients join the segment reduce instead of the fp32 gather backward).
# A REDUCTION-ORDER change only, on a backward that is already
# atomic/nondeterministic.
# ---------------------------------------------------------------------------
_MERGE_PAIRS: List[Tuple[str, str]] = []
_MERGE_LOGGED: bool = False
# Marker set on a merged expand output. DlrmHSTU.main_forward skips its
# concat_2D_jagged ONLY for a tensor carrying it, so any call that fell back
# (pair not in this call, uih key below min_rows, CW-sharded key, VBE, ...)
# is merged by main_forward exactly as before.
PREMERGED_ATTR: str = "_dlrmv4_uih_cand_premerged"


def enable_ec_merge_uih_candidate(pairs: List[Tuple[str, str]]) -> None:
    """Register the (uih, candidate) embedding pairs to merge in the expand."""
    global _MERGE_PAIRS
    _MERGE_PAIRS = [(str(u), str(c)) for (u, c) in pairs]
    logger.info(
        "DLRMV4 ec_merge_uih_candidate: registered pairs=%s. The 'armed OK' "
        "line prints on the first expand that actually merges.",
        _MERGE_PAIRS,
    )


# DLRM_HSTU_FP8_EC_CAT_FUSE (default on; =0 disables): the ContextualPreprocessor concatenates the
# merged artist_id/album_id expands along dim=1 ([N,512]+[N,512] -> [N,1024],
# 1.68 ms/rank/step at 1n) and the cat's backward hands each expand a
# column-slice gradient that the segment reduce then copies contiguous
# (2 x 0.94 ms). When every key of the registered group is merged in one call
# with equal row counts, ONE expand is issued with the per-key indices
# interleaved row-wise ([idx_a[0], idx_b[0], idx_a[1], ...]); its [k*N, W]
# output viewed as [N, k*W] IS the cat, byte for byte. Each key's values are
# column views of it, tagged with CATFUSED_ATTR so the preprocessor can use
# the base instead of torch.cat. The backward is one segment reduce over the
# contiguous [k*N, W] gradient (same per-row addends; the keys index disjoint
# unique rows, and the stable argsort keeps each segment in ascending token
# order, so the sums are the incumbent's up to the hot-piece atomics).
# With the grouped EC backward armed
# (DLRM_HSTU_FP8_EC_GROUP_BWD; off when absent, turned on by the
# apply_env_bootstrap binding in yambda_5b.gin) the interleaved index is ONE hoisted
# member of _ECExpandGroup, so its [k*N, W] gradient is segment-reduced in
# ACCUM mode into the shared fp32 accumulator; with it off it is one
# _CastExpandFp32Bwd exactly as shipped.
_CAT_GROUP: List[str] = []
CATFUSED_ATTR: str = "_dlrmv4_catfused"
_CAT_LOGGED: bool = False


def enable_ec_cat_fuse(names: List[str]) -> None:
    global _CAT_GROUP
    _CAT_GROUP = [str(n) for n in names]
    logger.info("DLRMV4 ec_cat_fuse: registered group=%s", _CAT_GROUP)


def is_premerged(t: torch.Tensor) -> bool:
    return bool(getattr(t, PREMERGED_ATTR, False))


def _merged_index(
    ri_u: torch.Tensor,
    ri_c: torch.Tensor,
    len_u: torch.Tensor,
    len_c: torch.Tensor,
    n_u: int,
    n_c: int,
) -> torch.Tensor:
    """reverse_indices in merged per-sample order [uih_b..., cand_b...].

    uih token at flat index i of sample b lands at i + off_c[b];
    candidate token at flat index k of sample b lands at k + off_u[b + 1].
    No host sync: the totals are the host-side length_per_key entries.

    DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE (default OFF in code; the gin binding
    turns it on, =0 restores the ~20-kernel chain below): one Triton kernel
    (ops/glue_fusion.py) writes the same integers -- a jagged interleave of
    two index arrays, pure copies. Fail-closed: a raising launch disables it
    and the chain below runs.
    """
    from generative_recommenders.ops import glue_fusion as _gf

    if _gf.merged_index_fusable(ri_u, ri_c, len_u, len_c):
        try:
            out = _gf.merged_index(ri_u, ri_c, len_u, len_c, n_u, n_c)
        except Exception as exc:  # fail closed to the aten chain
            _gf.disable("ec_merged_index", exc)
        else:
            _gf.note_armed(
                "ec_merged_index",
                "EC merged reverse index in one Triton kernel per merged pair "
                "(kill switch DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE=0)",
            )
            return out
    dev = ri_u.device
    len_u = len_u.to(torch.int64)
    len_c = len_c.to(torch.int64)
    off_u = torch.ops.fbgemm.asynchronous_complete_cumsum(len_u)
    off_c = torch.ops.fbgemm.asynchronous_complete_cumsum(len_c)
    samples = torch.arange(len_u.numel(), device=dev, dtype=torch.int64)
    seg_u = torch.repeat_interleave(samples, len_u, output_size=n_u)
    seg_c = torch.repeat_interleave(samples, len_c, output_size=n_c)
    dst_u = torch.arange(n_u, device=dev, dtype=torch.int64) + off_c[seg_u]
    dst_c = torch.arange(n_c, device=dev, dtype=torch.int64) + off_u[seg_c + 1]
    out = torch.empty(n_u + n_c, dtype=ri_u.dtype, device=dev)
    out.scatter_(0, dst_u, ri_u)
    out.scatter_(0, dst_c, ri_c)
    return out


def _merged_construct(
    embeddings: torch.Tensor,
    feats,
    embedding_names: List[str],
    reverse_indices: torch.Tensor,
    out_dtype: torch.dtype,
    length_per_key: List[int],
    offs: List[int],
    kept: List[str],
):
    """construct_jagged_tensors with per-key expands and pre-merged pairs.

    Returns None (caller falls back to the per-call-split path) unless every
    key is 1:1 and at least one registered pair is mergeable in this call.
    """
    from torchrec.sparse.jagged_tensor import JaggedTensor

    if len(set(embedding_names)) != len(embedding_names):
        return None
    pos = {name: i for i, name in enumerate(embedding_names)}
    kept_set = set(kept)
    # A pair merges only when BOTH keys are in this call and the uih key is
    # hoisted: then today's concat output is bf16 (concat takes values_left's
    # dtype), which is exactly what the bf16 gather produces.
    merge = {
        u: c
        for (u, c) in _MERGE_PAIRS
        if u in pos and c in pos and u not in kept_set
    }
    if not merge:
        return None

    w = embeddings.size(-1)
    lengths = feats.lengths().view(-1, feats.stride())
    lengths_tuple = torch.unbind(lengths, dim=0)

    def seg(i: int) -> torch.Tensor:
        return reverse_indices[offs[i]:offs[i] + length_per_key[i]]

    # EC cat-fuse plan (see _CAT_GROUP). Falls back to the per-key layout (and
    # the preprocessor to torch.cat) unless every group key merges in this
    # call with equal merged row counts.
    grp = _CAT_GROUP
    cat_comb: Optional[torch.Tensor] = None
    cat_n = 0
    if len(grp) > 1 and all(g in merge for g in grp):
        sizes = [
            length_per_key[pos[g]] + length_per_key[pos[merge[g]]] for g in grp
        ]
        if len(set(sizes)) == 1:
            cat_n = sizes[0]
            from generative_recommenders.ops import glue_fusion as _gf

            _parts = [
                (
                    seg(pos[g]), seg(pos[merge[g]]),
                    lengths_tuple[pos[g]], lengths_tuple[pos[merge[g]]],
                )
                for g in grp
            ]
            if all(_gf.merged_index_fusable(*p) for p in _parts):
                # DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE: same int64 values,
                # written straight into the stacked (row-interleaved)
                # layout: no per-key buffers, no stack.
                try:
                    cat_comb = _gf.merged_index_stacked(_parts, cat_n)
                except Exception as exc:  # fail closed to the stack below
                    _gf.disable("ec_merged_index", exc)
                    cat_comb = None
                else:
                    _gf.note_armed(
                        "ec_merged_index",
                        "EC merged reverse index in one Triton kernel per "
                        "merged pair (kill switch "
                        "DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE=0)",
                    )
        if len(set(sizes)) == 1 and cat_comb is None:
            cat_comb = torch.stack(
                [
                    _merged_index(
                        seg(pos[g]), seg(pos[merge[g]]),
                        lengths_tuple[pos[g]], lengths_tuple[pos[merge[g]]],
                        length_per_key[pos[g]], length_per_key[pos[merge[g]]],
                    )
                    for g in grp
                ],
                dim=1,
            ).reshape(-1)
    fused_vals: Dict[str, torch.Tensor] = {}

    def cat_views(flat: torch.Tensor, grouped: bool) -> None:
        # [k*N, W] interleaved expand output -> [N, k*W] (the cat) -> per-key
        # column views tagged for ContextualPreprocessor.
        global _CAT_LOGGED
        k = len(grp)
        big = flat.view(cat_n, k * w)
        for q, g in enumerate(grp):
            v = big[:, q * w:(q + 1) * w]
            setattr(v, CATFUSED_ATTR, (big, q, k))
            fused_vals[g] = v
        if not _CAT_LOGGED:
            _CAT_LOGGED = True
            logger.info(
                "DLRMV4 armed OK: ec_cat_fuse ACTIVE -- one interleaved "
                "expand for %s -> [%d, %d]%s", grp, cat_n, k * w,
                " (one member of the grouped EC backward)" if grouped else "",
            )

    values_by: Dict[str, torch.Tensor] = {}
    if _ec_group_enabled():
        names, idxs, casts, premerged = [], [], [], []
        cat_slot = -1
        for i, name in enumerate(embedding_names):
            if cat_comb is not None and name in grp:
                # The whole group is ONE hoisted member (the interleaved
                # index), placed at the first group key's position.
                if cat_slot < 0:
                    cat_slot = len(idxs)
                    idxs.append(cat_comb)
                    casts.append(True)
                continue
            if name in merge:
                j = pos[merge[name]]
                idxs.append(_merged_index(
                    seg(i), seg(j), lengths_tuple[i], lengths_tuple[j],
                    length_per_key[i], length_per_key[j],
                ))
                casts.append(True)
                premerged.append(True)
            else:
                idxs.append(seg(i))
                # kept keys keep the EC's own dtype, exactly as below.
                casts.append(name not in kept_set)
                premerged.append(False)
            names.append(name)
        # Indexed-X L1 (ops/ec_indexed_l1.py; DLRM_HSTU_FP8_EC_INDEXED_L1, default off): None -> unchanged.
        ix = _ix_plan(embeddings, out_dtype, casts, idxs, cat_slot, w, names, premerged)
        if ix is not None:  # the two hoisted members' casts become (truthy) records
            casts[ix.cat.member], casts[ix.item.member] = ix.cat, ix.item
        # E-X2b dh ports (dlrm_v4/train/ec_dgrad_ports.py; DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE,
        # default off): None -> the shipped call, unchanged.
        ex2b = _ex2b_plan(embeddings, out_dtype, casts, idxs, cat_slot, w)
        if ex2b is None:
            outs = list(
                _ECExpandGroup.apply(embeddings, out_dtype, tuple(casts), *idxs)
            )
        else:
            outs = ex2b.apply(embeddings, out_dtype, tuple(casts), idxs)
        if ix is not None:
            ix.tag_item(outs[ix.item.member])  # pre-pop member index, as E-X2b's _Plan.apply
        if cat_slot >= 0:
            flat = outs.pop(cat_slot)
            cat_views(flat, grouped=True)
            if ex2b is not None:
                ex2b.tag_cat(flat, getattr(fused_vals[grp[0]], CATFUSED_ATTR)[0])
            if ix is not None:
                ix.tag_cat(getattr(fused_vals[grp[0]], CATFUSED_ATTR)[0])  # `big`
        for name, values, pm in zip(names, outs, premerged):
            if pm:
                setattr(values, PREMERGED_ATTR, True)
            values_by[name] = values
        # Item (118) PREPROC_DEDUP (default off): tag the item member output and `big` for the preprocessor.
        _ppdd_tag(embeddings, out_dtype, casts, idxs, cat_slot, w, names, premerged, values_by, fused_vals, ix)
    elif cat_comb is not None:
        cat_views(
            _CastExpandFp32Bwd.apply(embeddings, cat_comb, out_dtype),
            grouped=False,
        )

    ret: Dict[str, "JaggedTensor"] = {}
    for i, name in enumerate(embedding_names):
        n = length_per_key[i]
        if name in fused_vals:
            j = pos[merge[name]]
            values = fused_vals[name]
            setattr(values, PREMERGED_ATTR, True)
            jt_lengths = lengths_tuple[i] + lengths_tuple[j]
        elif name in merge:
            j = pos[merge[name]]
            if name in values_by:
                values = values_by[name]
            else:
                idx = _merged_index(
                    seg(i), seg(j), lengths_tuple[i], lengths_tuple[j],
                    n, length_per_key[j],
                )
                values = _CastExpandFp32Bwd.apply(embeddings, idx, out_dtype)
                setattr(values, PREMERGED_ATTR, True)
            jt_lengths = lengths_tuple[i] + lengths_tuple[j]
        elif name in values_by:
            values = values_by[name]
            jt_lengths = lengths_tuple[i]
        elif name in kept_set:
            # Unchanged: small segments stay in the EC's own dtype.
            values = torch.gather(embeddings, 0, seg(i).unsqueeze(1).expand(-1, w))
            jt_lengths = lengths_tuple[i]
        else:
            values = _CastExpandFp32Bwd.apply(embeddings, seg(i), out_dtype)
            jt_lengths = lengths_tuple[i]
        ret[name] = JaggedTensor(lengths=jt_lengths, values=values, weights=None)

    global _MERGE_LOGGED
    if not _MERGE_LOGGED:
        _MERGE_LOGGED = True
        logger.info(
            "DLRMV4 armed OK: ec_merge_uih_candidate ACTIVE -- expand writes "
            "merged uih+candidate layout for %s; per-key expands (no split "
            "backward cat); main_forward skips concat_2D_jagged for these",
            sorted(merge.items()),
        )
    return ret


def hoisted_from_dtype() -> Optional[torch.dtype]:
    """The dtype the EC output had before the hoist, or None if inactive."""
    return _HOISTED_FROM


@gin.configurable
def configure_ec_bf16_output(
    bwd_chunk_rows: int = 0,
    min_rows: int = 100_000,
) -> None:
    """Tuning for the bf16 hoist itself. Called by `enable_bf16_ec_output()`.

    Both were `os.environ` reads at import once; they are gin now so
    a run's operative config records what it was tuned with.

    Args:
      bwd_chunk_rows: rows per backward chunk.  The backward needs an fp32 copy
        of the incoming gradient, and chunking bounds that temporary at the cost
        of several passes.  0 disables chunking, which is the shipped setting --
        the fp32 copy fits.  MUTUALLY EXCLUSIVE with `ec_segment_reduce`, which
        reads the gradient in place and never materialises the copy chunking
        exists to bound; arming both raises rather than silently picking one.
      min_rows: a key-segment shorter than this keeps its original dtype.  The
        hoist pays off in proportion to the expansion it skips, and the small
        contextual/candidate features (~1024 rows) expand too little to matter
        while `_item_embedding_mlp` runs outside autocast and would raise on a
        bf16 input.  100k keeps those in fp32 and hoists only the big tables.
    """
    global _BWD_CHUNK_ROWS, _MIN_ROWS
    _BWD_CHUNK_ROWS, _MIN_ROWS = bwd_chunk_rows, min_rows


@gin.configurable
def enable_segment_reduce(
    seg_hist: bool = False,
    split_above: int = 8192,
    block_s: int = 16,
    num_warps: int = 2,
    block_w_cap: int = 512,
    gpu_planner: bool = False,
    fixed_size_offsets: bool = False,
) -> None:
    """Arm the sort-then-segment-reduce backward for the hoisted EC expand.

    This is the third rung of the measured ladder (`+4.37%` e2e on top of
    `ec_bf16_output`).  It replaces `scatter_add_` -- which serialises many
    token rows into far fewer unique rows through global atomics -- with a sort that
    makes each output row a contiguous segment one program folds with `tl.sum`,
    so there is no atomic on the cold path.

    Arm it with `make_model.ec_segment_reduce = True` (the gin binds
    `$EC_SEGMENT_REDUCE`), never by calling this directly: `make_model` is where
    the `ec_bf16_output` precondition is enforced, and this function does not
    re-check it.

    NOT bit-identical to `scatter_add_`, and cannot be: atomics accumulate in
    nondeterministic order, so there are no bits to match.  Both were scored
    against an fp64 `index_add_` reference instead and the segment kernel's max
    error is 0.61x the incumbent's -- better conditioned than what it replaces.
    It carries a 3-seed convergence gate (delta 0) for that reason; see
    `docs/04-technical-guides/recommendation/dlrm_v4/perf-notes.md`.

    Args:
      seg_hist: log the `reverse_indices` segment-length distribution.
        Diagnostic, default off; no default is derived from its output.  Costs
        a `bincount` plus a `quantile` on each new maximum-size call, so leave
        it off for timing.

      gpu_planner: keep hot-piece planning on the GPU. Preserves the host
        planner's piece boundaries and fp32 reduction arithmetic; hot-piece
        atomic accumulation remains nondeterministic. Default off.
      fixed_size_offsets: construct offsets on the GPU using the unchanged
        argsort order. Requires the EC caller's valid IDs in [0, num_rows).
        Independent of gpu_planner; default off.

      split_above, block_s, num_warps, block_w_cap: the kernel's launch
        geometry, forwarded verbatim to `ec_segment_reduce`.  They live on THIS
        function rather than on that module's own `configure_segment_reduce`
        because `ec_segment_reduce` imports triton and so is imported lazily,
        which puts it past the point where gin parses -- a knob bound there
        would raise "No configurable matching".  The defaults (split_above
        8192, geometry (16, 2, 512)) are the robust pick of the synthetic
        config-derived re-derivation (s15p1b run 70728); no real training-data
        lengths were used.  They must equal `ec_segment_reduce`'s module
        defaults (checked by tests/ec_segment_reduce_config_test.py).  They are
        dormant in steady state (the default-on fused HIP dgrad path does not
        read them) and affect only warmup backward calls, fallbacks and
        self-tests.
    """
    if type(gpu_planner) is not bool or type(fixed_size_offsets) is not bool:
        raise ValueError("gpu_planner and fixed_size_offsets must be booleans")
    global _SEGMENT_REDUCE, _SEG_HIST
    _SEGMENT_REDUCE, _SEG_HIST = True, seg_hist
    # Importing here rather than at module scope keeps Triton off the import
    # path of every run that does not arm this -- which is also why the geometry
    # knobs are arguments of THIS function and get forwarded, rather than being
    # gin-bound over there where gin could never see them.
    from generative_recommenders.dlrm_v4.train.ec_segment_reduce import (
        configure_segment_reduce,
    )

    configure_segment_reduce(
        split_above=split_above,
        block_s=block_s,
        num_warps=num_warps,
        block_w_cap=block_w_cap,
        gpu_planner=gpu_planner,
        fixed_size_offsets=fixed_size_offsets,
    )


def _log_seg_histogram(reverse_indices: torch.Tensor, num_rows: int) -> None:
    """Dump the real segment-length distribution once, then never again."""
    cnt = torch.bincount(reverse_indices, minlength=num_rows).to(torch.float64)
    q = torch.tensor([0.5, 0.9, 0.99, 0.999], device=cnt.device, dtype=cnt.dtype)
    p = torch.quantile(cnt, q).tolist()
    logger.info(
        "DLRMV4 ec_segment_reduce histogram: T=%d U=%d dup=%.2fx  len "
        "min=%d p50=%.0f p90=%.0f p99=%.0f p999=%.0f max=%d  "
        "top-1%%-of-rows carries %.1f%% of tokens",
        reverse_indices.numel(),
        num_rows,
        reverse_indices.numel() / max(num_rows, 1),
        int(cnt.min()),
        p[0],
        p[1],
        p[2],
        p[3],
        int(cnt.max()),
        100.0
        * torch.topk(cnt, max(num_rows // 100, 1)).values.sum().item()
        / max(cnt.sum().item(), 1.0),
    )


def _segment_reduce_backward(
    ctx, reverse_indices: torch.Tensor, grad_out: torch.Tensor
) -> tuple:
    """``scatter_add_`` replaced by sort-then-segment-reduce.

    Deliberately does NOT fall back.  A knob that silently does nothing and a
    knob that does nothing useful are indistinguishable in a step time, and
    this one is measured as an A/B arm -- an inert arm would be read as a null
    result rather than as a broken one.
    """
    global _SEG_ARMED_LOGGED, _SEG_HIST_MAX_T
    if _BWD_CHUNK_ROWS > 0:
        raise RuntimeError(
            "EC_SEGMENT_REDUCE=1 is incompatible with "
            f"DLRMV4_EC_BF16_BWD_CHUNK={_BWD_CHUNK_ROWS}: the segment kernel "
            "reads the incoming gradient in place and never materialises the "
            "fp32 copy that chunking exists to bound. Unset one of them."
        )
    from generative_recommenders.dlrm_v4.train.ec_segment_reduce import (
        segment_reduce_rows,
    )

    # Arming stays one-shot -- it answers "is this path live", which the first
    # call settles.  The histogram does not: it answers "what shape does this
    # path actually see", and only the largest call seen so far can answer that.
    if _SEG_HIST and reverse_indices.numel() > _SEG_HIST_MAX_T:
        _SEG_HIST_MAX_T = reverse_indices.numel()
        _log_seg_histogram(reverse_indices, ctx.num_rows)
    if not _SEG_ARMED_LOGGED:
        _SEG_ARMED_LOGGED = True
        logger.info(
            "DLRMV4 armed OK: ec_segment_reduce backward active "
            "(T=%d -> U=%d rows, W=%d, grad dtype=%s, accum fp32 -> %s)",
            reverse_indices.numel(),
            ctx.num_rows,
            ctx.width,
            grad_out.dtype,
            ctx.in_dtype,
        )
    # fp32 accumulator out of the kernel; .to() is a no-op when in_dtype is
    # fp32, which is the whole point of this module's backward.
    grad_emb = segment_reduce_rows(grad_out, reverse_indices, ctx.num_rows)
    return grad_emb.to(ctx.in_dtype), None, None


# EC expand gather v2 (DLRM_HSTU_FP8_EC_GATHER_V2: default OFF in code, the gin
# binding apply_env_bootstrap.DLRM_HSTU_FP8_EC_GATHER_V2 turns it on, "0" is
# the kill switch; read once at import): index tensors of at least ec_expand_gather.min_rows()
# rows go through the multi-row Triton row copy in ec_expand_gather (same
# bytes as torch.gather, raw-bit checked at production shapes; one launch
# instead of 2-4 int32-split vectorized_gather_kernel launches); everything
# else stays on torch.gather. Imported lazily on first use.
_EC_GATHER_V2: bool = os.environ.get(
    "DLRM_HSTU_FP8_EC_GATHER_V2", ""
).strip().lower() in ("1", "true", "on", "yes")


def _expand(src: torch.Tensor, idx: torch.Tensor, w: int) -> torch.Tensor:
    """``torch.gather(src, 0, idx[:, None].expand(-1, w))`` -- a pure row copy.

    Fail-closed: if the Triton gather raises (compile/launch), it is disabled
    for the rest of the process with one warning and torch.gather (the same
    bytes) takes over.
    """
    global _EC_GATHER_V2
    if _EC_GATHER_V2:
        from generative_recommenders.dlrm_v4.train import ec_expand_gather as _eg

        if idx.numel() >= _eg.min_rows() and _eg.supported(src, idx):
            try:
                return _eg.expand_rows(src, idx)
            except Exception as exc:  # fail closed to torch.gather
                _EC_GATHER_V2 = False
                logger.warning(
                    "DLRMV4 ec gather v2 DISABLED for this process (%s: %s); "
                    "torch.gather expands every EC key from here on",
                    type(exc).__name__,
                    str(exc)[:300],
                )
    return torch.gather(src, 0, idx.unsqueeze(1).expand(-1, w))


class _CastExpandFp32Bwd(torch.autograd.Function):
    """bf16 expand forward; fp32 ``scatter_add_`` backward.

    forward:  out[t] = embeddings[reverse_indices[t]].to(out_dtype)
    backward: grad_emb = zeros(U, W, in_dtype).scatter_add_(
                  0, idx, grad_out.to(in_dtype))
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(  # type: ignore[override]
        ctx,
        embeddings: torch.Tensor,
        reverse_indices: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.save_for_backward(reverse_indices)
        ctx.in_dtype = embeddings.dtype
        ctx.num_rows = embeddings.size(0)
        ctx.width = embeddings.size(-1)
        # Cast FIRST -- 69k rows instead of 7.1M -- then expand.
        return _expand(embeddings.to(out_dtype), reverse_indices, embeddings.size(-1))

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (reverse_indices,) = ctx.saved_tensors
        w = ctx.width
        if _SEGMENT_REDUCE:
            return _segment_reduce_backward(ctx, reverse_indices, grad_out)
        grad_emb = torch.zeros(
            (ctx.num_rows, w), dtype=ctx.in_dtype, device=grad_out.device
        )
        n = grad_out.size(0)
        chunk = _BWD_CHUNK_ROWS if _BWD_CHUNK_ROWS > 0 else n
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            idx = reverse_indices[lo:hi].unsqueeze(1).expand(-1, w)
            # .to(ctx.in_dtype) is the whole point: the accumulation, and the
            # accumulator, stay fp32.
            grad_emb.scatter_add_(0, idx, grad_out[lo:hi].to(ctx.in_dtype))
        return grad_emb, None, None


_EC_GROUP_ENV = "DLRM_HSTU_FP8_EC_GROUP_BWD"
_EC_GROUP_LOGGED = False


def _ec_group_enabled() -> bool:
    """Grouped EC expand backward (one fp32 accumulator for every key of a call).

    Off when ``DLRM_HSTU_FP8_EC_GROUP_BWD`` is absent (repo convention: the
    gin binding ``apply_env_bootstrap.DLRM_HSTU_FP8_EC_GROUP_BWD`` turns it on);
    ``=0`` restores one autograd node per key (the DLRM_HSTU_FP8_ prefix is
    forwarded into the container by launch_slurm.sh). Read per call. Only
    engages with the segment-reduce backward armed and no backward chunking,
    the shipped configuration.
    """
    v = os.environ.get(_EC_GROUP_ENV, "0").strip().lower()
    return (
        v in ("1", "true", "on", "yes")
        and _SEGMENT_REDUCE
        and _BWD_CHUNK_ROWS == 0
    )


_EX2B_ENV = "DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE"
_EX2B_ON: Optional[bool] = None


def _ex2b_plan(embeddings, out_dtype, casts, idxs, cat_slot: int, w: int):
    """E-X2b (actckpt; preprocessor L1 dgrads fused into this backward): the
    dh-port plan for one grouped call, or None -> the shipped _ECExpandGroup
    call. Off (absent) by default; read once, at the first grouped call. The
    module is imported only with the knob on."""
    global _EX2B_ON
    if _EX2B_ON is None:
        _EX2B_ON = os.environ.get(_EX2B_ENV, "").strip().lower() in (
            "1", "true", "on", "yes",
        )
    if not _EX2B_ON:
        return None
    from generative_recommenders.dlrm_v4.train import ec_dgrad_ports as _xp

    return _xp.plan(embeddings, out_dtype, casts, idxs, cat_slot, w)


# Indexed-X L1 (rccl R-C3 / actckpt E-X2; ops/ec_indexed_l1.py, ops/triton/triton_ec_indexed_l1.py;
# DLRM_HSTU_FP8_EC_INDEXED_L1, default off): the two hoisted members go to the preprocessor L1 as (U, idx)
# placeholders and the L1 forward / wgrad read X through the index. Read once, at the first grouped call; the
# protocol module is imported only with the knob on. Fail-closed.
_IX_ENV = "DLRM_HSTU_FP8_EC_INDEXED_L1"
_IX_ON: Optional[bool] = None
# make_model's consumer-chain registration: None = not registered, "" = Indexed-X-aware, else what is missing.
_IX_CHAIN: Optional[str] = None
# The uih_post_id feature (make_model): the merged item member the content MLP's pair 0 reads.
_IX_UIH_POST_ID: Optional[str] = None


def note_ec_indexed_l1_chain(fuse_linear_norm: bool, fuse_ln_residual: bool, bf16_training: bool,
                             transducer_bf16_dtype: bool, uih_post_id: str) -> None:
    """make_model, after the merge and CAT_FUSE registrations: whether every consumer of the hoisted members is
    Indexed-X-aware (FusedLinearNormMLP pairs under the grouped tail and E-M3a, autocast(bf16), the transducer's
    .to() an identity, the uih_post_id pair merged in the expand so main_forward skips its concat, the 2-key
    CAT_FUSE group so the additional MLP reads `big`). Registration only: nothing changes unless
    DLRM_HSTU_FP8_EC_INDEXED_L1 is on."""
    global _IX_CHAIN, _IX_UIH_POST_ID
    tail = os.environ.get("DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL", "0") == "1"  # == preprocessors.py:49-51
    merged = any(u == uih_post_id for (u, _c) in _MERGE_PAIRS)  # enable_ec_merge_uih_candidate registered it
    missing = [name for name, ok in (
        ("fuse_linear_norm", fuse_linear_norm), ("fuse_ln_residual", fuse_ln_residual),
        ("bf16_training", bf16_training), ("transducer_bf16_dtype", transducer_bf16_dtype),
        ("DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL", tail),
        (f"ec_merge_uih_candidate ({uih_post_id})", merged),
        ("DLRM_HSTU_FP8_EC_CAT_FUSE (2-key group)", len(_CAT_GROUP) == 2)) if not ok]
    _IX_CHAIN = ", ".join(missing) + " off" if missing else ""
    _IX_UIH_POST_ID = str(uih_post_id)


def _ix_plan(embeddings, out_dtype, casts, idxs, cat_slot: int, w: int, names, premerged):
    """Indexed-X records for one grouped call (ops/ec_indexed_l1.plan), or None -> the shipped expand.
    names / premerged: _merged_construct's lists (no entry for the cat member)."""
    global _IX_ON
    if _IX_ON is None:
        _IX_ON = os.environ.get(_IX_ENV, "").strip().lower() in ("1", "true", "on", "yes")
    if not _IX_ON:
        return None
    try:
        from generative_recommenders.ops import ec_indexed_l1 as _ixp
    except Exception as exc:  # fail closed: the shipped expand for the rest of the run
        _IX_ON = False
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(f"[ec-indexed-l1] FAILED CLOSED (DISABLED for the rest of the run) -> shipped EC "
                            f"expand + L1 GEMMs (import: {type(exc).__name__}: {str(exc)[:300]})")
        return None
    try:
        return _ixp.plan(embeddings, out_dtype, casts, idxs, cat_slot, w, cat_group=len(_CAT_GROUP),
                         chain=_IX_CHAIN, names=names, premerged=premerged, uih_post_id=_IX_UIH_POST_ID,
                         expand=_expand)
    except Exception as exc:  # fail closed
        _ixp.K.latch_off("ec plan", exc)
        return None


# Item (118) PREPROC_DEDUP (ops/preproc_dedup.py; DLRM_HSTU_FP8_PREPROC_DEDUP, default off, only "1" arms it): the
# grouped call's merged item member output and `big` carry one record, and the preprocessor runs the content /
# additional MLPs once per unique (item, artist, album) row triple. Read once, at the first grouped call; the module
# is imported only with the knob on. Fail-closed.
_PPDD_ENV = "DLRM_HSTU_FP8_PREPROC_DEDUP"
_PPDD_ON: Optional[bool] = None
# make_model's registration (knob on only): None = not registered, "" = complete, else what is missing.
_PPDD_CHAIN: Optional[str] = None
_PPDD_UIH_POST_ID: Optional[str] = None


def note_preproc_dedup_chain(fuse_linear_norm: bool, fuse_ln_residual: bool, bf16_training: bool,
                             ec_segment_reduce: bool, transducer_bf16_dtype: bool, uih_post_id: str) -> None:
    """make_model (knob on), after the merge and CAT_FUSE registrations: whether the chain the dedup path runs
    is configured (FusedLinearNormMLP F1 + F2, autocast(bf16), the segment-reduce EC backward, the transducer's
    .to() an identity so the tagged member reaches the preprocessor, the uih_post_id pair merged in the expand, the
    2-key CAT_FUSE group). Registration only: nothing changes unless the knob is "1"."""
    global _PPDD_CHAIN, _PPDD_UIH_POST_ID
    merged = any(u == uih_post_id for (u, _c) in _MERGE_PAIRS)  # enable_ec_merge_uih_candidate registered it
    missing = [name for name, ok in (
        ("fuse_linear_norm", fuse_linear_norm), ("fuse_ln_residual", fuse_ln_residual),
        ("bf16_training", bf16_training), ("ec_segment_reduce", ec_segment_reduce),
        ("transducer_bf16_dtype", transducer_bf16_dtype), (f"ec_merge_uih_candidate ({uih_post_id})", merged),
        ("DLRM_HSTU_FP8_EC_CAT_FUSE (2-key group)", len(_CAT_GROUP) == 2)) if not ok]
    _PPDD_CHAIN = ", ".join(missing) + " off" if missing else ""
    _PPDD_UIH_POST_ID = str(uih_post_id)


def ppdd_enabled() -> bool:
    """DLRM_HSTU_FP8_PREPROC_DEDUP == "1" (ops/preproc_dedup.enabled's rule), read once, without importing it."""
    global _PPDD_ON
    if _PPDD_ON is None:
        _PPDD_ON = os.environ.get(_PPDD_ENV, "").strip() == "1"
    return _PPDD_ON


def _ppdd_tag(embeddings, out_dtype, casts, idxs, cat_slot: int, w: int, names, premerged, values_by, fused_vals,
              ix) -> None:
    """Item (118): tag the item member output and `big` of the call (ops/preproc_dedup.tag). Knob off: returns."""
    global _PPDD_ON
    if not ppdd_enabled():
        return
    try:
        from generative_recommenders.ops import preproc_dedup as _pd
    except Exception as exc:  # fail closed: today's statements for the rest of the run
        _PPDD_ON = False
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(f"[preproc-dedup] DISABLED (FAILED CLOSED) at import: {type(exc).__name__}: "
                            f"{str(exc)[:300]} -> today's statements for the rest of the run")
        return
    _pd.tag(embeddings, out_dtype, casts, idxs, cat_slot, w, names=names, premerged=premerged,
            values_by=values_by, fused_vals=fused_vals, cat_group=list(_CAT_GROUP), ix=ix, chain=_PPDD_CHAIN,
            uih_post_id=_PPDD_UIH_POST_ID, expand=_expand, cast_expand=_CastExpandFp32Bwd.apply)


def _ec_joint_segments(hoisted, U: int):
    """DLRM_HSTU_FP8_EC_JOINT_SORT (default OFF in code; the gin binding turns
    it on): both hoisted members' incumbent (order, offsets) from ONE stable
    argsort (ec_offsets.build_segments_joint_valid_ids), or None -> each member
    sorts on its own as before. Only on the shipped fixed-size-offsets path
    (build_segments == build_segments_valid_ids), two hoisted members. The
    reducer is not involved (segment_reduce_rows takes the supplied order and
    offsets on the LPT and the two-launch path alike). Fail-closed: a raising
    call disables it."""
    if len(hoisted) != 2:
        return None
    try:
        from generative_recommenders.dlrm_v4.train import ec_offsets as eof
        from generative_recommenders.dlrm_v4.train import ec_segment_reduce as sr

        if not eof._JOINT_SORT or not sr._FIXED_SIZE_OFFSETS:
            return None
        ids = [idx.reshape(-1) for idx, _ in hoisted]
        if not eof.joint_sort_engages(ids, U):
            return None
        return eof.build_segments_joint_valid_ids(ids, U)
    except Exception as exc:  # fail closed: per-member sorts from now on
        try:
            from generative_recommenders.dlrm_v4.train import ec_offsets as eof

            eof._JOINT_SORT = False
        except Exception:
            pass
        logger.warning(
            "DLRMV4 ec joint sort FAILED CLOSED -> per-member argsort for the "
            "rest of the run (%s: %s)", type(exc).__name__, str(exc)[:300],
        )
        return None


def _ec_reduce_fused(hoisted, segs, U: int):
    """DLRM_HSTU_FP8_EC_REDUCE_FUSED (default OFF in code; the gin binding
    apply_env_bootstrap.DLRM_HSTU_FP8_EC_REDUCE_FUSED turns it on, "0" is the
    kill switch): the hoisted members' segment reduces as ONE grouped launch
    (ec_gpu_plan.segment_reduce_grouped: one plan chain, one main launch, no
    [U, W] zero fill and no member-1 read-modify-write pass), with the values
    of the per-member EC_REDUCE_LPT calls. Returns (acc or None, segs): None ->
    the caller runs the per-member calls on the returned segments (built here
    when the joint sort did not supply them). Engages with 1-2 hoisted
    members, the GPU planner on and EC_REDUCE_LPT taking the call (the fused
    path reproduces LPT; the production key is validated offline, any other
    width / dtype / stride class by a one-time raw-bit self-test). Fail-closed:
    a raising call disables it for the process with one warning."""
    if not hoisted or len(hoisted) > 2:
        return None, segs
    try:
        from generative_recommenders.dlrm_v4.train import ec_gpu_plan as gpl
        from generative_recommenders.dlrm_v4.train import ec_segment_reduce as sr

        if not gpl.fused_enabled() or not sr._GPU_PLANNER:
            return None, segs
        if segs is None:
            segs = [sr.build_segments(idx, U) for idx, _ in hoisted]
        members = [(g, segs[i][0], segs[i][1]) for i, (_, g) in enumerate(hoisted)]
        return gpl.segment_reduce_grouped(members, U), segs
    except Exception as exc:  # fail closed: per-member reduces from now on
        try:
            from generative_recommenders.dlrm_v4.train import ec_gpu_plan as gpl

            gpl._FUSED_ENABLED = False
        except Exception:
            pass
        logger.warning(
            "DLRMV4 ec reduce fused FAILED CLOSED -> per-member segment reduce "
            "for the rest of the run (%s: %s)", type(exc).__name__, str(exc)[:300],
        )
        return None, segs


class _ECExpandGroup(torch.autograd.Function):
    """All per-key expands of ONE EC call as one autograd node.

    forward:  out_k = gather(embeddings.to(out_dtype) if cast_k else embeddings,
                             0, idx_k)          -- bitwise the per-key forward,
              with the [U, W] cast done once instead of once per hoisted key.
    backward: ONE fp32 [U, W] accumulator.  Hoisted keys are segment-reduced
              into it (the first allocates, later ones use the kernel's ACCUM
              mode: out[row] += segment sum); kept keys ``scatter_add_`` into
              it.  The per-key path instead allocated a zeroed [U, W] grad for
              every kept key (GatherBackward new_zeros) and autograd summed the
              K full-size grads with K-1 [U, W] fp32 ``add_`` kernels -- at 4n
              (U=155k, 15 keys) ~2.7 ms/rank-step of fill/add on the main stream.
    Numerics: same fp32 accumulator dtype; only the fp32 addition ORDER of the
    per-key contributions changes (hot pieces were already atomic, i.e. the EC
    grad was already not bit-reproducible).
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(  # type: ignore[override]
        ctx,
        embeddings: torch.Tensor,
        out_dtype: torch.dtype,
        casts: Tuple[bool, ...],
        *idxs: torch.Tensor,
    ):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*idxs)
        ctx.casts = casts
        ctx.in_dtype = embeddings.dtype
        ctx.num_rows = embeddings.size(0)
        ctx.width = embeddings.size(-1)
        w = embeddings.size(-1)
        emb_c = embeddings.to(out_dtype) if any(casts) else None
        outs = []
        # Indexed-X (ops/ec_indexed_l1.py): a casts entry may be an IXRecord (truthy; never with the knob
        # off). It issues the member's stride-0 placeholder (or, latched off, the shipped expand) and keeps
        # emb_c / idx for the preprocessor L1. ctx then holds only bools and the consumer ledgers.
        ix_ledgers = []
        has_ix = False
        for c, idx in zip(casts, idxs):
            if getattr(c, "ix_record", False):
                has_ix = True
                outs.append(c.issue(emb_c, idx, w, out_dtype))
                if c.issued:
                    ix_ledgers.append((len(outs) - 1, c.ledger))
                continue
            src = emb_c if c else embeddings
            outs.append(_expand(src, idx, w))
        if has_ix:
            ctx.casts = tuple(bool(c) for c in casts)  # the node keeps no record (and so no emb_c)
            if ix_ledgers:
                ctx.ix_ledgers = tuple(ix_ledgers)  # consumer accounting only, no tensors
        return tuple(outs)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, *grads):  # type: ignore[override]
        global _EC_GROUP_LOGGED
        from generative_recommenders.dlrm_v4.train.ec_segment_reduce import (
            segment_reduce_rows,
        )

        idxs = ctx.saved_tensors
        U, w = ctx.num_rows, ctx.width
        # Indexed-X (ops/ec_indexed_l1.py): each issued member's gradient must come from an Indexed-X-aware
        # consumer in this backward (raises on a NaN reader or a lost gradient). Absent when off.
        ixl = getattr(ctx, "ix_ledgers", None)
        if ixl is not None:
            from generative_recommenders.ops import ec_indexed_l1 as _ixp

            _ixp.ec_backward_check(ixl, grads)
        # E-X2b (ec_dgrad_ports.ECExpandGroupPorts) seeds the accumulator with
        # the hoisted members' fused sums and passes their grads as None; the
        # shipped class never sets ex2b_acc.
        acc: Optional[torch.Tensor] = getattr(ctx, "ex2b_acc", None)
        n_seg = n_sc = 0
        # Hoisted (large) keys first: the first segment reduce allocates the
        # accumulator, later ones accumulate in-kernel.
        global _SEG_ARMED_LOGGED, _SEG_HIST_MAX_T
        hoisted = [
            (idx, g) for c, idx, g in zip(ctx.casts, idxs, grads)
            if g is not None and c
        ]
        # DLRM_HSTU_FP8_EC_JOINT_SORT: both members' incumbent segments from
        # one stable argsort (None -> each member sorts on its own, below).
        segs = _ec_joint_segments(hoisted, U)
        for i, (idx, g) in enumerate(hoisted):
            # Keep the documented segment-reduce arming line and the seg_hist
            # diagnostic working on the grouped path.
            if _SEG_HIST and idx.numel() > _SEG_HIST_MAX_T:
                _SEG_HIST_MAX_T = idx.numel()
                _log_seg_histogram(idx, U)
            if not _SEG_ARMED_LOGGED:
                _SEG_ARMED_LOGGED = True
                logger.info(
                    "DLRMV4 armed OK: ec_segment_reduce backward active "
                    "(T=%d -> U=%d rows, W=%d, grad dtype=%s, accum fp32 -> %s)",
                    idx.numel(), U, w, g.dtype, ctx.in_dtype,
                )
        # DLRM_HSTU_FP8_EC_REDUCE_FUSED: all hoisted members in one grouped
        # launch (None -> the per-member calls below, on the same segments).
        # Never over an E-X2b seed: then `hoisted` is empty, and the per-member
        # loop below would accumulate into the seed anyway.
        acc_f = None
        if acc is None:
            acc_f, segs = _ec_reduce_fused(hoisted, segs, U)
        if acc_f is not None:
            acc = acc_f
            n_seg = len(hoisted)
        else:
            for i, (idx, g) in enumerate(hoisted):
                order, offsets = segs[i] if segs is not None else (None, None)
                acc = segment_reduce_rows(
                    g, idx, U, order=order, offsets=offsets, accumulate_into=acc
                )
                n_seg += 1
        for c, idx, g in zip(ctx.casts, idxs, grads):
            if g is None or c:
                continue
            if acc is None:
                acc = torch.zeros((U, w), dtype=torch.float32, device=g.device)
            acc.scatter_add_(
                0, idx.unsqueeze(1).expand(-1, w), g.to(torch.float32)
            )
            n_sc += 1
        if not _EC_GROUP_LOGGED:
            _EC_GROUP_LOGGED = True
            logger.warning(
                "DLRMV4 armed OK: ec_group_bwd ACTIVE -- one fp32 [%d, %d] "
                "accumulator for %d segment-reduced + %d scatter keys "
                "(kill switch %s=0)", U, w, n_seg, n_sc, _EC_GROUP_ENV,
            )
        grad_emb = None if acc is None else acc.to(ctx.in_dtype)
        return (grad_emb, None, None) + (None,) * len(idxs)


def _log_once(
    embeddings: torch.Tensor, out_dtype: torch.dtype, hoisted: List[str],
    kept: List[str],
) -> None:
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    logger.info(
        "armed OK: ec_bf16_output ACTIVE -- construct_jagged_tensors saw "
        "embeddings dtype=%s shape=%s at the call site; expanding in %s. "
        "hoisted keys=%s ; kept at %s (below %d rows)=%s ; "
        "backward accumulates in %s (bwd_chunk_rows=%d)",
        embeddings.dtype,
        tuple(embeddings.shape),
        out_dtype,
        hoisted,
        embeddings.dtype,
        _MIN_ROWS,
        kept,
        embeddings.dtype,
        _BWD_CHUNK_ROWS,
    )


def enable_bf16_ec_output(out_dtype: torch.dtype = torch.bfloat16) -> bool:
    """Patch torchrec so the dedup expand runs in ``out_dtype``. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    # Pick up `configure_ec_bf16_output`'s gin bindings (or its defaults) before
    # the patched path can run. Calling it is what makes the bindings take
    # effect -- gin applies them at call time, not at import.
    configure_ec_bf16_output()

    import torchrec.distributed.embedding as tde
    import torchrec.modules.utils as tmu
    from torchrec.sparse.jagged_tensor import JaggedTensor

    original = tmu.construct_jagged_tensors

    def _wrapped(
        embeddings: torch.Tensor,
        features,
        embedding_names,
        need_indices: bool = False,
        features_to_permute_indices=None,
        original_features=None,
        reverse_indices: Optional[torch.Tensor] = None,
        seq_vbe_ctx=None,
        use_gather_select: bool = False,
        **newer_flags,  # torchrec >= 1.9 adds use_sorted_select (off by default): passed through
    ):
        # Engage only on the path this was measured on: dedup on (there is an
        # expand to hoist above), gather-select on (so the backward being
        # preserved really is scatter_add_), no VBE reindex (fbgemm's permute
        # runs after the expand and is out of scope), and an fp32 input.
        if (
            reverse_indices is None
            or not use_gather_select
            or seq_vbe_ctx is not None
            or embeddings.dtype == out_dtype
            or not embeddings.is_floating_point()
            or any(newer_flags.values())  # e.g. use_sorted_select: torchrec's own backward
        ):
            return original(
                embeddings=embeddings,
                features=features,
                embedding_names=embedding_names,
                need_indices=need_indices,
                features_to_permute_indices=features_to_permute_indices,
                original_features=original_features,
                reverse_indices=reverse_indices,
                seq_vbe_ctx=seq_vbe_ctx,
                use_gather_select=use_gather_select,
                **newer_flags,
            )

        feats = original_features if original_features is not None else features
        length_per_key: List[int] = feats.length_per_key()

        hoisted, kept = [], []
        offs, acc = [], 0
        for i, name in enumerate(embedding_names):
            offs.append(acc)
            acc += length_per_key[i]
            (hoisted if length_per_key[i] >= _MIN_ROWS else kept).append(name)

        global _HOISTED_FROM
        _HOISTED_FROM = embeddings.dtype
        _log_once(embeddings, out_dtype, hoisted, kept)

        if _MERGE_PAIRS and not need_indices:
            merged = _merged_construct(
                embeddings, feats, embedding_names, reverse_indices, out_dtype,
                length_per_key, offs, kept,
            )
            if merged is not None:
                return merged

        expanded = _CastExpandFp32Bwd.apply(embeddings, reverse_indices, out_dtype)
        ret: Dict[str, JaggedTensor] = original(
            embeddings=expanded,
            features=features,
            embedding_names=embedding_names,
            need_indices=need_indices,
            features_to_permute_indices=features_to_permute_indices,
            original_features=original_features,
            reverse_indices=None,
            seq_vbe_ctx=None,
            use_gather_select=use_gather_select,
            **newer_flags,
        )
        if not kept:
            return ret

        # Re-expand the small segments from the ORIGINAL fp32 tensor so their
        # consumers see exactly the bits they see today. These are 1024-row
        # segments; the cost is noise. Only the 1:1 (non-CW) case is handled --
        # a key appearing at several indices is concatenated inside the
        # original and is left on the bf16 path.
        seen: Dict[str, List[int]] = {}
        for i, name in enumerate(embedding_names):
            seen.setdefault(name, []).append(i)
        w = embeddings.size(-1)
        for name in kept:
            idxs = seen[name]
            if len(idxs) != 1:
                continue
            i = idxs[0]
            lo, n = offs[i], length_per_key[i]
            if n == 0:
                ret[name] = JaggedTensor(
                    lengths=ret[name].lengths(),
                    values=ret[name].values().to(embeddings.dtype),
                    weights=ret[name].weights_or_none(),
                )
                continue
            sub = reverse_indices[lo:lo + n].unsqueeze(1).expand(-1, w)
            ret[name] = JaggedTensor(
                lengths=ret[name].lengths(),
                values=torch.gather(embeddings, 0, sub),
                weights=ret[name].weights_or_none(),
            )
        return ret

    tmu.construct_jagged_tensors = _wrapped
    if getattr(tde, "construct_jagged_tensors", None) is original:
        tde.construct_jagged_tensors = _wrapped
    _PATCHED = True
    logger.info(
        "DLRMV4 ec_bf16_output: patched construct_jagged_tensors in "
        "torchrec.modules.utils and torchrec.distributed.embedding "
        "(out_dtype=%s, min_rows=%d). The 'armed OK' line prints on the first "
        "call that actually hoists.",
        out_dtype,
        _MIN_ROWS,
    )
    return True
