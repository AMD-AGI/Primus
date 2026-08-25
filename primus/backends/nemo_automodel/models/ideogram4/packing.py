###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Multi-sample row layout for Ideogram-4: K training samples in ONE batch row.

WHY:
  The var-len path shipped before this one puts exactly ONE sample in a row, as
  ``[left-pad][text][image]`` -- two segments, so its packing factor is 1.0 and every row
  still pays for padding the caption out to the corpus maximum. Attention already skips
  that padding (the pad block is its own segment), but the projections and the FFN do not:
  a pad token is a full-width token for all 34 blocks. Packing K samples into one row
  removes those slots entirely.

  This module owns the layout and NOTHING else. It is pure host-side integer arithmetic --
  no tensor VALUES are ever read -- which is what keeps the packing a graph INPUT rather
  than something recovered inside the compiled region (see ``attention.py``'s COMPILE
  section for why a device->host read there is a liveness bug under FSDP2, not a slowdown).

THE ROW:
    row = [ slack ][ t_1 ][ img_1 ][ t_2 ][ img_2 ] ... [ t_K ][ img_K ]
            >=1      ragged  fixed    ragged  fixed

  * ``S = text_budget + K * num_image_tokens`` -- a DATASET-level constant, exactly as
    ``max_text_tokens`` is for the one-sample layout. Nothing here depends on the captions
    in the current batch, so ``torch.compile`` sees one shape for the whole run.
  * Segments per row is ``K + 1`` (one slack + one per sample), so ``cu_seqlens`` always
    has ``B*(K+1)+1`` entries. This is the direct generalization of the reserved pad
    column: the packer guarantees ``sum(t_i) <= text_budget - 1`` so the slack segment is
    never zero-length, which var-len flash cannot represent.
  * ``segment_ids`` is ``-1`` over slack and ``1..K`` over the samples. The model builds
    ``(seg_i == seg_j)``, so distinct ids are all that isolates the samples from each
    other; slack shares one id and forms its own block at the row head, which keeps every
    query row non-empty (an all-False mask row would produce NaN).
  * Each sample's text positions restart at 0 and each image block restarts at
    ``IMAGE_POSITION_OFFSET``. Reusing them across samples in a row is safe precisely
    because the segments cannot attend to each other, and it keeps position magnitudes
    where they are today rather than growing with K (they are already at the edge of what
    bf16 represents exactly, which is why MRoPE runs with autocast disabled upstream).

ROW ASSIGNMENT IS NOT DECIDED HERE:
  ``build_packed_layout`` reads ``text_lengths`` in order and groups them ``K`` at a time,
  so row ``j`` holds samples ``j*K .. j*K+K-1``. Choosing WHICH samples share a row is the
  batch sampler's job (``data/packed_sampler.py``) -- it is the only component that can see
  enough of the dataset to keep every row inside the budget. Splitting it this way means
  this module has no feasibility policy to get wrong: it validates the budget and raises.

K == 1 IS THE OLD LAYOUT:
  With ``pack_size=1`` and ``text_budget = text_capacity + 1`` the row is
  ``[slack][text][image]`` with ``slack = text_budget - t`` -- byte-identical to what
  ``build_cu_seqlens`` + ``_prepare_ids`` produce today, including the reserved pad column
  (which is now just "slack >= 1" rather than a column concatenated onto the features).
  That identity is asserted in the unit tests and is the regression anchor for this change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Per-token role / layout constants live with the diffusers transformer definition. They are
# resolved here (rather than in the adapter) because both the adapter and this module need
# them and the adapter imports this file.
try:
    from diffusers.models.transformers.transformer_ideogram4 import (
        IMAGE_POSITION_OFFSET,
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
        SEQUENCE_PADDING_INDICATOR,
    )
except Exception:  # pragma: no cover - keep the module importable without diffusers (tests, lint)
    # Values as of diffusers 0.39.0, read out of the installed package rather than guessed.
    # They are the layout contract with the model, so a stale literal here silently mislabels
    # every token: the warning below is the only signal that the real ones were not used.
    IMAGE_POSITION_OFFSET = 65536
    LLM_TOKEN_INDICATOR = 3
    OUTPUT_IMAGE_INDICATOR = 2
    SEQUENCE_PADDING_INDICATOR = -1
    logger.warning(
        "[PrimusIdeogram4] Could not import the Ideogram-4 layout constants from diffusers; "
        "falling back to the diffusers 0.39.0 values (pad=%d, llm=%d, image=%d, offset=%d). "
        "Fine for import-only use, but if a real run reaches this the constants may have been "
        "renamed upstream and the token layout will be wrong with no further error.",
        SEQUENCE_PADDING_INDICATOR,
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
        IMAGE_POSITION_OFFSET,
    )

# Segment id of the first (slack) segment in every row. Any value distinct from 1..K works;
# the model only ever compares ids for equality. Kept at the model's own padding indicator so
# a dump of segment_ids reads the same as the one-sample layout's.
SLACK_SEGMENT_ID = SEQUENCE_PADDING_INDICATOR


@dataclass(frozen=True)
class PackedLayout:
    """Everything the adapter needs to fold ``N`` samples into ``B`` rows and back.

    The three index tensors are the whole reason this is worth precomputing: they turn the
    scatter-in and gather-out into single ``index_add_`` / ``index_select`` calls whose
    operand shapes depend only on ``(N, K, text_capacity, num_image_tokens)``. Nothing about
    them varies with the captions, so they are constant-shaped graph inputs.

    Attributes:
        num_samples: ``N``, samples in the micro-batch. Unchanged by packing -- it is what
            the flow-matching pipeline still sees on both sides of the adapter.
        num_rows: ``B = N // pack_size``, rows actually handed to the transformer.
        pack_size: ``K``, samples per row.
        seq_len: ``S = text_budget + K * num_image_tokens``, the row width.
        text_budget: total token slots a row reserves for captions, slack included.
        text_capacity: width of the incoming left-padded ``llm_features`` text axis.
        num_image_tokens: ``grid_h * grid_w``, constant across samples in this phase.
        segments_per_row: ``K + 1``. Published to the attention processor so its staleness
            check knows how many entries ``cu_seqlens`` should have for this batch.
        max_seqlen: STATIC upper bound on any one segment, never the batch's true maximum --
            Dynamo guards Python ints by value, so a data-derived bound recompiles on almost
            every batch.
        cu_seqlens: ``int32 (B*(K+1)+1,)`` cumulative segment starts over the flattened
            ``(B*S)`` sequence. Deliberately left on the HOST; ``publish_packing`` performs
            the single host->device copy.
        segment_ids: ``long (B, S)``.
        position_ids: ``long (B, S, 3)``.
        indicator: ``long (B, S)``.
        image_dst: ``long (N * num_image_tokens,)`` flat position in ``(B*S)`` of every
            sample's image tokens, in sample-major order. Used to scatter latents in and to
            gather the predicted velocity back out.
        text_dst: ``long (N * text_capacity,)`` flat destination of every slot of the
            left-padded ``llm_features``. Left-pad slots point at the DUSTBIN index ``B*S``,
            so the scatter target is allocated one row longer and that row is discarded.
        token_sample: ``long (B*S,)`` which sample owns each token; slack points at the
            dustbin index ``N``. Used to expand per-sample sigma into the per-token
            ``timestep`` the transformer accepts.
    """

    num_samples: int
    num_rows: int
    pack_size: int
    seq_len: int
    text_budget: int
    text_capacity: int
    num_image_tokens: int
    segments_per_row: int
    max_seqlen: int
    cu_seqlens: torch.Tensor
    segment_ids: torch.Tensor
    position_ids: torch.Tensor
    indicator: torch.Tensor
    image_dst: torch.Tensor
    text_dst: torch.Tensor
    token_sample: torch.Tensor

    @property
    def dustbin_token_index(self) -> int:
        """Scatter target row that absorbs left-pad slots and is then sliced off."""
        return self.num_rows * self.seq_len

    @property
    def dustbin_sample_index(self) -> int:
        """``token_sample`` value for slack tokens, i.e. "no sample owns this"."""
        return self.num_samples


def derive_text_budget(*, pack_size: int, max_text_length: int, mean_text_length: float) -> int:
    """A default ``text_budget`` from the caption-length distribution.

    ``pack_size * max_text_length + 1`` is always feasible but buys nothing -- it is K copies
    of today's row. The useful budget sits near ``K * mean``, which is where the packing gain
    comes from. This returns enough room for one worst-case caption plus ``K-1`` average ones,
    which in practice the packer satisfies comfortably while still removing most of the
    padding. It is a DEFAULT, not a guarantee: :func:`build_packed_layout` raises if a row
    overflows, and the knob is there to be raised.

    For ``pack_size == 1`` this returns ``max_text_length + 1`` exactly, which reproduces the
    one-sample layout (a caption filling the full width still leaves the reserved slack slot).

    Args:
        pack_size: ``K``, samples per row.
        max_text_length: longest caption in the dataset, in tokens.
        mean_text_length: mean caption length in tokens.

    Returns:
        A ``text_budget`` in tokens, including the reserved slack slot.
    """
    if pack_size < 1:
        raise ValueError(f"pack_size must be >= 1, got {pack_size}")
    per_extra = int(-(-mean_text_length // 1))  # ceil, without importing math for one call
    return int(max_text_length) + (int(pack_size) - 1) * per_extra + 1


def build_cu_seqlens(
    text_lengths: Sequence[int],
    max_text_tokens: int,
    num_image_tokens: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Var-len packing for the ONE-sample ``[left-pad][text][image]`` layout, on the HOST.

    Retained as the ``pack_size == 1`` reference: :func:`build_packed_layout` must reproduce
    this exactly, which is what pins the generalization to the layout that has already been
    validated in production.

    Each row ``b`` of the flattened ``(B*S)`` sequence (``S = max_text_tokens +
    num_image_tokens``) contributes exactly TWO segments, matching what the model's
    ``(seg_i == seg_j)`` mask encodes:

      * ``[b*S, b*S+offset)``      the left-pad block (attends only to itself; discarded)
      * ``[b*S+offset, (b+1)*S)``  the text+image block (Ideogram attends these jointly)

    with ``offset = max_text_tokens - text_lengths[b]``.

    Because the count is always ``2*B``, the returned tensor's length is always ``2*B+1``
    -- independent of the captions in the batch. That is what keeps the compiled graph
    reusable: a segment count that varied with the data would change this tensor's shape
    and force a recompile per distinct length pattern.

    Args:
        text_lengths: per-sample real (non-pad) text token counts, as Python ints.
        max_text_tokens: padded width of the text region, INCLUDING the reserved pad
            column, so that ``offset >= 1`` on every row.
        num_image_tokens: ``grid_h * grid_w``.
        device: destination device. Building here costs one host->device copy; the thing
            being avoided is device->host reads inside the compiled region.

    Returns:
        ``int32`` ``(2*len(text_lengths)+1,)`` cumulative segment starts, beginning at 0
        and ending at ``B*S`` -- the ``flash_attn_varlen_func`` contract.

    Raises:
        ValueError: if any row would produce an empty pad segment (``offset == 0``).
            Zero-length segments are not representable for var-len flash, and the
            reserved pad column exists precisely to make this unreachable.
    """
    seq_len = max_text_tokens + num_image_tokens
    starts: List[int] = []
    for b, num_text in enumerate(text_lengths):
        offset = max_text_tokens - int(num_text)
        if offset < 1:
            raise ValueError(
                f"row {b} has text_length={int(num_text)} with max_text_tokens={max_text_tokens}, "
                "leaving no padding. Every row needs >=1 pad token so the segment count (and "
                "therefore cu_seqlens' shape) is the same for every batch."
            )
        base = b * seq_len
        starts.append(base)
        starts.append(base + offset)
    starts.append(len(text_lengths) * seq_len)
    return torch.tensor(starts, dtype=torch.int32, device=device)


def build_packed_layout(
    text_lengths: Sequence[int],
    *,
    pack_size: int,
    text_budget: int,
    grid_h: int,
    grid_w: int,
    text_capacity: int,
    device: Optional[torch.device] = None,
) -> PackedLayout:
    """Lay ``N`` samples out as ``B = N // pack_size`` packed rows.

    Samples are grouped in the order given: row ``j`` holds ``text_lengths[j*K : (j+1)*K]``.
    Deciding that order is the batch sampler's job -- see the module docstring.

    ``device`` moves the per-token tensors but deliberately NOT ``cu_seqlens``, mirroring the
    split that the one-sample path already uses: the per-token ids are consumed by the model
    as ordinary inputs, while ``cu_seqlens`` is published into the attention modules' shared
    buffer with a single ``copy_`` and so is cheapest to hand over on the host.

    Args:
        text_lengths: ``N`` real (non-pad) caption lengths, as Python ints or 0-d tensors.
        pack_size: ``K`` samples per row. Must divide ``N``.
        text_budget: token slots a row reserves for captions, slack included. Every row must
            satisfy ``sum(t_i) <= text_budget - 1``.
        grid_h: latent grid height (post-patchify).
        grid_w: latent grid width.
        text_capacity: width of the incoming left-padded ``llm_features`` text axis. Only
            ``text_dst`` depends on it; it is NOT the row's text budget.
        device: destination for the per-token tensors. ``None`` leaves everything on the host.

    Returns:
        A :class:`PackedLayout`.

    Raises:
        ValueError: if ``pack_size`` does not divide ``N``; if a caption is empty or longer
            than ``text_capacity``; or if a row's captions leave no slack. All three are
            conditions the sampler is supposed to have made unreachable, so they are raised
            rather than repaired -- a silently truncated caption does not show up in the loss
            curve.
    """
    lengths = [int(t) for t in text_lengths]
    num_samples = len(lengths)
    k = int(pack_size)
    if k < 1:
        raise ValueError(f"pack_size must be >= 1, got {k}")
    if num_samples == 0:
        raise ValueError("text_lengths is empty; there is nothing to pack")
    if num_samples % k:
        raise ValueError(
            f"pack_size={k} does not divide the micro-batch size {num_samples}. The row count "
            "would then vary between steps and recompile the graph; make local_batch_size a "
            "multiple of pack_size (and keep drop_last=true)."
        )

    num_rows = num_samples // k
    num_image_tokens = int(grid_h) * int(grid_w)
    text_budget = int(text_budget)
    text_capacity = int(text_capacity)
    seq_len = text_budget + k * num_image_tokens
    segments_per_row = k + 1
    total_tokens = num_rows * seq_len

    for i, t in enumerate(lengths):
        if t < 1:
            raise ValueError(f"sample {i} has text_length={t}; every sample needs >=1 real text token")
        if t > text_capacity:
            raise ValueError(
                f"sample {i} has text_length={t}, longer than the llm_features text width "
                f"{text_capacity}; the dataloader must left-pad to at least the longest caption."
            )

    # Shared across every sample: the grid is fixed in this phase, and text positions are
    # sliced from one arange rather than rebuilt per sample.
    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
    text_pos_all = torch.arange(text_capacity).view(-1, 1).expand(text_capacity, 3)
    image_offsets = torch.arange(num_image_tokens)

    segment_ids = torch.full((num_rows, seq_len), SLACK_SEGMENT_ID, dtype=torch.long)
    # Slack keeps indicator 0, as the one-sample layout's left-pad region does: the model
    # indexes embed_image_indicator with (indicator == OUTPUT_IMAGE_INDICATOR), so 0 and the
    # padding indicator land on the same embedding row anyway.
    indicator = torch.zeros((num_rows, seq_len), dtype=torch.long)
    position_ids = torch.zeros((num_rows, seq_len, 3), dtype=torch.long)
    token_sample = torch.full((total_tokens,), num_samples, dtype=torch.long)
    image_dst = torch.empty((num_samples * num_image_tokens,), dtype=torch.long)
    text_dst = torch.full((num_samples * text_capacity,), total_tokens, dtype=torch.long)

    starts: List[int] = []
    for row in range(num_rows):
        row_lengths = lengths[row * k : (row + 1) * k]
        slack = text_budget - sum(row_lengths)
        if slack < 1:
            raise ValueError(
                f"row {row} packs captions {row_lengths} totalling {sum(row_lengths)} tokens, "
                f"which leaves no slack in text_budget={text_budget}. Every row needs >=1 slack "
                "token so the segment count -- and therefore cu_seqlens' shape -- is the same "
                "for every batch, and because var-len flash cannot represent a zero-length "
                "segment. Raise text_budget or lower pack_size."
            )

        base = row * seq_len
        starts.append(base)  # the slack segment
        cursor = slack
        for local, t in enumerate(row_lengths):
            sample = row * k + local
            text_lo = cursor
            text_hi = text_lo + t
            image_hi = text_hi + num_image_tokens

            starts.append(base + text_lo)
            # 1..K rather than the global sample index: the mask only compares ids WITHIN a
            # row, and keeping them small makes a segment_ids dump readable at any batch size.
            segment_ids[row, text_lo:image_hi] = local + 1
            indicator[row, text_lo:text_hi] = LLM_TOKEN_INDICATOR
            indicator[row, text_hi:image_hi] = OUTPUT_IMAGE_INDICATOR
            position_ids[row, text_lo:text_hi] = text_pos_all[:t]
            position_ids[row, text_hi:image_hi] = image_pos
            token_sample[base + text_lo : base + image_hi] = sample

            image_dst[sample * num_image_tokens : (sample + 1) * num_image_tokens] = (
                base + text_hi + image_offsets
            )
            # llm_features is LEFT-padded, so sample `sample`'s real tokens occupy its last
            # ``t`` slots. Everything before that keeps pointing at the dustbin.
            slot_lo = sample * text_capacity + (text_capacity - t)
            text_dst[slot_lo : (sample + 1) * text_capacity] = base + text_lo + torch.arange(t)

            cursor = image_hi

        if cursor != seq_len:
            raise AssertionError(
                f"row {row} consumed {cursor} of {seq_len} slots; the layout arithmetic and "
                "seq_len have diverged"
            )
    starts.append(total_tokens)

    cu_seqlens = torch.tensor(starts, dtype=torch.int32)
    if cu_seqlens.numel() != num_rows * segments_per_row + 1:
        raise AssertionError(
            f"built {cu_seqlens.numel()} cu_seqlens entries, expected "
            f"{num_rows * segments_per_row + 1}; the segment count is no longer data-independent"
        )

    if device is not None:
        segment_ids = segment_ids.to(device)
        position_ids = position_ids.to(device)
        indicator = indicator.to(device)
        image_dst = image_dst.to(device)
        text_dst = text_dst.to(device)
        token_sample = token_sample.to(device)

    return PackedLayout(
        num_samples=num_samples,
        num_rows=num_rows,
        pack_size=k,
        seq_len=seq_len,
        text_budget=text_budget,
        text_capacity=text_capacity,
        num_image_tokens=num_image_tokens,
        segments_per_row=segments_per_row,
        # Static and data-independent: the largest a sample segment can be is
        # (text_budget - 1) + num_image_tokens and the largest slack can be is text_budget - K,
        # so this bounds both. For K == 1 it equals seq_len, which is what the one-sample path
        # already passes.
        max_seqlen=text_budget + num_image_tokens,
        cu_seqlens=cu_seqlens,
        segment_ids=segment_ids,
        position_ids=position_ids,
        indicator=indicator,
        image_dst=image_dst,
        text_dst=text_dst,
        token_sample=token_sample,
    )
