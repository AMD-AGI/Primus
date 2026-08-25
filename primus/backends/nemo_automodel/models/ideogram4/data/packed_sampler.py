###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Which samples share a packed row: the cardinality-constrained bin packer.

WHY THIS IS A SAMPLER AND NOT PART OF THE LAYOUT:
  ``packing.py`` lays out a row once the group is chosen, and raises if the group does not
  fit. Choosing the group needs to see more than one row at a time, which makes it a
  sampling decision. Keeping the two apart means the layout has no feasibility policy to get
  wrong, and this module has no tensor arithmetic to get wrong.

THE CONSTRAINT, which is what makes this not ordinary bin packing:
  Every row must hold EXACTLY ``pack_size`` samples -- not "at most", not "as many as fit".
  A row with a different count has a different segment count, which changes ``cu_seqlens``'
  shape and recompiles the graph; the whole static-shape strategy rests on that number being
  constant. So this is a *balanced* partition problem (fixed number of bins, fixed
  cardinality, minimize the maximum sum) rather than the usual "use as few bins as possible".

  That difference is why the textbook bin-packing heuristics do not transfer directly -- they
  choose how many bins to open. What this reduces to instead is makespan minimization on a
  fixed number of machines with a job-count cap, and the standard rule for that is LPT
  (longest processing time first): sort descending and give each caption to the row with the
  smallest running total that still has a free slot.

  The reason LPT and not the cheaper snake dealing (deal the sorted list across the rows in
  boustrophedon order) is caption-length distributions with a long tail, which real corpora
  have. Snake dealing balances *strata* -- each row gets one item from each quartile -- so the
  row that receives the single longest caption also receives mid-length companions and comes
  out far heavier than the average. LPT leaves that row alone until every other row has caught
  up, so the longest caption ends up paired with the shortest ones, which is the arrangement
  the budget is sized for. On an exponential corpus at K=4 that is the difference between a
  peak row of 811 tokens and 643 -- and since the longest single caption is 640, 643 is within
  a hair of what any arrangement could achieve.

  LPT is a heuristic, not a guarantee, so :func:`pack_rows` verifies the budget afterwards,
  attempts a bounded repair by swapping between the heaviest and lightest rows, and RAISES if
  it still cannot fit. It never truncates a caption to make the numbers work: a silently
  shortened caption does not show up in the loss curve.

ROWS ARE PACKED OVER THE WHOLE EPOCH SHARD, THEN SHUFFLED INTO BATCHES:
  Packing within each micro-batch is the obvious design and it is measurably worse. With
  ``samples_per_batch = 8`` and ``K = 2`` there are four rows to balance and the deal is
  already optimal over them, so when a batch happens to hold several long captions there is
  nothing left to repair and the budget has to absorb the whole variance. Packing across the
  rank's entire shard instead gives the packer thousands of captions to draw from, so the row
  holding the longest caption can be filled out with the shortest ones.

  This is sound because of a property specific to packing: the segment mask makes the samples
  in a row mutually invisible, so WHICH samples share a row changes nothing about the
  gradients -- it is purely a padding-efficiency question. What must stay random is which
  samples share a BATCH, since that does affect the gradient, and it does: the rows are
  shuffled before being dealt into batches, so batch composition is re-rolled every epoch even
  though the length pairing is stable.

  Every batch still holds exactly ``samples_per_batch // pack_size`` rows, which is the number
  the compiled graph is keyed on.
"""
from __future__ import annotations

import heapq
import logging
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


def deal_longest_first(items: Sequence[Tuple[int, int]], num_rows: int) -> List[List[Tuple[int, int]]]:
    """Deal ``items`` across ``num_rows`` equal-cardinality rows, longest caption first.

    Each caption goes to the row with the smallest running total that still has a free slot
    (LPT / worst-fit-decreasing). A row leaves the heap once it is full, which is what enforces
    the exact cardinality the static segment count depends on.

    Args:
        items: ``(index, length)`` pairs. ``len(items)`` must be divisible by ``num_rows``.
        num_rows: how many rows to fill. Each receives ``len(items) // num_rows`` items.

    Returns:
        One list of ``(index, length)`` per row.
    """
    # Descending by length, index as tiebreak so the result is a pure function of the input
    # (two ranks handed the same group must produce the same rows). The heap key carries the
    # row index for the same reason -- equal sums must break the same way everywhere.
    order = sorted(items, key=lambda item: (-item[1], item[0]))
    per_row = len(items) // num_rows
    rows: List[List[Tuple[int, int]]] = [[] for _ in range(num_rows)]

    open_rows: List[Tuple[int, int, int]] = [(0, 0, row) for row in range(num_rows)]
    heapq.heapify(open_rows)
    for index, length in order:
        total, filled, row = heapq.heappop(open_rows)
        rows[row].append((index, length))
        if filled + 1 < per_row:
            heapq.heappush(open_rows, (total + length, filled + 1, row))
    return rows


def _repair(rows: List[List[Tuple[int, int]]], capacity: int, max_passes: int) -> bool:
    """Swap between the heaviest and lightest rows until every row fits, or give up.

    Returns True when every row is within ``capacity``. Each pass moves the single largest item
    out of the heaviest over-capacity row in exchange for the smallest item in the lightest
    row, which is the swap that reduces the maximum sum the most.
    """
    for _ in range(max_passes):
        sums = [sum(length for _, length in row) for row in rows]
        heaviest = max(range(len(rows)), key=lambda r: sums[r])
        if sums[heaviest] <= capacity:
            return True
        lightest = min(range(len(rows)), key=lambda r: sums[r])
        if heaviest == lightest:
            return False

        give = max(range(len(rows[heaviest])), key=lambda i: rows[heaviest][i][1])
        take = min(range(len(rows[lightest])), key=lambda i: rows[lightest][i][1])
        if rows[heaviest][give][1] <= rows[lightest][take][1]:
            # Nothing left to gain: the heaviest row's largest item is no larger than the
            # lightest row's smallest, so no swap can reduce the maximum.
            return False
        rows[heaviest][give], rows[lightest][take] = rows[lightest][take], rows[heaviest][give]

    sums = [sum(length for _, length in row) for row in rows]
    return max(sums) <= capacity


def pack_rows(
    items: Sequence[Tuple[int, int]],
    *,
    pack_size: int,
    text_budget: int,
    max_repair_passes: int = 16,
) -> List[List[int]]:
    """Partition ``items`` into rows of exactly ``pack_size`` that fit the text budget.

    Args:
        items: ``(sample_index, caption_length)`` pairs.
        pack_size: ``K``, samples per row. Must divide ``len(items)``.
        text_budget: token slots a row reserves for captions. Rows must satisfy
            ``sum(lengths) <= text_budget - 1``; the reserved slot keeps the slack segment
            non-empty, which var-len flash requires and the static segment count depends on.
        max_repair_passes: how many swap attempts to make before declaring the group
            infeasible.

    Returns:
        ``len(items) // pack_size`` rows, each a list of ``pack_size`` sample indices.

    Raises:
        ValueError: if ``pack_size`` does not divide the group, if a single caption cannot fit
            a row on its own, or if no arrangement found fits the budget.
    """
    if pack_size < 1:
        raise ValueError(f"pack_size must be >= 1, got {pack_size}")
    if len(items) % pack_size:
        raise ValueError(
            f"pack_size={pack_size} does not divide the group size {len(items)}; every row must "
            "hold exactly pack_size samples or the segment count stops being constant."
        )

    capacity = text_budget - 1
    if pack_size == 1:
        # Degenerate but worth short-circuiting: with one sample per row there is nothing to
        # balance, and the only thing that can fail is a caption wider than the budget. Order
        # is preserved, so K=1 leaves the caller's shuffle exactly as it was.
        for index, length in items:
            if length > capacity:
                raise ValueError(
                    f"sample {index} has a {length}-token caption but text_budget={text_budget} "
                    f"leaves only {capacity} usable slots. Raise text_budget."
                )
        return [[index] for index, _ in items]

    longest = max(items, key=lambda item: item[1])
    if longest[1] + (pack_size - 1) > capacity:
        # Every other sample in the row needs at least one token, so this is unsatisfiable
        # regardless of how the group is arranged. Reported separately because the fix is
        # different: the budget is too small for the corpus, not for this group.
        raise ValueError(
            f"sample {longest[0]} has a {longest[1]}-token caption, which cannot share a row "
            f"with {pack_size - 1} others inside text_budget={text_budget} (they need at least "
            f"one token each, and one slot stays slack). Raise text_budget to at least "
            f"{longest[1] + pack_size} or lower pack_size."
        )

    num_rows = len(items) // pack_size
    rows = deal_longest_first(items, num_rows)
    if not _repair(rows, capacity, max_repair_passes):
        sums = sorted((sum(length for _, length in row) for row in rows), reverse=True)
        raise ValueError(
            f"could not pack {len(items)} samples into {num_rows} rows of {pack_size} within "
            f"text_budget={text_budget} (usable {capacity}); the heaviest rows came to {sums[:3]} "
            "after repair. These captions are longer than the budget assumes. Raise text_budget, "
            "or lower pack_size."
        )
    return [[index for index, _ in row] for row in rows]


class Ideogram4PackedBatchSampler(Sampler[List[int]]):
    """Yields micro-batches whose samples are already grouped into packed rows.

    Each yielded batch is a flat list of ``samples_per_batch`` dataset indices in ROW-MAJOR
    order: the first ``pack_size`` indices share row 0, the next ``pack_size`` share row 1, and
    so on. That is exactly the order :func:`packing.build_packed_layout` assumes, so the
    collate has nothing to decide.

    Sharding mirrors ``DistributedSampler`` with ``drop_last=True`` -- truncate to a multiple of
    the world size, then stride by rank. Every rank therefore yields the SAME number of
    batches, which is not a nicety: a rank that ran out of batches early would leave its peers
    waiting in a collective.

    Args:
        text_lengths: caption length for every sample in the dataset, indexed by dataset index.
            Read from the cache metadata, so constructing this costs no sample loads.
        samples_per_batch: samples per rank per step (``local_batch_size``). Must be a multiple
            of ``pack_size``.
        pack_size: ``K``, samples per row.
        text_budget: token slots a row reserves for captions, slack included.
        num_replicas: data-parallel world size.
        rank: this process's data-parallel rank.
        shuffle: reshuffle every epoch. :meth:`set_epoch` must be called for this to take
            effect, which the diffusion recipe's epoch loop does.
        seed: base seed; the epoch is added to it.
        drop_last: must be True. A partial final batch has fewer rows, which changes every
            packed shape and recompiles the graph.
    """

    def __init__(
        self,
        *,
        text_lengths: Sequence[int],
        samples_per_batch: int,
        pack_size: int,
        text_budget: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if pack_size < 1:
            raise ValueError(f"pack_size must be >= 1, got {pack_size}")
        if samples_per_batch % pack_size:
            raise ValueError(
                f"local_batch_size={samples_per_batch} is not a multiple of pack_size={pack_size}; "
                "the row count per step would vary and recompile the graph."
            )
        if not drop_last:
            raise ValueError(
                "Ideogram4PackedBatchSampler requires drop_last=True. A short final batch has "
                "fewer rows than the others, so every packed tensor changes shape and "
                "torch.compile recompiles -- for one batch per epoch."
            )

        self.text_lengths = [int(t) for t in text_lengths]
        self.samples_per_batch = int(samples_per_batch)
        self.pack_size = int(pack_size)
        self.text_budget = int(text_budget)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        # Mirror DistributedSampler(drop_last=True): every rank sees the same count.
        self.samples_per_rank = len(self.text_lengths) // self.num_replicas
        self.num_batches = self.samples_per_rank // self.samples_per_batch
        if self.num_batches == 0:
            raise ValueError(
                f"{len(self.text_lengths)} samples over {self.num_replicas} rank(s) gives "
                f"{self.samples_per_rank} per rank, which is fewer than one batch of "
                f"{self.samples_per_batch}. Lower local_batch_size or add data."
            )

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for the next epoch. Called by the diffusion recipe's epoch loop."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            order = torch.randperm(len(self.text_lengths), generator=generator).tolist()
        else:
            order = list(range(len(self.text_lengths)))

        order = order[: self.samples_per_rank * self.num_replicas]
        mine = order[self.rank :: self.num_replicas]
        # Drop the tail that cannot fill a whole batch BEFORE packing, so the rows divide
        # evenly into batches.
        mine = mine[: self.num_batches * self.samples_per_batch]

        rows = pack_rows(
            [(index, self.text_lengths[index]) for index in mine],
            pack_size=self.pack_size,
            text_budget=self.text_budget,
        )

        # Length pairing is stable across epochs (sorting by length undoes the shuffle), so
        # without this the same samples would share a batch every epoch. Row membership does
        # not affect gradients; batch membership does.
        if self.shuffle:
            rows = [rows[i] for i in torch.randperm(len(rows), generator=generator).tolist()]

        rows_per_batch = self.samples_per_batch // self.pack_size
        for batch in range(self.num_batches):
            group = rows[batch * rows_per_batch : (batch + 1) * rows_per_batch]
            yield [index for row in group for index in row]
