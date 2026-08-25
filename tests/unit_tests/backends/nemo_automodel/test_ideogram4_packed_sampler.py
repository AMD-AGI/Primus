###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 packed batch sampler and its bin packer.

WHY THIS TEST EXISTS:
  The sampler decides which samples share a transformer row. Three of its guarantees have
  failure modes that nothing inside the model would catch:

    1. **Feasibility.** ``build_packed_layout`` raises if a row exceeds the text budget, so an
       infeasible grouping is a crash mid-epoch -- possibly hours in, possibly only on one rank
       and only for one unlucky shuffle. The distribution sweep below turns that into a test
       failure instead of a training failure, which is why it sweeps caption distributions
       (uniform, normal, bimodal, long-tailed) rather than one well-behaved one: the default
       budget is derived from a mean, and a bimodal corpus -- where the mean describes no actual
       sample -- is exactly where a mean-based estimate would break.
    2. **Equal batch counts across ranks.** A rank that yields fewer batches than its peers
       leaves them blocked in a collective. That is a hang, not an error.
    3. **Exact cardinality.** Every row must hold exactly ``pack_size`` samples, or the segment
       count changes and torch.compile recompiles.

  The packer's output QUALITY is asserted too (peak row sum near the lower bound), because a
  correct-but-badly-balanced packing silently gives back the throughput the feature exists to
  win: it would pass every correctness test and show up only as a disappointing benchmark.
"""

import random

import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    pytest.skip("Ideogram-4 packed sampler tests require torch", allow_module_level=True)

from primus.backends.nemo_automodel.models.ideogram4.data.packed_sampler import (
    Ideogram4PackedBatchSampler,
    deal_longest_first,
    pack_rows,
)
from primus.backends.nemo_automodel.models.ideogram4.packing import (
    build_packed_layout,
    derive_text_budget,
)


def _lengths(kind: str, count: int, seed: int = 0):
    """Caption-length corpora that stress the packer differently.

    ``bimodal`` and ``long_tailed`` are the ones that matter: in the first, the mean the budget
    is derived from describes no actual sample; in the second, one caption is several times the
    mean, so the row holding it has to be filled with the shortest ones available.
    """
    rng = random.Random(seed)
    if kind == "uniform":
        return [rng.randint(1, 640) for _ in range(count)]
    if kind == "normal":
        return [max(1, min(640, int(rng.gauss(380, 90)))) for _ in range(count)]
    if kind == "bimodal":
        return [rng.choice([rng.randint(20, 60), rng.randint(500, 640)]) for _ in range(count)]
    if kind == "long_tailed":
        return [max(1, min(640, int(rng.expovariate(1 / 120)))) for _ in range(count)]
    if kind == "constant":
        return [383] * count
    raise AssertionError(f"unknown corpus {kind}")


CORPORA = ["uniform", "normal", "bimodal", "long_tailed", "constant"]
PACK_SIZES = [1, 2, 4, 8]


def _budget(lengths, pack_size):
    return derive_text_budget(
        pack_size=pack_size,
        max_text_length=max(lengths),
        mean_text_length=sum(lengths) / len(lengths),
    )


def _lower_bound(lengths, pack_size):
    """Two hard bounds on the peak row sum; whichever is larger.

    Some row must be at least the average (``pack_size * mean``), and the row holding the
    longest caption cannot be lighter than that caption. Which one binds depends on the corpus:
    the longest caption for a long tail, the average for a bimodal one.
    """
    return max(pack_size * sum(lengths) / len(lengths), float(max(lengths)))


def _sampler(lengths, *, pack_size, text_budget, batch=8, world=1, rank=0, seed=1, shuffle=True):
    return Ideogram4PackedBatchSampler(
        text_lengths=lengths,
        samples_per_batch=batch,
        pack_size=pack_size,
        text_budget=text_budget,
        num_replicas=world,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )


class TestPackRows:
    """The packer itself: exact cardinality, budget respected, nothing lost or duplicated."""

    @pytest.mark.parametrize("pack_size", PACK_SIZES)
    @pytest.mark.parametrize("corpus", CORPORA)
    def test_every_row_fits_and_holds_exactly_pack_size(self, corpus, pack_size):
        lengths = _lengths(corpus, 512)
        budget = _budget(lengths, pack_size)
        rows = pack_rows(list(enumerate(lengths)), pack_size=pack_size, text_budget=budget)

        assert len(rows) == len(lengths) // pack_size
        assert sorted(i for row in rows for i in row) == list(
            range(len(lengths))
        ), "the packer lost or duplicated a sample; every index must appear exactly once"
        for row in rows:
            assert len(row) == pack_size
            total = sum(lengths[i] for i in row)
            assert total <= budget - 1, (
                f"row {row} sums to {total} tokens, over the {budget - 1} usable slots in "
                f"text_budget={budget}. build_packed_layout would raise on this mid-epoch."
            )

    @pytest.mark.parametrize("pack_size", [2, 4, 8])
    @pytest.mark.parametrize("corpus", CORPORA)
    def test_peak_row_is_close_to_the_lower_bound(self, corpus, pack_size):
        """Correct but unbalanced packing gives back the throughput this feature exists to win."""
        lengths = _lengths(corpus, 4096)
        budget = _budget(lengths, pack_size)
        rows = pack_rows(list(enumerate(lengths)), pack_size=pack_size, text_budget=budget)

        peak = max(sum(lengths[i] for i in row) for row in rows)
        bound = _lower_bound(lengths, pack_size)
        assert peak >= bound * 0.99, "a peak below both lower bounds is arithmetically impossible"
        assert peak <= bound * 1.15, (
            f"{corpus}/K={pack_size}: peak row {peak:.0f} tokens against a {bound:.0f}-token "
            f"lower bound ({peak / bound:.2f}x). The packing is valid but wasteful -- rows are "
            "carrying slack that the whole feature exists to remove."
        )

    @pytest.mark.parametrize("corpus", ["long_tailed", "bimodal"])
    def test_longest_first_beats_snake_dealing_on_skewed_corpora(self, corpus):
        """Why the packer deals by running total instead of by length stratum.

        Snake dealing (one item per row per quartile) is cheaper and balances a uniform corpus
        just as well, but it hands the row holding the longest caption a set of mid-length
        companions. On a skewed corpus that alone pushes the peak row past the budget.
        """
        lengths = _lengths(corpus, 4096, seed=5)
        items = list(enumerate(lengths))
        pack_size, num_rows = 4, len(lengths) // 4

        actual = max(sum(length for _, length in row) for row in deal_longest_first(items, num_rows))

        order = sorted(items, key=lambda item: (-item[1], item[0]))
        snake = [[] for _ in range(num_rows)]
        cursor = 0
        for slot in range(pack_size):
            sweep = range(num_rows) if slot % 2 == 0 else range(num_rows - 1, -1, -1)
            for row in sweep:
                snake[row].append(order[cursor])
                cursor += 1
        snake_peak = max(sum(length for _, length in row) for row in snake)

        assert actual < snake_peak, (
            f"{corpus}: dealing by running total peaked at {actual} vs snake dealing's "
            f"{snake_peak}; beating the cheaper stratified deal here is the point of the heap"
        )
        bound = _lower_bound(lengths, pack_size)
        assert actual <= bound * 1.05, (
            f"{corpus}: peak row {actual} against a {bound:.0f}-token lower bound "
            f"({actual / bound:.3f}x); the longest captions are not being paired with the "
            "shortest ones"
        )

    def test_is_deterministic_and_order_independent(self):
        """Two ranks handed the same group must produce the same rows."""
        lengths = _lengths("uniform", 128, seed=9)
        items = list(enumerate(lengths))
        assert pack_rows(items, pack_size=4, text_budget=2000) == pack_rows(
            items, pack_size=4, text_budget=2000
        )
        # The same multiset in a different order must give the same partition, since the sort
        # is by (length, index) rather than by position.
        shuffled = list(items)
        random.Random(3).shuffle(shuffled)
        assert sorted(map(sorted, pack_rows(shuffled, pack_size=4, text_budget=2000))) == sorted(
            map(sorted, pack_rows(items, pack_size=4, text_budget=2000))
        )

    def test_rejects_a_caption_that_cannot_share_a_row(self):
        with pytest.raises(ValueError, match="cannot share a row"):
            pack_rows([(0, 100), (1, 5), (2, 5), (3, 5)], pack_size=2, text_budget=101)

    def test_rejects_an_infeasible_group_loudly(self):
        with pytest.raises(ValueError, match="could not pack"):
            pack_rows([(i, 60) for i in range(8)], pack_size=4, text_budget=200)

    def test_rejects_group_not_divisible_by_pack_size(self):
        with pytest.raises(ValueError, match="does not divide"):
            pack_rows([(0, 5), (1, 5), (2, 5)], pack_size=2, text_budget=100)

    def test_pack_size_one_preserves_the_given_order(self):
        """K=1 must not reorder by length, or it would undo the caller's shuffle."""
        rows = pack_rows([(7, 5), (3, 9), (1, 2)], pack_size=1, text_budget=10)
        assert rows == [[7], [3], [1]]


class TestSamplerShardingAndShapes:
    """Guarantees whose failure mode is a hang or a recompile, not a wrong number."""

    @pytest.mark.parametrize("world", [1, 2, 8])
    def test_all_ranks_yield_the_same_number_of_batches(self, world):
        lengths = _lengths("uniform", 1000)
        budget = _budget(lengths, 4)
        counts = {
            len(list(_sampler(lengths, pack_size=4, text_budget=budget, world=world, rank=r)))
            for r in range(world)
        }
        assert len(counts) == 1, (
            f"ranks disagree on batch count: {counts}. The short rank finishes its epoch early "
            "and its peers block forever in the next collective -- a hang, not an error."
        )
        assert counts != {0}

    @pytest.mark.parametrize("world", [2, 8])
    def test_ranks_see_disjoint_samples(self, world):
        lengths = _lengths("uniform", 1000)
        budget = _budget(lengths, 4)
        seen = [
            [
                i
                for batch in _sampler(lengths, pack_size=4, text_budget=budget, world=world, rank=r)
                for i in batch
            ]
            for r in range(world)
        ]
        flat = [i for rank in seen for i in rank]
        assert len(flat) == len(set(flat)), "two ranks trained on the same sample in one epoch"

    @pytest.mark.parametrize("pack_size", PACK_SIZES)
    def test_every_batch_has_the_configured_sample_count(self, pack_size):
        lengths = _lengths("normal", 1024)
        for batch in _sampler(lengths, pack_size=pack_size, text_budget=_budget(lengths, pack_size)):
            assert len(batch) == 8, (
                f"batch of {len(batch)} samples instead of 8; a short batch has fewer rows, so "
                "every packed tensor changes shape and torch.compile recompiles"
            )

    @pytest.mark.parametrize("pack_size", [2, 4])
    @pytest.mark.parametrize("corpus", CORPORA)
    def test_batches_are_row_major_and_the_layout_accepts_them(self, corpus, pack_size):
        """The end-to-end contract that ties the two halves of the feature together.

        The sampler promises consecutive runs of ``pack_size`` share a row and fit the budget;
        ``build_packed_layout`` is the thing that raises if they do not.
        """
        lengths = _lengths(corpus, 512)
        budget = _budget(lengths, pack_size)
        for batch in _sampler(lengths, pack_size=pack_size, text_budget=budget):
            layout = build_packed_layout(
                [lengths[i] for i in batch],
                pack_size=pack_size,
                text_budget=budget,
                grid_h=2,
                grid_w=2,
                text_capacity=max(lengths),
            )
            assert layout.num_rows == 8 // pack_size

    def test_shuffle_is_deterministic_per_epoch_and_changes_between_epochs(self):
        lengths = _lengths("uniform", 512)
        budget = _budget(lengths, 4)

        def batches(epoch):
            sampler = _sampler(lengths, pack_size=4, text_budget=budget)
            sampler.set_epoch(epoch)
            return list(sampler)

        assert batches(0) == batches(0), "same epoch gave different batches; runs are not reproducible"
        assert batches(1) != batches(0), (
            "set_epoch did not change the batches. Length pairing is stable across epochs by "
            "design, so without the row shuffle the same samples would share a batch forever."
        )

    def test_epoch_reshuffles_batch_membership_not_just_order(self):
        """A permuted batch ORDER would pass the test above while changing nothing that matters."""
        lengths = _lengths("uniform", 512)
        budget = _budget(lengths, 4)

        def membership(epoch):
            sampler = _sampler(lengths, pack_size=4, text_budget=budget)
            sampler.set_epoch(epoch)
            return {frozenset(batch) for batch in sampler}

        assert membership(0) != membership(1), (
            "the epoch shuffle only reordered the batches; which samples share a gradient step "
            "must change too"
        )

    def test_no_shuffle_is_reproducible(self):
        lengths = _lengths("normal", 256)
        budget = _budget(lengths, 2)
        first = list(_sampler(lengths, pack_size=2, text_budget=budget, shuffle=False))
        assert first == list(_sampler(lengths, pack_size=2, text_budget=budget, shuffle=False))


class TestSamplerRejections:
    def test_rejects_batch_size_not_a_multiple_of_pack_size(self):
        with pytest.raises(ValueError, match="not a multiple of pack_size"):
            _sampler([10] * 64, pack_size=4, text_budget=100, batch=6)

    def test_rejects_drop_last_false(self):
        with pytest.raises(ValueError, match="requires drop_last=True"):
            Ideogram4PackedBatchSampler(
                text_lengths=[10] * 64,
                samples_per_batch=8,
                pack_size=4,
                text_budget=100,
                drop_last=False,
            )

    def test_rejects_a_dataset_too_small_for_one_batch(self):
        with pytest.raises(ValueError, match="fewer than one batch"):
            _sampler([10] * 8, pack_size=2, text_budget=100, batch=8, world=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
