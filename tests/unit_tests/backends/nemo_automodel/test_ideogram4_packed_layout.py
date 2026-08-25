###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 multi-sample row layout (``build_packed_layout``).

WHY THIS TEST EXISTS:
  Packing K samples into one row is a change whose failure mode is silence. If two samples
  in a row can attend to each other, the loss still descends and the model is quietly
  corrupt; if an index tensor is off by the left-pad offset, the model trains on the wrong
  conditioning for every sample and nothing raises. So the assertions here are exact
  equalities against two independent references, not tolerances:

    1. ``pack_size=1`` must reproduce ``build_cu_seqlens`` -- the one-sample builder that is
       already validated in production. This pins the generalization to a known-good layout.
    2. For every K, the host-built ``cu_seqlens`` must equal what
       ``blockdiag_bool_mask_to_cu_seqlens`` derives from the mask the MODEL actually builds
       out of ``segment_ids``. This is the assertion that catches cross-sample attention: if
       the two disagree, var-len flash and the model's own mask describe different segments.

  The index tensors get round-trip tests rather than hand-written expected values, because a
  hand-written expectation would encode the same off-by-one the implementation might have.

Only ``torch`` is required. The ``_prepare_ids`` cross-check needs ``nemo_automodel`` (the
adapter class is built lazily against its ``ModelAdapter`` base) and skips without it.
"""

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - CPU-less lint environments
    pytest.skip("Ideogram-4 packed layout tests require torch", allow_module_level=True)

from primus.backends.nemo_automodel.models.ideogram4.attention import (
    blockdiag_bool_mask_to_cu_seqlens,
)
from primus.backends.nemo_automodel.models.ideogram4.packing import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SLACK_SEGMENT_ID,
    build_cu_seqlens,
    build_packed_layout,
    derive_text_budget,
)


def _bool_mask(segment_ids):
    """The dense ``(B,1,L,L)`` mask the diffusers block materializes from segment ids."""
    return (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)


# (label, text_lengths, pack_size, text_budget, grid_h, grid_w, text_capacity)
# text_budget always leaves >=1 slack per row, which the sampler is responsible for in a
# real run.
PACKED_CASES = [
    ("k1_min_slack", [4, 4], 1, 5, 1, 3, 4),
    ("k1_ragged", [1, 2, 3, 4], 1, 5, 1, 3, 4),
    ("k2_ragged", [3, 7, 2, 9], 2, 21, 2, 3, 12),
    ("k2_min_slack", [10, 10, 4, 4], 2, 21, 2, 3, 12),
    ("k2_all_equal", [5, 5, 5, 5], 2, 21, 2, 3, 12),
    ("k4_ragged", [3, 7, 2, 9, 5, 4, 8, 1], 4, 41, 2, 3, 12),
    ("k4_single_row", [1, 2, 3, 4], 4, 11, 4, 4, 6),
    ("k8_wide_image", [1, 3, 5, 7, 9, 11, 13, 15], 8, 65, 8, 8, 16),
]
PACKED_IDS = [c[0] for c in PACKED_CASES]
PACKED_ARGS = [c[1:] for c in PACKED_CASES]


def _layout(text_lengths, pack_size, text_budget, grid_h, grid_w, text_capacity):
    return build_packed_layout(
        text_lengths,
        pack_size=pack_size,
        text_budget=text_budget,
        grid_h=grid_h,
        grid_w=grid_w,
        text_capacity=text_capacity,
    )


class TestOneSampleIdentity:
    """``pack_size=1`` must be the layout that already ships, not merely equivalent to it."""

    @pytest.mark.parametrize(
        "text_lengths,text_capacity,grid_h,grid_w",
        [([4, 4], 4, 1, 3), ([1, 2, 3, 4], 4, 1, 3), ([7, 13, 2], 15, 8, 8), ([2], 4, 1, 7)],
    )
    def test_cu_seqlens_matches_the_shipped_builder(self, text_lengths, text_capacity, grid_h, grid_w):
        # text_budget = text_capacity + 1 is the reserved-pad-column convention: a caption
        # filling the full width still leaves exactly one slack slot.
        budget = text_capacity + 1
        layout = _layout(text_lengths, 1, budget, grid_h, grid_w, text_capacity)
        reference = build_cu_seqlens(text_lengths, budget, grid_h * grid_w)

        assert torch.equal(layout.cu_seqlens, reference), (
            f"pack_size=1 diverged from build_cu_seqlens for text_lengths={text_lengths}: "
            f"got {layout.cu_seqlens.tolist()} vs {reference.tolist()}. The one-sample layout is "
            "the validated baseline; a difference here means the generalization changed "
            "production behaviour rather than extending it."
        )
        assert layout.segments_per_row == 2
        assert layout.max_seqlen == layout.seq_len, (
            "for K=1 the static bound is the whole row, which is what the one-sample adapter "
            "already passes to the kernel"
        )

    def test_matches_prepare_ids(self):
        """The per-token ids must match the adapter helper the model has been trained with."""
        pytest.importorskip(
            "nemo_automodel",
            reason="Ideogram-4 adapter class is built against nemo_automodel's ModelAdapter base",
        )
        from primus.backends.nemo_automodel.models.ideogram4.adapter import (
            get_ideogram4_adapter_class,
        )

        text_lengths, grid_h, grid_w, text_capacity = [2, 4, 1], 2, 3, 5
        budget = text_capacity + 1
        expected_pos, expected_seg, expected_ind = get_ideogram4_adapter_class()._prepare_ids(
            text_lengths, grid_h, grid_w, budget, torch.device("cpu")
        )

        layout = _layout(text_lengths, 1, budget, grid_h, grid_w, text_capacity)

        assert torch.equal(layout.segment_ids, expected_seg), "segment_ids drifted from _prepare_ids"
        assert torch.equal(layout.position_ids, expected_pos), "position_ids drifted from _prepare_ids"
        assert torch.equal(layout.indicator, expected_ind), "indicator drifted from _prepare_ids"


class TestMaskParity:
    """Host-built ``cu_seqlens`` must equal what the model's own mask implies, for every K."""

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_matches_mask_derived_reference(self, args):
        layout = _layout(*args)
        expected, expected_max, is_trivial = blockdiag_bool_mask_to_cu_seqlens(_bool_mask(layout.segment_ids))

        assert not is_trivial, (
            "every row here has slack, so the mask must not be trivial -- if it is, the "
            "fixture no longer exercises the var-len path"
        )
        assert layout.cu_seqlens.dtype == torch.int32, "flash_attn_varlen_func requires int32"
        assert torch.equal(layout.cu_seqlens, expected), (
            f"host-built cu_seqlens disagrees with the mask-derived reference: "
            f"{layout.cu_seqlens.tolist()} vs {expected.tolist()}. A mismatch means var-len "
            "flash attends across segment boundaries -- padding or a NEIGHBOURING SAMPLE leaks "
            "into real tokens and training corrupts with no error raised anywhere."
        )
        assert layout.max_seqlen >= expected_max, (
            f"static max_seqlen {layout.max_seqlen} under-estimates the longest real segment "
            f"{expected_max}; flash would be given a bound smaller than a sequence it must "
            "schedule."
        )

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_no_cross_sample_attention(self, args):
        """Directly: no token may attend to a token owned by a different sample."""
        layout = _layout(*args)
        mask = _bool_mask(layout.segment_ids)[:, 0]
        owner = layout.token_sample.view(layout.num_rows, layout.seq_len)

        for row in range(layout.num_rows):
            same_owner = owner[row].unsqueeze(1) == owner[row].unsqueeze(0)
            leak = mask[row] & ~same_owner
            assert not bool(
                leak.any()
            ), f"row {row} lets {int(leak.sum())} token pairs attend across sample boundaries"


class TestStaticShapes:
    """The properties torch.compile depends on: nothing may vary with the captions."""

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_segment_count_is_k_plus_one_per_row(self, args):
        layout = _layout(*args)
        expected = layout.num_rows * (layout.pack_size + 1) + 1
        assert layout.cu_seqlens.numel() == expected, (
            f"expected {expected} cu_seqlens entries for B={layout.num_rows}, K={layout.pack_size}, "
            f"got {layout.cu_seqlens.numel()}. A data-dependent length recompiles the graph per "
            "caption pattern."
        )
        assert layout.segments_per_row == layout.pack_size + 1

    def test_every_tensor_shape_is_independent_of_caption_lengths(self):
        """Same N and K, wildly different captions -> byte-identical shapes everywhere."""
        shapes = set()
        # K=4 over a 41-token budget leaves 40 slots per row, so every pattern here stays at
        # or below 10 tokens a caption -- the point is varying the lengths, not the budget.
        for lengths in ([1] * 8, [10] * 8, [1, 10, 5, 3, 9, 2, 8, 7], [7, 2, 10, 9, 1, 4, 6, 8]):
            layout = _layout(lengths, 4, 41, 2, 3, 12)
            shapes.add(
                (
                    tuple(layout.cu_seqlens.shape),
                    tuple(layout.segment_ids.shape),
                    tuple(layout.position_ids.shape),
                    tuple(layout.indicator.shape),
                    tuple(layout.image_dst.shape),
                    tuple(layout.text_dst.shape),
                    tuple(layout.token_sample.shape),
                    layout.max_seqlen,
                    layout.seq_len,
                )
            )
        assert len(shapes) == 1, f"a shape or static bound varied with caption lengths: {shapes}"

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_cu_seqlens_contract(self, args):
        """Starts at 0, ends at B*S, strictly increasing -- no empty segments."""
        layout = _layout(*args)
        assert int(layout.cu_seqlens[0]) == 0
        assert int(layout.cu_seqlens[-1]) == layout.num_rows * layout.seq_len
        diffs = layout.cu_seqlens[1:] - layout.cu_seqlens[:-1]
        assert bool((diffs > 0).all()), (
            f"zero-length or negative segment in {layout.cu_seqlens.tolist()}; var-len flash "
            "cannot represent an empty sequence"
        )

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_segment_lengths_match_the_layout(self, args):
        """Per row: one slack block, then one ``t_i + n_img`` block per sample."""
        text_lengths, pack_size, text_budget = args[0], args[1], args[2]
        layout = _layout(*args)
        diffs = (layout.cu_seqlens[1:] - layout.cu_seqlens[:-1]).tolist()

        expected = []
        for row in range(layout.num_rows):
            row_lengths = [int(t) for t in text_lengths[row * pack_size : (row + 1) * pack_size]]
            expected.append(text_budget - sum(row_lengths))
            expected += [t + layout.num_image_tokens for t in row_lengths]
        assert diffs == expected, f"segment lengths {diffs} do not match the layout {expected}"


class TestIndexTensors:
    """Round-trip rather than hand-written expectations, so an off-by-one cannot be shared."""

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_image_scatter_gather_is_identity(self, args):
        """Scatter distinct per-token values into the rows, gather them back unchanged."""
        layout = _layout(*args)
        n_img = layout.num_image_tokens
        src = torch.arange(1, layout.num_samples * n_img + 1, dtype=torch.float32).unsqueeze(1)

        buffer = torch.zeros(layout.dustbin_token_index + 1, 1)
        buffer.index_add_(0, layout.image_dst, src)
        gathered = buffer[: layout.dustbin_token_index].index_select(0, layout.image_dst)

        assert torch.equal(gathered, src), (
            "image_dst is not a bijection onto the image-token slots; the predicted velocity "
            "would be gathered from the wrong positions"
        )
        assert layout.image_dst.unique().numel() == layout.num_samples * n_img
        # Every destination must actually be an image slot in the model's own view.
        indicator_flat = layout.indicator.reshape(-1)
        assert bool((indicator_flat[layout.image_dst] == OUTPUT_IMAGE_INDICATOR).all())

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_text_dst_maps_left_padded_features(self, args):
        """Real caption slots land on text positions; left-pad slots land in the dustbin."""
        text_lengths, _, _, _, _, text_capacity = args
        layout = _layout(*args)
        lengths = [int(t) for t in text_lengths]
        dst = layout.text_dst.view(layout.num_samples, text_capacity)
        indicator_flat = layout.indicator.reshape(-1)
        owner = layout.token_sample

        for sample, t in enumerate(lengths):
            pad_slots = dst[sample, : text_capacity - t]
            real_slots = dst[sample, text_capacity - t :]
            assert bool((pad_slots == layout.dustbin_token_index).all()), (
                f"sample {sample}: left-pad slots must point at the dustbin, else zeroed pad "
                "features overwrite real tokens"
            )
            assert real_slots.numel() == t
            assert bool(
                (indicator_flat[real_slots] == LLM_TOKEN_INDICATOR).all()
            ), f"sample {sample}: a real caption slot landed outside the text region"
            assert bool(
                (owner[real_slots] == sample).all()
            ), f"sample {sample}: its caption features landed in ANOTHER sample's segment"
            # Left-padded input means the real tokens are the LAST t; they must stay in order.
            assert torch.equal(real_slots, torch.arange(int(real_slots[0]), int(real_slots[0]) + t))

        real = layout.text_dst[layout.text_dst != layout.dustbin_token_index]
        assert (
            real.unique().numel() == real.numel() == sum(lengths)
        ), "two caption slots share a destination; index_add_ would sum two samples' features"

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_token_sample_covers_exactly_the_non_slack_tokens(self, args):
        text_lengths = [int(t) for t in args[0]]
        layout = _layout(*args)

        occupied = layout.segment_ids.reshape(-1) != SLACK_SEGMENT_ID
        assert torch.equal(layout.token_sample != layout.dustbin_sample_index, occupied), (
            "token_sample and segment_ids disagree about which slots are slack; the per-token "
            "timestep would be applied to a different set of tokens than the mask isolates"
        )
        counts = torch.bincount(layout.token_sample, minlength=layout.num_samples + 1)
        for sample, t in enumerate(text_lengths):
            assert int(counts[sample]) == t + layout.num_image_tokens, (
                f"sample {sample} owns {int(counts[sample])} tokens, expected "
                f"{t + layout.num_image_tokens}"
            )

    @pytest.mark.parametrize("args", PACKED_ARGS, ids=PACKED_IDS)
    def test_per_token_timestep_expansion(self, args):
        """The use token_sample exists for: broadcast per-sample sigma over its own span."""
        layout = _layout(*args)
        sigma = torch.linspace(0.1, 0.9, layout.num_samples)
        padded = torch.cat([1.0 - sigma, torch.zeros(1)])

        timestep = padded[layout.token_sample].view(layout.num_rows, layout.seq_len)

        assert timestep.shape == (layout.num_rows, layout.seq_len)
        owner = layout.token_sample.view(layout.num_rows, layout.seq_len)
        for sample in range(layout.num_samples):
            span = owner == sample
            assert bool(
                (timestep[span] == (1.0 - sigma[sample])).all()
            ), f"sample {sample}'s timestep did not cover exactly its own tokens"
        assert bool((timestep[owner == layout.dustbin_sample_index] == 0.0).all())


class TestRejections:
    """Every one of these is a condition the sampler should have prevented. Raise, never repair."""

    def test_rejects_row_without_slack(self):
        with pytest.raises(ValueError, match="leaves no slack"):
            _layout([6, 6], 2, 12, 2, 3, 12)

    def test_rejects_pack_size_not_dividing_batch(self):
        with pytest.raises(ValueError, match="does not divide"):
            _layout([1, 2, 3], 2, 21, 2, 3, 12)

    def test_rejects_caption_longer_than_capacity(self):
        with pytest.raises(ValueError, match="longer than the llm_features text width"):
            _layout([13, 2], 1, 32, 2, 3, 12)

    def test_rejects_empty_caption(self):
        with pytest.raises(ValueError, match="needs >=1 real text token"):
            _layout([0, 2], 1, 32, 2, 3, 12)

    def test_rejects_empty_batch(self):
        with pytest.raises(ValueError, match="nothing to pack"):
            _layout([], 1, 32, 2, 3, 12)


class TestDeriveTextBudget:
    def test_one_sample_budget_reproduces_the_reserved_pad_column(self):
        assert derive_text_budget(pack_size=1, max_text_length=639, mean_text_length=383.2) == 640

    def test_budget_grows_with_pack_size_but_stays_below_k_times_max(self):
        max_len, mean_len = 639, 383.2
        for k in (2, 4, 8):
            budget = derive_text_budget(pack_size=k, max_text_length=max_len, mean_text_length=mean_len)
            assert budget > max_len, "one worst-case caption must always fit"
            assert (
                budget < k * max_len + 1
            ), "a budget of K*max is K copies of the one-sample row and saves nothing"

    def test_derived_budget_admits_an_all_mean_length_batch(self):
        """The common case the default is tuned for: K average captions in one row."""
        mean_len = 383.2
        for k in (2, 4, 8):
            budget = derive_text_budget(pack_size=k, max_text_length=639, mean_text_length=mean_len)
            lengths = [int(mean_len)] * (2 * k)
            _layout(lengths, k, budget, 4, 4, 639)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
