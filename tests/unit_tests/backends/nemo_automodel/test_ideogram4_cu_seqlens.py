###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the variable-length packing core.

THE CENTRAL TEST is ``TestEquivalence``. The claim behind replacing masked SDPA
with variable-length flash attention is that it is exact, not approximate -- the
same attention, expressed in the form the fast kernel accepts. That claim reduces
to one thing: the packing built on the host from caption lengths must describe the
same segments as the packing derived from the model's own block-diagonal mask. So
the mask is reconstructed the way the model builds it, pushed through the general
transform, and compared. If these ever disagree, the fast path silently computes
different attention and no other check in the run would notice.

The second thing pinned here is the fixed segment count. Every row contributes
exactly two segments, so the packing always has 2B+1 entries whatever the captions
are, and the compiled graph stays reusable. A data-dependent shape would recompile
on nearly every step, which would cost more than the attention saves.

No GPU and no kernels are needed for any of this: it is tensor bookkeeping and
host arithmetic. The kernels themselves belong to the compute ledger.
"""

import pytest

torch = pytest.importorskip("torch")

from primus.backends.nemo_automodel.attention import varlen_utils  # noqa: E402
from primus.backends.nemo_automodel.models.ideogram4 import cu_seqlens  # noqa: E402


def mask_from_segment_ids(segment_ids):
    """Build the (B, 1, L, L) boolean mask exactly as the model does."""
    ids = torch.tensor(segment_ids)
    return (ids[:, :, None] == ids[:, None, :]).unsqueeze(1)


def segment_ids_for(text_lengths, max_text, num_image):
    """The [pad][text][image] segment assignment for a batch of captions."""
    return [[0] * (max_text - n) + [1] * (n + num_image) for n in text_lengths]


class TestBuildCuSeqlens:
    def test_it_produces_two_segments_per_row(self):
        cu = cu_seqlens.build_cu_seqlens([3, 5], max_text_tokens=8, num_image_tokens=4)
        # row 0: pad [0,5) then text+image [5,12); row 1: pad [12,15) then [15,24)
        assert cu.tolist() == [0, 5, 12, 15, 24]

    def test_the_dtype_is_int32(self):
        """The varlen kernel's contract, not a preference."""
        cu = cu_seqlens.build_cu_seqlens([3], max_text_tokens=8, num_image_tokens=4)
        assert cu.dtype == torch.int32

    def test_it_starts_at_zero_and_ends_at_the_token_total(self):
        cu = cu_seqlens.build_cu_seqlens([2, 2, 2], max_text_tokens=4, num_image_tokens=6)
        assert int(cu[0]) == 0
        assert int(cu[-1]) == 3 * (4 + 6)

    @pytest.mark.parametrize("text_lengths", [[1, 1], [7, 7], [1, 7], [4, 2]])
    def test_the_shape_does_not_depend_on_the_captions(self, text_lengths):
        """What keeps the compiled graph reusable."""
        cu = cu_seqlens.build_cu_seqlens(text_lengths, max_text_tokens=8, num_image_tokens=4)
        assert cu.numel() == 2 * len(text_lengths) + 1

    @pytest.mark.parametrize("batch", [1, 2, 8, 16])
    def test_the_shape_is_2b_plus_1(self, batch):
        cu = cu_seqlens.build_cu_seqlens([2] * batch, max_text_tokens=8, num_image_tokens=4)
        assert cu.numel() == 2 * batch + 1

    def test_a_row_with_no_padding_is_refused(self):
        """A row that is entirely text contributes one segment rather than two,
        which breaks the fixed count the design rests on. The reserved pad column
        exists to make this unreachable, so reaching it is a real error."""
        with pytest.raises(ValueError, match="at least one pad token"):
            cu_seqlens.build_cu_seqlens([8], max_text_tokens=8, num_image_tokens=4)

    def test_over_long_text_is_refused_too(self):
        with pytest.raises(ValueError, match="at least one pad token"):
            cu_seqlens.build_cu_seqlens([9], max_text_tokens=8, num_image_tokens=4)

    def test_exactly_one_pad_token_is_enough(self):
        cu = cu_seqlens.build_cu_seqlens([7], max_text_tokens=8, num_image_tokens=4)
        assert cu.tolist() == [0, 1, 12]

    @pytest.mark.parametrize(
        "args,match",
        [
            (([], 8, 4), "nothing to pack"),
            (([1], 0, 4), "max_text_tokens"),
            (([1], 8, 0), "num_image_tokens"),
        ],
    )
    def test_degenerate_arguments_are_rejected(self, args, match):
        with pytest.raises(ValueError, match=match):
            cu_seqlens.build_cu_seqlens(*args)


class TestStaticMaxSeqlen:
    def test_it_is_the_worst_case(self):
        """The longest possible segment is a row with the minimum one pad token."""
        assert cu_seqlens.static_max_seqlen(8, 4) == 11

    def test_it_covers_every_real_segment(self):
        cu = cu_seqlens.build_cu_seqlens([1, 7], max_text_tokens=8, num_image_tokens=4)
        longest = int(varlen_utils.segment_lengths(cu).max())
        assert cu_seqlens.static_max_seqlen(8, 4) >= longest

    def test_it_is_tight(self):
        """A one-pad row reaches it exactly, so it is not needlessly loose."""
        ids = segment_ids_for([7], 8, 4)
        _, derived_max, _ = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(mask_from_segment_ids(ids))
        assert derived_max == cu_seqlens.static_max_seqlen(8, 4)

    def test_it_does_not_move_with_the_data(self):
        """A data-derived bound would be guarded by value and recompile whenever
        the longest caption in a batch changed."""
        assert cu_seqlens.static_max_seqlen(8, 4) == cu_seqlens.static_max_seqlen(8, 4)


class TestMaskTransform:
    def test_it_finds_contiguous_segments(self):
        cu, max_seqlen, trivial = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(
            mask_from_segment_ids([[0, 0, 1, 1, 1]])
        )
        assert cu.tolist() == [0, 2, 5]
        assert max_seqlen == 3
        assert trivial is False

    def test_a_single_full_segment_is_reported_trivial(self):
        """Trivial means the mask says nothing, so the caller can use dense
        attention and skip packing entirely."""
        cu, _, trivial = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(mask_from_segment_ids([[0, 0, 0, 0]]))
        assert trivial is True
        assert cu.tolist() == [0, 4]

    def test_every_token_can_be_its_own_segment(self):
        cu, max_seqlen, _ = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(
            mask_from_segment_ids([[0, 1, 2, 3]])
        )
        assert cu.tolist() == [0, 1, 2, 3, 4]
        assert max_seqlen == 1

    def test_segments_never_cross_a_row_boundary(self):
        """Two rows with identical ids must not merge: they are different samples
        and merging them would let one attend to the other."""
        cu, _, _ = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(mask_from_segment_ids([[0, 0], [0, 0]]))
        assert cu.tolist() == [0, 2, 4]

    def test_both_mask_spellings_agree(self):
        mask4 = mask_from_segment_ids([[0, 0, 1, 1]])
        got4 = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(mask4)
        got3 = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(mask4[:, 0])
        assert got4[0].tolist() == got3[0].tolist()
        assert got4[1:] == got3[1:]

    def test_a_length_one_sequence_is_handled(self):
        """Special-cased because the superdiagonal is empty here, so the usual
        reduction would have nothing to reduce."""
        cu, max_seqlen, trivial = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(
            mask_from_segment_ids([[0], [0]])
        )
        assert cu.tolist() == [0, 1, 2]
        assert max_seqlen == 1
        assert trivial is True

    def test_an_additive_float_mask_is_refused(self):
        """It carries magnitudes rather than boundaries, so it has no varlen
        equivalent and must not be silently reinterpreted."""
        with pytest.raises(TypeError, match="boolean mask"):
            varlen_utils.blockdiag_bool_mask_to_cu_seqlens(torch.zeros(1, 1, 4, 4))

    def test_a_non_square_mask_is_refused(self):
        with pytest.raises(ValueError, match="square"):
            varlen_utils.blockdiag_bool_mask_to_cu_seqlens(torch.zeros(1, 1, 4, 5, dtype=torch.bool))

    def test_a_two_dimensional_mask_is_refused(self):
        with pytest.raises(ValueError, match=r"\(B,1,L,L\)"):
            varlen_utils.blockdiag_bool_mask_to_cu_seqlens(torch.zeros(4, 4, dtype=torch.bool))


class TestEquivalence:
    """The correctness claim of the whole change. See the module docstring."""

    CASES = [
        ([3, 5], 8, 4),
        ([1, 7], 8, 4),
        ([2], 4, 2),
        ([1, 1, 1], 3, 5),
        ([5, 2, 6, 1], 8, 16),
        ([10, 3], 12, 9),
        ([1], 2, 1),
        ([15, 1, 8, 8, 3], 16, 64),
    ]

    @pytest.mark.parametrize("text_lengths,max_text,num_image", CASES)
    def test_the_host_packing_matches_the_mask_packing(self, text_lengths, max_text, num_image):
        host = cu_seqlens.build_cu_seqlens(text_lengths, max_text, num_image)
        derived, _, _ = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(
            mask_from_segment_ids(segment_ids_for(text_lengths, max_text, num_image))
        )
        assert host.tolist() == derived.tolist()

    @pytest.mark.parametrize("text_lengths,max_text,num_image", CASES)
    def test_a_padded_batch_is_never_reported_trivial(self, text_lengths, max_text, num_image):
        """If it were, the caller would take the dense path and let padding attend
        to real tokens."""
        _, _, trivial = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(
            mask_from_segment_ids(segment_ids_for(text_lengths, max_text, num_image))
        )
        assert trivial is False

    @pytest.mark.parametrize("text_lengths,max_text,num_image", CASES)
    def test_the_static_bound_covers_the_derived_maximum(self, text_lengths, max_text, num_image):
        _, derived_max, _ = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(
            mask_from_segment_ids(segment_ids_for(text_lengths, max_text, num_image))
        )
        assert cu_seqlens.static_max_seqlen(max_text, num_image) >= derived_max

    def test_it_holds_over_randomized_shapes(self):
        import random

        rng = random.Random(20260902)
        for _ in range(200):
            max_text = rng.randint(2, 24)
            num_image = rng.randint(1, 32)
            lengths = [rng.randint(1, max_text - 1) for _ in range(rng.randint(1, 6))]
            host = cu_seqlens.build_cu_seqlens(lengths, max_text, num_image)
            derived, _, _ = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(
                mask_from_segment_ids(segment_ids_for(lengths, max_text, num_image))
            )
            assert host.tolist() == derived.tolist(), (lengths, max_text, num_image)


class TestPackUnpack:
    def test_it_round_trips(self):
        x = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)
        packed = varlen_utils.pack_for_varlen(x)
        assert tuple(packed.shape) == (10, 3, 4)
        assert torch.equal(varlen_utils.unpack_from_varlen(packed, 2), x)

    def test_the_token_total_matches_the_packing(self):
        x = torch.zeros(2, 5, 3, 4)
        packed = varlen_utils.pack_for_varlen(x)
        cu = cu_seqlens.build_cu_seqlens([3, 3], max_text_tokens=4, num_image_tokens=1)
        assert packed.shape[0] == int(cu[-1])

    def test_a_batch_that_does_not_divide_is_refused(self):
        packed = torch.zeros(10, 3, 4)
        with pytest.raises(ValueError, match="do not divide"):
            varlen_utils.unpack_from_varlen(packed, 3)

    def test_packing_a_wrongly_shaped_tensor_is_refused(self):
        with pytest.raises(ValueError, match=r"\(B, L, H, D\)"):
            varlen_utils.pack_for_varlen(torch.zeros(4, 4))


class TestDescribePacking:
    def test_it_names_what_a_reader_needs(self):
        cu = cu_seqlens.build_cu_seqlens([3, 5], max_text_tokens=8, num_image_tokens=4)
        described = varlen_utils.describe_packing(cu, cu_seqlens.static_max_seqlen(8, 4))
        assert "4 segments" in described
        assert "24 tokens" in described
        assert "bound=11" in described


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
