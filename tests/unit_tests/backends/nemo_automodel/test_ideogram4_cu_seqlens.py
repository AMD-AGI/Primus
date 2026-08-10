###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 host-side var-len packing (``build_cu_seqlens``).

WHY THIS TEST EXISTS:
  There are now TWO producers of the same ``cu_seqlens``:

    1. ``build_cu_seqlens(text_lengths, ...)``           -- host-side, from Python ints
    2. ``blockdiag_bool_mask_to_cu_seqlens(mask)``       -- device-side, from mask values

  Producer 1 is what training uses (it keeps the compiled graph break-free); producer 2 is
  the original, and remains the reference. If they ever disagree, var-len flash attends
  across what should be separate segments -- padding leaks into real tokens and training
  corrupts with NO error raised anywhere. Exact equality is therefore the assertion, not
  closeness.

  The tests also pin the two properties the compiled graph depends on:
    * the segment count is exactly ``2*B`` for ANY caption lengths, so ``cu_seqlens`` has a
      fixed shape and does not trigger a recompile per length pattern;
    * a row with no padding (``offset == 0``) is rejected loudly, since a zero-length
      segment is not representable for var-len flash.

Only ``torch`` is required for the core parity test. The ``_prepare_ids`` cross-check needs
``nemo_automodel`` (the adapter class is built lazily against its ``ModelAdapter`` base) and
skips when the submodule is not importable. Note that it therefore runs in TWO different
environments -- with diffusers present (in-container) and without (lint/CI) -- which is why the
padding-segment id is imported from the adapter instead of hardcoded.
"""

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - CPU-less lint environments
    pytest.skip("Ideogram-4 cu_seqlens tests require torch", allow_module_level=True)

from primus.backends.nemo_automodel.ideogram4_adapter import (
    SEQUENCE_PADDING_INDICATOR,
    build_cu_seqlens,
)
from primus.backends.nemo_automodel.ideogram4_varlen_attn import (
    blockdiag_bool_mask_to_cu_seqlens,
)

# Taken from the adapter rather than hardcoded, because the adapter resolves it from diffusers
# when diffusers is importable and falls back to a literal otherwise -- and the two disagree
# (0.39.0 pads with -1). The VALUE is irrelevant to everything here: the model's mask is
# ``seg_i == seg_j``, so only which positions share an id matters, and the adapter writes the
# valid region as a literal 1.
VALID_SEGMENT = 1


def _segment_ids(text_lengths, max_text_tokens, num_image_tokens):
    """``[left-pad][text][image]`` segment ids -- the layout ``_prepare_ids`` produces.

    ``PADDING(0)`` over the left-pad region, ``1`` over the whole ``[text][image]`` region,
    which is what makes the model's ``(seg_i == seg_j)`` mask block-diagonal per row.
    """
    total = max_text_tokens + num_image_tokens
    seg = torch.full((len(text_lengths), total), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    for b, num_text in enumerate(text_lengths):
        offset = max_text_tokens - int(num_text)
        seg[b, offset:] = VALID_SEGMENT
    return seg


def _bool_mask(segment_ids):
    """The dense ``(B,1,L,L)`` mask the diffusers block materializes from segment ids."""
    return (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)


# (label, text_lengths, max_text_tokens, num_image_tokens)
# max_text_tokens already INCLUDES the adapter's reserved pad column, so every case leaves
# at least one pad token per row.
LENGTH_PATTERNS = [
    ("min_padding_one_slot", [4, 4], 5, 3),
    ("maximally_ragged", [1, 2, 3, 4], 5, 3),
    ("all_equal", [3, 3, 3, 3], 5, 3),
    ("single_row", [2], 5, 7),
    ("mixed_min_and_slack", [4, 1, 4, 2], 5, 3),
    ("wide_image_region", [7, 13, 2], 16, 64),
    ("batch_of_eight", [1, 3, 5, 7, 9, 11, 13, 15], 16, 32),
]


class TestBuildCuSeqlensParity:
    """``build_cu_seqlens`` must agree EXACTLY with the mask-derived reference."""

    @pytest.mark.parametrize(
        "text_lengths,max_text_tokens,num_image_tokens",
        [p[1:] for p in LENGTH_PATTERNS],
        ids=[p[0] for p in LENGTH_PATTERNS],
    )
    def test_matches_mask_derived_reference(self, text_lengths, max_text_tokens, num_image_tokens):
        seg = _segment_ids(text_lengths, max_text_tokens, num_image_tokens)
        expected, expected_max_seqlen, is_trivial = blockdiag_bool_mask_to_cu_seqlens(_bool_mask(seg))

        actual = build_cu_seqlens(text_lengths, max_text_tokens, num_image_tokens)

        assert not is_trivial, (
            "every row should have padding in these fixtures, so the mask must NOT be "
            "trivial -- if it is, the fixture no longer exercises the var-len path"
        )
        assert actual.dtype == torch.int32, "flash_attn_varlen_func requires int32 cu_seqlens"
        assert torch.equal(actual, expected), (
            "host-built cu_seqlens disagrees with the mask-derived reference for "
            f"text_lengths={text_lengths}, max_text_tokens={max_text_tokens}: "
            f"got {actual.tolist()} vs {expected.tolist()}. A mismatch here means var-len "
            "flash attends across segment boundaries and training corrupts silently. Check "
            "that build_cu_seqlens' offset arithmetic still mirrors _prepare_ids."
        )
        # The static bound the adapter passes must never under-estimate the real longest
        # segment, or flash gets a max_seqlen smaller than a sequence it has to schedule.
        assert max_text_tokens + num_image_tokens >= expected_max_seqlen

    @pytest.mark.parametrize(
        "text_lengths,max_text_tokens,num_image_tokens",
        [p[1:] for p in LENGTH_PATTERNS],
        ids=[p[0] for p in LENGTH_PATTERNS],
    )
    def test_shape_is_two_segments_per_row(self, text_lengths, max_text_tokens, num_image_tokens):
        """Fixed ``2*B+1`` length is what keeps the compiled graph from recompiling."""
        cu = build_cu_seqlens(text_lengths, max_text_tokens, num_image_tokens)
        assert cu.numel() == 2 * len(text_lengths) + 1, (
            f"expected 2*B+1={2 * len(text_lengths) + 1} entries for B={len(text_lengths)}, got "
            f"{cu.numel()}. A data-dependent length changes this tensor's shape between "
            "batches and forces a torch.compile recompile per length pattern."
        )

    def test_shape_is_independent_of_caption_lengths(self):
        """Same batch size, wildly different captions -> identical cu_seqlens shape."""
        shapes = {
            build_cu_seqlens(lengths, 16, 32).shape
            for lengths in ([1, 1, 1, 1], [15, 15, 15, 15], [1, 15, 8, 3], [7, 2, 15, 11])
        }
        assert len(shapes) == 1, f"cu_seqlens shape varied with caption lengths: {shapes}"

    @pytest.mark.parametrize(
        "text_lengths,max_text_tokens,num_image_tokens",
        [p[1:] for p in LENGTH_PATTERNS],
        ids=[p[0] for p in LENGTH_PATTERNS],
    )
    def test_contract_boundaries(self, text_lengths, max_text_tokens, num_image_tokens):
        """Starts at 0, ends at B*S, non-decreasing, and no empty segments."""
        seq_len = max_text_tokens + num_image_tokens
        cu = build_cu_seqlens(text_lengths, max_text_tokens, num_image_tokens)

        assert int(cu[0]) == 0
        assert int(cu[-1]) == len(text_lengths) * seq_len
        diffs = cu[1:] - cu[:-1]
        assert bool((diffs > 0).all()), (
            f"zero-length or negative segment in {cu.tolist()}; var-len flash cannot "
            "represent an empty sequence"
        )

    def test_segment_lengths_match_the_layout(self):
        """Per row: the pad block is ``offset`` long, the text+image block the remainder."""
        text_lengths, max_text_tokens, num_image_tokens = [2, 4, 1], 6, 5
        cu = build_cu_seqlens(text_lengths, max_text_tokens, num_image_tokens)
        diffs = (cu[1:] - cu[:-1]).tolist()

        expected = []
        for num_text in text_lengths:
            offset = max_text_tokens - num_text
            expected += [offset, num_text + num_image_tokens]
        assert diffs == expected, f"segment lengths {diffs} do not match the layout {expected}"

    def test_rejects_row_without_padding(self):
        """``offset == 0`` must fail loudly -- the reserved pad column prevents it."""
        with pytest.raises(ValueError, match="leaving no padding"):
            build_cu_seqlens([5, 3], max_text_tokens=5, num_image_tokens=4)

    def test_accepts_torch_scalar_lengths(self):
        """``text_lengths`` may arrive as tensor elements; ints must be coerced."""
        as_ints = build_cu_seqlens([2, 3], 5, 4)
        as_tensors = build_cu_seqlens(list(torch.tensor([2, 3])), 5, 4)
        assert torch.equal(as_ints, as_tensors)


class TestPrepareIdsAgreement:
    """The adapter's own ``segment_ids`` must match the layout this test assumes."""

    def test_prepare_ids_segment_ids_match_local_layout(self):
        pytest.importorskip(
            "nemo_automodel",
            reason="Ideogram-4 adapter class is built against nemo_automodel's ModelAdapter base",
        )
        from primus.backends.nemo_automodel.ideogram4_adapter import (
            get_ideogram4_adapter_class,
        )

        adapter_cls = get_ideogram4_adapter_class()
        text_lengths, grid_h, grid_w, max_text_tokens = [2, 4, 1], 2, 3, 6
        num_image_tokens = grid_h * grid_w

        _, segment_ids, _ = adapter_cls._prepare_ids(
            text_lengths, grid_h, grid_w, max_text_tokens, torch.device("cpu")
        )

        assert torch.equal(
            segment_ids, _segment_ids(text_lengths, max_text_tokens, num_image_tokens)
        ), "this test file's segment-id layout has drifted from the adapter's _prepare_ids"

        # Closing the loop: the mask the model builds from the adapter's real segment_ids
        # yields the same packing the adapter hands to the processor.
        expected, _, _ = blockdiag_bool_mask_to_cu_seqlens(_bool_mask(segment_ids))
        actual = build_cu_seqlens(text_lengths, max_text_tokens, num_image_tokens)
        assert torch.equal(actual, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
