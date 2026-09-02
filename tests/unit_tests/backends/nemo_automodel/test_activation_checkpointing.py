###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the shared activation-checkpointing helper.

WHY THIS TEST EXISTS:
  Both known bugs in this area come from Python truthiness, and neither is
  visible at runtime. ``"selective"`` is a non-empty string, so a bare
  ``if activation_checkpointing:`` runs the full-AC branch and the two settings
  become the same thing. ``"false"`` is also a non-empty string, so the same test
  turns checkpointing *on* for a run configured to have none. Nothing raises and
  the config echo looks right either way, so the only symptom is memory and step
  time -- which is to say, nobody notices unless they are already measuring.

  These tests pin the truth table directly. That is cheap here and expensive
  anywhere else, because the alternative is inferring the mode from a memory
  measurement on a multi-GPU run.

No torch and no AutoModel: the parallelizer is a recording stub.
"""

import types

import pytest

from primus.backends.nemo_automodel.distributed import activation_checkpointing as ac


@pytest.fixture
def parallelizer():
    """Records which AC path was taken, without performing any of it."""
    calls = {"selective": [], "wrapped": []}

    def apply_selective(model, layers, has_kv_sharing, **kwargs):
        calls["selective"].append((list(layers), has_kv_sharing, kwargs))

    def checkpoint_wrapper(block, **kwargs):
        calls["wrapped"].append(block)
        return ("wrapped", block)

    return types.SimpleNamespace(
        is_selective_activation_checkpointing=lambda v: v == "selective",
        apply_selective_checkpointing_to_layers=apply_selective,
        checkpoint_wrapper=checkpoint_wrapper,
        CheckpointImpl=types.SimpleNamespace(NO_REENTRANT="no_reentrant"),
        calls=calls,
    )


def model_with(**attrs):
    return types.SimpleNamespace(**attrs)


class TestNormalize:
    @pytest.mark.parametrize("raw", ["false", "False", "FALSE", "0", "off", "no", "none", "", "  ", " off "])
    def test_false_like_strings_become_false(self, raw):
        assert ac.normalize(raw) is False

    @pytest.mark.parametrize("raw", ["full", "selective", True])
    def test_meaningful_values_survive(self, raw):
        """'selective' in particular must not be coerced to a bool, or the caller
        can no longer tell it apart from 'full'."""
        assert ac.normalize(raw) == raw

    def test_false_stays_false(self):
        assert ac.normalize(False) is False


class TestApply:
    def test_selective_uses_the_shared_machinery(self, parallelizer):
        model = model_with(blocks=[1, 2, 3])
        mode, count = ac.apply(parallelizer, model, ("blocks",), "selective", log_prefix="[t]")
        assert (mode, count) == (ac.MODE_SELECTIVE, 3)
        assert len(parallelizer.calls["selective"]) == 1
        assert not parallelizer.calls["wrapped"], "selective must not also wrap blocks"

    def test_full_wraps_every_block_in_place(self, parallelizer):
        blocks = [1, 2, 3]
        model = model_with(blocks=blocks)
        mode, count = ac.apply(parallelizer, model, ("blocks",), "full", log_prefix="[t]")
        assert (mode, count) == (ac.MODE_FULL, 3)
        # In place: these are ModuleLists, so the wrapper has to take the block's
        # place or nothing is checkpointed.
        assert blocks == [("wrapped", 1), ("wrapped", 2), ("wrapped", 3)]

    def test_true_means_full(self, parallelizer):
        mode, count = ac.apply(parallelizer, model_with(blocks=[1]), ("blocks",), True, log_prefix="[t]")
        assert (mode, count) == (ac.MODE_FULL, 1)

    @pytest.mark.parametrize("raw", ["false", "off", "no", "none", "", False])
    def test_false_like_does_nothing(self, parallelizer, raw):
        blocks = [1, 2, 3]
        mode, count = ac.apply(parallelizer, model_with(blocks=blocks), ("blocks",), raw, log_prefix="[t]")
        assert (mode, count) == (ac.MODE_OFF, 0)
        assert blocks == [1, 2, 3]
        assert not parallelizer.calls["selective"] and not parallelizer.calls["wrapped"]

    def test_several_block_lists_are_all_wrapped(self, parallelizer):
        """A model may split its blocks across lists -- dual-stream and
        single-stream, say -- and all of them must be covered."""
        dual, single = [1, 2], [3, 4, 5]
        model = model_with(transformer_blocks=dual, single_transformer_blocks=single)
        mode, count = ac.apply(
            parallelizer,
            model,
            ("transformer_blocks", "single_transformer_blocks"),
            "full",
            log_prefix="[t]",
        )
        assert (mode, count) == (ac.MODE_FULL, 5)
        assert dual == [("wrapped", 1), ("wrapped", 2)]
        assert single == [("wrapped", 3), ("wrapped", 4), ("wrapped", 5)]

    def test_selective_flattens_across_lists_in_one_call(self, parallelizer):
        """The helper replaces each block with an identity, so one call covering
        both lists is correct -- and two calls would not be."""
        model = model_with(transformer_blocks=[1, 2], single_transformer_blocks=[3])
        ac.apply(
            parallelizer,
            model,
            ("transformer_blocks", "single_transformer_blocks"),
            "selective",
            log_prefix="[t]",
        )
        assert len(parallelizer.calls["selective"]) == 1
        layers, has_kv_sharing, _kwargs = parallelizer.calls["selective"][0]
        assert layers == [1, 2, 3], "order must be stable so the logged count means something"
        assert has_kv_sharing is False, "diffusion transformers have no KV cache"

    def test_missing_block_attrs_warns_rather_than_raising(self, parallelizer):
        mode, count = ac.apply(
            parallelizer, model_with(something_else=[1]), ("blocks",), "full", log_prefix="[t]"
        )
        assert (mode, count) == (ac.MODE_NO_BLOCKS, 0)

    def test_absent_attrs_are_skipped_not_fatal(self, parallelizer):
        """A model with only one of the two lists is normal, not an error."""
        model = model_with(transformer_blocks=[1, 2])
        mode, count = ac.apply(
            parallelizer,
            model,
            ("transformer_blocks", "single_transformer_blocks"),
            "full",
            log_prefix="[t]",
        )
        assert (mode, count) == (ac.MODE_FULL, 2)

    def test_enable_compile_is_forwarded(self, parallelizer):
        ac.apply(
            parallelizer,
            model_with(blocks=[1]),
            ("blocks",),
            "selective",
            enable_compile=True,
            log_prefix="[t]",
        )
        _layers, _kv, kwargs = parallelizer.calls["selective"][0]
        assert kwargs["enable_compile"] is True


class TestStride:
    """The fourth setting between selective and full: which blocks get wrapped.

    The reason it exists is that a configuration can be slightly short of memory
    and find that both full and selective AC hand back several times what was
    needed, charging recompute for all of it. Wrapping k of N blocks sheds roughly
    k/N of the peak and costs roughly k/N of the recompute, so the tests below pin
    exactly which indices are chosen -- that count is the whole interface.
    """

    @staticmethod
    def wrapped_indices(blocks):
        return [i for i, b in enumerate(blocks) if isinstance(b, tuple)]

    @pytest.mark.parametrize(
        "stride,expected",
        [
            (2, [0, 2, 4, 6, 8]),
            (3, [0, 3, 6, 9]),
            (4, [0, 4, 8]),
            (10, [0]),
            (20, [0]),  # a stride past the end still wraps the first block
        ],
    )
    def test_it_wraps_every_nth_block(self, parallelizer, stride, expected):
        blocks = list(range(10))
        model = model_with(blocks=blocks)
        mode, count = ac.apply(parallelizer, model, ("blocks",), True, stride=stride, log_prefix="[t]")
        assert mode == ac.MODE_FULL
        assert self.wrapped_indices(blocks) == expected
        assert count == len(expected)

    @pytest.mark.parametrize("stride", [0, 1])
    def test_a_stride_of_zero_or_one_is_plain_full_ac(self, parallelizer, stride):
        blocks = list(range(6))
        _mode, count = ac.apply(
            parallelizer,
            model_with(blocks=blocks),
            ("blocks",),
            True,
            stride=stride,
            log_prefix="[t]",
        )
        assert count == 6
        assert self.wrapped_indices(blocks) == list(range(6))

    def test_the_default_is_plain_full_ac(self, parallelizer):
        """So a caller that does not know about the stride is unaffected."""
        blocks = list(range(5))
        _mode, count = ac.apply(parallelizer, model_with(blocks=blocks), ("blocks",), True, log_prefix="[t]")
        assert count == 5

    def test_the_position_runs_across_block_lists(self, parallelizer):
        """A model can keep its blocks in more than one list. A stride that
        restarted per list would wrap the first block of each, giving an uneven
        spread and a total that depended on how the lists happened to be split."""
        first = list(range(3))
        second = list(range(3))
        model = model_with(dual=first, single=second)
        _mode, count = ac.apply(parallelizer, model, ("dual", "single"), True, stride=2, log_prefix="[t]")
        # Positions 0..5 across both lists, so 0, 2, 4 -> first[0], first[2], second[1].
        assert self.wrapped_indices(first) == [0, 2]
        assert self.wrapped_indices(second) == [1]
        assert count == 3

    def test_it_does_nothing_when_ac_is_off(self, parallelizer):
        blocks = list(range(4))
        mode, count = ac.apply(
            parallelizer,
            model_with(blocks=blocks),
            ("blocks",),
            False,
            stride=2,
            log_prefix="[t]",
        )
        assert (mode, count) == (ac.MODE_OFF, 0)
        assert self.wrapped_indices(blocks) == []

    def test_combining_it_with_selective_is_refused(self, parallelizer):
        """Refused rather than ignored: a caller that asked for both wanted less
        recompute than selective AC gives, and quietly handing them plain selective
        AC would look like it worked."""
        with pytest.raises(ValueError, match="different granularities"):
            ac.apply(
                parallelizer,
                model_with(blocks=[1, 2, 3]),
                ("blocks",),
                "selective",
                stride=2,
                log_prefix="[t]",
            )

    def test_a_stride_of_one_with_selective_is_fine(self, parallelizer):
        """Since it means the same thing as not asking for a stride at all."""
        mode, _count = ac.apply(
            parallelizer,
            model_with(blocks=[1, 2, 3]),
            ("blocks",),
            "selective",
            stride=1,
            log_prefix="[t]",
        )
        assert mode == ac.MODE_SELECTIVE

    def test_a_negative_stride_is_refused(self, parallelizer):
        with pytest.raises(ValueError, match=">= 1"):
            ac.apply(
                parallelizer,
                model_with(blocks=[1, 2]),
                ("blocks",),
                True,
                stride=-2,
                log_prefix="[t]",
            )

    def test_the_count_is_reported_so_a_run_can_be_read_back(self, parallelizer, caplog):
        """The wrapped count is what converts a memory target into a setting, so it
        has to be in the log of the run that used it."""
        with caplog.at_level("INFO"):
            ac.apply(
                parallelizer,
                model_with(blocks=list(range(9))),
                ("blocks",),
                True,
                stride=3,
                log_prefix="[t]",
            )
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "3 of 9" in message
        assert "every 3" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
