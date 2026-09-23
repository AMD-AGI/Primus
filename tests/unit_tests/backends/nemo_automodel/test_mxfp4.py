###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the MXFP4 linear path.

WHAT THESE ARE DEFENDING:
  Two failure modes here are silent, and both are pinned below.

  The first is precedence. MXFP4 and FP8 are both requested by env var and both
  claim AutoModel's swap symbol. If the order inverted, asking for four bits
  would give eight and the run would look completely normal, so
  ``test_asking_for_both_gives_mxfp4`` is load-bearing rather than decorative.

  The second is the token padding. AITER's FP4 GEMM returns wrong numbers -- it
  does not raise -- when the contraction dimension is not a multiple of 256, and
  the only GEMM of a Linear that contracts over tokens is the weight gradient. So
  the arithmetic gets a truth table, including the short-text-sequence case that
  motivates it.

  The band arithmetic is third: "first N and last M blocks" is off-by-one bait,
  and either mistake silently changes which layers run in which precision.

Most of this deliberately avoids torch, so it runs anywhere. Float4Linear itself
needs both torch and primus_turbo and belongs to the compute ledger.
"""

import pytest

from primus.backends.nemo_automodel.quantization import _common, _fp4_common, fp8_linear

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="needs torch")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every knob unset, so a default under test is really the default."""
    for var in (
        "PRIMUS_TURBO_FP4",
        "PRIMUS_TURBO_FP8",
        "PRIMUS_TURBO_FP4_SR",
        "PRIMUS_TURBO_FP4_PRESHUFFLE",
        "PRIMUS_TURBO_FP4_BACKWARD",
        "PRIMUS_TURBO_FP4_KEEP_SENSITIVE",
        "PRIMUS_TURBO_FP4_SENSITIVE_LAYERS",
        "PRIMUS_TURBO_FP4_SENSITIVE_START",
        "PRIMUS_TURBO_FP4_SENSITIVE_END",
        "PRIMUS_TURBO_FP4_SENSITIVE_PRECISION",
    ):
        monkeypatch.delenv(var, raising=False)


class TestPrecedence:
    def test_mxfp4_outranks_fp8(self):
        by_name = {e.name: e for e in _common.registered_backends()}
        assert by_name[_fp4_common.BACKEND_NAME].precedence > by_name[fp8_linear.BACKEND_NAME].precedence

    def test_asking_for_both_gives_mxfp4(self, monkeypatch):
        """The silent one: if this inverted, a four-bit request would train in
        eight bits with nothing in the logs to say so."""
        monkeypatch.setenv("PRIMUS_TURBO_FP4", "1")
        monkeypatch.setenv("PRIMUS_TURBO_FP8", "1")
        assert _common.active_backend().name == _fp4_common.BACKEND_NAME
        assert _common.is_active(fp8_linear.BACKEND_NAME) is False

    def test_fp8_alone_still_wins_when_fp4_is_unset(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_FP8", "1")
        assert _common.active_backend().name == fp8_linear.BACKEND_NAME

    def test_the_losing_fp8_request_is_announced(self, monkeypatch, caplog):
        monkeypatch.setenv("PRIMUS_TURBO_FP4", "1")
        monkeypatch.setenv("PRIMUS_TURBO_FP8", "1")
        with caplog.at_level("WARNING"):
            _common.active_backend()
        assert any("takes precedence" in r.message for r in caplog.records)

    def test_the_selector_is_usable_without_torch(self):
        """The patch condition consults the selector, and a condition should be
        answerable without importing a kernel stack."""
        import subprocess
        import sys

        code = (
            "import sys; sys.modules['torch']=None\n"
            "from primus.backends.nemo_automodel.quantization import _fp4_common\n"
            "print(_fp4_common.BACKEND_NAME)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


class TestKnobDefaults:
    def test_the_swap_is_off_by_default(self):
        assert _fp4_common.is_enabled() is False

    def test_backward_defaults_to_pure_mxfp4(self):
        assert _fp4_common.backward_precision() == "mxfp4"

    def test_gradient_sr_defaults_off(self):
        assert _fp4_common.gradient_sr_enabled() is False

    def test_preshuffle_defaults_on(self):
        """Turbo dispatches to AITER when preshuffle is on, so this also picks the
        backend; on matches Turbo's own tuned path."""
        assert _fp4_common.preshuffle_enabled() is True

    def test_conditioning_layers_are_kept_in_bf16_by_default(self):
        assert _fp4_common.keep_sensitive_bf16() is True

    def test_the_band_is_off_by_default(self):
        assert _fp4_common.sensitive_band_enabled() is False


class TestKnobValidation:
    """An unrecognised value raises rather than falling back, because training in
    a different precision than the one requested leaves no trace."""

    def test_an_invalid_backward_precision_raises(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_FP4_BACKWARD", "fp4")
        with pytest.raises(ValueError, match="PRIMUS_TURBO_FP4_BACKWARD"):
            _fp4_common.backward_precision()

    def test_the_retired_pad_mode_is_rejected(self, monkeypatch):
        """Padding is unconditional now, so a config still asking for the old
        opt-in mode should fail loudly instead of appearing to be honoured."""
        monkeypatch.setenv("PRIMUS_TURBO_FP4_BACKWARD", "mxfp4_pad")
        with pytest.raises(ValueError, match="expected one of"):
            _fp4_common.backward_precision()

    def test_backward_precision_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_FP4_BACKWARD", "FP8")
        assert _fp4_common.backward_precision() == "fp8"

    def test_an_invalid_band_precision_raises(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_FP4_SENSITIVE_PRECISION", "fp16")
        with pytest.raises(ValueError, match="PRIMUS_TURBO_FP4_SENSITIVE_PRECISION"):
            _fp4_common.sensitive_precision()


class TestBlockIndex:
    @pytest.mark.parametrize(
        "fqn,expected",
        [
            ("blocks.7.attn1.to_q", ("blocks", 7)),
            ("transformer_blocks.0.ff.net.0.proj", ("transformer_blocks", 0)),
            ("single_transformer_blocks.37.proj_out", ("single_transformer_blocks", 37)),
            ("blocks.12", ("blocks", 12)),
            ("proj_out", (None, None)),
            ("condition_embedder.time_proj", (None, None)),
        ],
    )
    def test_it_parses_block_lists_and_indices(self, fqn, expected):
        assert _fp4_common.block_index(fqn) == expected


class TestSensitiveBand:
    COUNTS = {"blocks": 40}

    @pytest.mark.parametrize(
        "idx,in_band",
        [(0, True), (1, True), (2, False), (20, False), (31, False), (32, True), (39, True)],
    )
    def test_first_two_and_last_eight_are_in_the_band(self, monkeypatch, idx, in_band):
        monkeypatch.setenv("PRIMUS_TURBO_FP4_SENSITIVE_START", "2")
        monkeypatch.setenv("PRIMUS_TURBO_FP4_SENSITIVE_END", "8")
        assert _fp4_common.is_sensitive_block(f"blocks.{idx}.attn.to_q", self.COUNTS) is in_band

    def test_a_non_block_layer_is_never_in_the_band(self):
        assert _fp4_common.is_sensitive_block("proj_out", self.COUNTS) is False

    def test_an_unknown_block_list_is_not_in_the_band(self):
        assert _fp4_common.is_sensitive_block("other_blocks.0.x", self.COUNTS) is False

    def test_each_block_list_is_banded_independently(self, monkeypatch):
        """A model with a dual-stream and a single-stream list has two different
        last indices, so a single global count would band the wrong layers."""
        monkeypatch.setenv("PRIMUS_TURBO_FP4_SENSITIVE_START", "1")
        monkeypatch.setenv("PRIMUS_TURBO_FP4_SENSITIVE_END", "1")
        counts = {"transformer_blocks": 19, "single_transformer_blocks": 38}
        assert _fp4_common.is_sensitive_block("transformer_blocks.18.x", counts) is True
        assert _fp4_common.is_sensitive_block("single_transformer_blocks.18.x", counts) is False
        assert _fp4_common.is_sensitive_block("single_transformer_blocks.37.x", counts) is True

    def test_an_oversized_band_covers_everything(self, monkeypatch):
        """Overlapping start and end must not cancel out into an empty band."""
        monkeypatch.setenv("PRIMUS_TURBO_FP4_SENSITIVE_START", "50")
        monkeypatch.setenv("PRIMUS_TURBO_FP4_SENSITIVE_END", "50")
        assert all(_fp4_common.is_sensitive_block(f"blocks.{i}.x", self.COUNTS) for i in range(40))


class TestTokenPadding:
    """AITER's FP4 GEMM returns wrong numbers, without raising, when the
    contraction dim is not a multiple of 256. Only the weight gradient contracts
    over tokens, and the token count is a runtime property."""

    def test_the_contraction_multiple_matches_the_kernel(self):
        assert _fp4_common.AITER_K_MULTIPLE == 256

    def test_the_feature_alignment_is_separate_and_smaller(self):
        """128 is enforced at swap time on in/out_features; 256 is a runtime
        property of the token count. Conflating them would under-pad."""
        assert _fp4_common.FP4_ALIGN == 128
        assert _fp4_common.FP4_ALIGN < _fp4_common.AITER_K_MULTIPLE

    @pytest.mark.parametrize(
        "tokens,padded",
        [
            (256, 256),
            (512, 512),
            (1, 256),
            (255, 256),
            (257, 512),
            # A short text sequence times batch, which is the shape that lands
            # off the multiple routinely and motivates the whole fix.
            (1808, 2048),
            (61200, 61440),
        ],
    )
    def test_the_padded_count_is_the_next_multiple(self, tokens, padded):
        assert _fp4_common.pad_multiple(tokens) == padded

    def test_an_aligned_count_is_left_alone(self):
        """So the aligned case pays nothing for a fix it does not need."""
        assert _fp4_common.pad_multiple(2048) == 2048


@requires_torch
class TestCountBlocks:
    def test_it_finds_block_counts_by_walking_linears(self):
        """Counted from the module tree rather than from an attribute name, so it
        works on a model whose block list is named something unexpected."""
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(5)])
        assert _fp4_common.count_blocks(model) == {"blocks": 5}

    def test_it_finds_two_block_lists_separately(self):
        model = torch.nn.Module()
        model.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])
        model.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(7)])
        assert _fp4_common.count_blocks(model) == {
            "transformer_blocks": 3,
            "single_transformer_blocks": 7,
        }

    def test_a_model_with_no_blocks_gives_no_counts(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8))
        assert _fp4_common.count_blocks(model) == {}


class TestPatchRegistration:
    def _patches(self):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        return {p.id: p for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")}

    def test_the_patch_is_registered(self):
        assert "nemo_automodel.quantization.mxfp4_linear" in self._patches()

    def test_it_is_gated_off_by_default(self):
        assert self._patches()["nemo_automodel.quantization.mxfp4_linear"].condition(None) is False

    def test_it_turns_on_when_requested(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_TURBO_FP4", "1")
        assert self._patches()["nemo_automodel.quantization.mxfp4_linear"].condition(None) is True

    def test_only_one_quantization_patch_is_ever_active(self, monkeypatch):
        """They rebind the same symbol, so two active conditions would mean the
        second silently overwrote the first."""
        monkeypatch.setenv("PRIMUS_TURBO_FP4", "1")
        monkeypatch.setenv("PRIMUS_TURBO_FP8", "1")
        by_id = self._patches()
        active = [
            pid
            for pid in ("nemo_automodel.quantization.mxfp4_linear", "nemo_automodel.quantization.fp8_linear")
            if by_id[pid].condition(None)
        ]
        assert active == ["nemo_automodel.quantization.mxfp4_linear"]

    def test_the_swap_precedes_the_model_strategies(self):
        by_id = self._patches()
        for strategy in (
            "nemo_automodel.models.wan.parallelize",
            "nemo_automodel.models.flux.parallelize",
        ):
            assert by_id["nemo_automodel.quantization.mxfp4_linear"].priority < by_id[strategy].priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
