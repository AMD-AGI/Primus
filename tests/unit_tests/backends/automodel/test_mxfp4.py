###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the MXFP4 linear path.

WHAT THESE ARE DEFENDING:
  Two failure modes here are silent, and both are pinned below.

  The first is precedence. MXFP4 and FP8 are both requested by a setting and both
  claim AutoModel's swap symbol. If the order inverted, asking for four bits
  would give eight and the run would look completely normal, so
  ``test_asking_for_both_gives_mxfp4`` is load-bearing rather than decorative.

  The second is the token padding. AITER's FP4 GEMM needs the contraction
  dimension to be a multiple of 256 (current builds refuse otherwise; older ones
  returned wrong numbers without raising), and the only GEMM of a Linear that
  contracts over tokens is the weight gradient. So the arithmetic gets a truth
  table, including the short-text-sequence case that motivates it.

  The band arithmetic is third: "first N and last M blocks" is off-by-one bait,
  and either mistake silently changes which layers run in which precision.

Most of this deliberately avoids torch, so it runs anywhere. Float4Linear itself
needs both torch and primus_turbo and is exercised on hardware.
"""

import types

import pytest

from primus.backends.automodel.quantization import _common, _fp4_common, fp8_linear

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="needs torch")


class TestPrecedence:
    def test_mxfp4_outranks_fp8(self):
        by_name = {e.name: e for e in _common.registered_backends()}
        assert by_name[_fp4_common.BACKEND_NAME].precedence > by_name[fp8_linear.BACKEND_NAME].precedence

    def test_asking_for_both_gives_mxfp4(self, set_option):
        """The silent one: if this inverted, a four-bit request would train in
        eight bits with nothing in the logs to say so."""
        set_option("primus_turbo.mxfp4_linear", True)
        set_option("primus_turbo.fp8_linear", True)
        assert _common.active_backend().name == _fp4_common.BACKEND_NAME
        assert _common.is_active(fp8_linear.BACKEND_NAME) is False

    def test_fp8_alone_still_wins_when_fp4_is_unset(self, set_option):
        set_option("primus_turbo.fp8_linear", True)
        assert _common.active_backend().name == fp8_linear.BACKEND_NAME

    def test_the_losing_fp8_request_is_announced(self, set_option, caplog, monkeypatch):
        # Announced once per process, so an earlier test may already have used it up.
        monkeypatch.setattr(_common, "_warned_conflicts", set())
        set_option("primus_turbo.mxfp4_linear", True)
        set_option("primus_turbo.fp8_linear", True)
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
            "from primus.backends.automodel.quantization import _fp4_common\n"
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

    def test_an_invalid_backward_precision_raises(self, set_option):
        set_option("primus_turbo.mxfp4_backward", "fp4")
        with pytest.raises(ValueError, match="primus_turbo.mxfp4_backward"):
            _fp4_common.backward_precision()

    def test_the_retired_pad_mode_is_rejected(self, set_option):
        """Padding is unconditional now, so a config still asking for the old
        opt-in mode should fail loudly instead of appearing to be honoured."""
        set_option("primus_turbo.mxfp4_backward", "mxfp4_pad")
        with pytest.raises(ValueError, match="expected one of"):
            _fp4_common.backward_precision()

    def test_backward_precision_is_case_insensitive(self, set_option):
        set_option("primus_turbo.mxfp4_backward", "FP8")
        assert _fp4_common.backward_precision() == "fp8"

    def test_an_invalid_band_precision_raises(self, set_option):
        set_option("primus_turbo.mxfp4_sensitive_precision", "fp16")
        with pytest.raises(ValueError, match="primus_turbo.mxfp4_sensitive_precision"):
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
    def test_first_two_and_last_eight_are_in_the_band(self, idx, in_band):
        assert _fp4_common.is_sensitive_block(f"blocks.{idx}.attn.to_q", self.COUNTS, 2, 8) is in_band

    def test_the_defaults_are_two_and_eight(self):
        assert (_fp4_common.sensitive_start(), _fp4_common.sensitive_end()) == (2, 8)

    def test_a_non_block_layer_is_never_in_the_band(self):
        assert _fp4_common.is_sensitive_block("proj_out", self.COUNTS, 2, 8) is False

    def test_an_unknown_block_list_is_not_in_the_band(self):
        assert _fp4_common.is_sensitive_block("other_blocks.0.x", self.COUNTS, 2, 8) is False

    def test_no_block_counts_means_no_band(self):
        """What the caller passes when the band is off."""
        assert _fp4_common.is_sensitive_block("blocks.0.x", {}, 2, 8) is False

    def test_each_block_list_is_banded_independently(self):
        """A model with a dual-stream and a single-stream list has two different
        last indices, so a single global count would band the wrong layers."""
        counts = {"transformer_blocks": 19, "single_transformer_blocks": 38}
        assert _fp4_common.is_sensitive_block("transformer_blocks.18.x", counts, 1, 1) is True
        assert _fp4_common.is_sensitive_block("single_transformer_blocks.18.x", counts, 1, 1) is False
        assert _fp4_common.is_sensitive_block("single_transformer_blocks.37.x", counts, 1, 1) is True

    def test_an_oversized_band_covers_everything(self):
        """Overlapping start and end must not cancel out into an empty band."""
        assert all(_fp4_common.is_sensitive_block(f"blocks.{i}.x", self.COUNTS, 50, 50) for i in range(40))

    def test_a_band_covering_a_whole_list_is_reported(self):
        counts = {"transformer_blocks": 19, "single_transformer_blocks": 38}
        assert _fp4_common.lists_covered_by_band(counts, 10, 10) == {"transformer_blocks": 19}
        assert _fp4_common.lists_covered_by_band(counts, 2, 8) == {}

    @pytest.mark.parametrize(
        "setting", ["primus_turbo.mxfp4_sensitive_start", "primus_turbo.mxfp4_sensitive_end"]
    )
    def test_a_negative_band_size_is_refused(self, set_option, setting):
        """A negative size would silently shrink or empty the band."""
        set_option(setting, -1)
        with pytest.raises(ValueError, match=setting):
            _fp4_common.sensitive_start() if setting.endswith("start") else _fp4_common.sensitive_end()


class TestFeatureAlignment:
    """The forward contracts over in_features and the dgrad over out_features,
    so the multiple is the GEMM backend's."""

    def test_aiter_needs_256_when_preshuffle_is_on(self):
        assert _fp4_common.preshuffle_enabled() is True
        assert _fp4_common.feature_multiple() == 256

    def test_hipblaslt_needs_128_when_preshuffle_is_off(self, set_option):
        set_option("primus_turbo.mxfp4_preshuffle", False)
        assert _fp4_common.feature_multiple() == 128

    @pytest.mark.parametrize("shape,multiple,eligible", [((384, 256), 256, False), ((384, 256), 128, True)])
    def test_the_multiple_decides_eligibility(self, monkeypatch, shape, multiple, eligible):
        import sys

        module = types.ModuleType("nemo_automodel._diffusers.auto_diffusion_pipeline")
        module._is_fp8_training_safe_linear = lambda name, linear: True
        for name in ("nemo_automodel", "nemo_automodel._diffusers"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, module.__name__, module)
        linear = types.SimpleNamespace(weight=types.SimpleNamespace(shape=shape))
        assert _fp4_common.is_fp4_training_safe_linear("blocks.0.ff", linear, multiple) is eligible


class TestTokenPadding:
    """AITER's FP4 GEMM needs the contraction dim to be a multiple of 256. Only
    the weight gradient contracts over tokens, and the token count is a runtime
    property."""

    def test_the_contraction_multiple_matches_the_kernel(self):
        assert _fp4_common.AITER_K_MULTIPLE == 256

    def test_the_hipblaslt_feature_alignment_is_smaller(self):
        """128 is HIPBLASLT's multiple for in/out_features; the token count is
        always padded to AITER's 256."""
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
        import primus.backends.automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        return {p.id: p for p in PatchRegistry.iter_patches(backend="automodel", phase="before_train")}

    def test_the_patch_is_registered(self):
        assert "automodel.quantization.mxfp4_linear" in self._patches()

    def test_it_is_gated_off_by_default(self):
        assert self._patches()["automodel.quantization.mxfp4_linear"].condition(None) is False

    def test_it_turns_on_when_requested(self, set_option):
        set_option("primus_turbo.mxfp4_linear", True)
        assert self._patches()["automodel.quantization.mxfp4_linear"].condition(None) is True

    def test_only_one_quantization_patch_is_ever_active(self, set_option):
        """They rebind the same symbol, so two active conditions would mean the
        second silently overwrote the first."""
        set_option("primus_turbo.mxfp4_linear", True)
        set_option("primus_turbo.fp8_linear", True)
        by_id = self._patches()
        active = [
            pid
            for pid in ("automodel.quantization.mxfp4_linear", "automodel.quantization.fp8_linear")
            if by_id[pid].condition(None)
        ]
        assert active == ["automodel.quantization.mxfp4_linear"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
