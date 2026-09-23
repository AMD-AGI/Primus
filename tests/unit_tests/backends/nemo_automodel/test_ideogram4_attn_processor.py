###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 var-len attention processor.

WHAT THESE ARE DEFENDING:
  The processor chooses between four paths, and choosing wrong is silent in three
  of the four cases. So each test below pins WHICH path a given set of conditions
  takes, by recording which kernel was called rather than by inspecting the output:

    * a provided packing must take the var-len path and must NOT look at the mask,
      because that is the only path that is both exact on ragged batches and free
      of the device-to-host reads that graph-break under compilation;
    * a padded batch with no provided packing must derive the packing from the
      mask, not fall through to dense -- dense would let padding attend to real
      tokens and corrupt training with no error;
    * a trivial mask must take dense, because packing would be busywork;
    * an additive float mask must go back to the original dispatch rather than be
      reinterpreted as boundaries.

  The stale-packing guard is the other load-bearing test. The buffer outlives the
  step that published it, so a caller that bypasses the adapter would otherwise
  attend on a packing built for a different batch size. The guard compares only
  static shapes, so it is free, and it turns a silent corruption into an error.

The kernels are replaced with recorders. Nothing here measures attention numerics
-- that needs a GPU and belongs to the V7 compute stage. What is measured is
dispatch, which is where the silent mistakes live.
"""

import pytest

torch = pytest.importorskip("torch")

from primus.backends.nemo_automodel.attention import varlen_utils  # noqa: E402
from primus.backends.nemo_automodel.models.ideogram4 import (  # noqa: E402
    attn_processor,
    cu_seqlens,
    packing_buffer,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "PRIMUS_IDEOGRAM_VARLEN_ATTN",
        "PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS",
        "PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE",
    ):
        monkeypatch.delenv(var, raising=False)


class FakeAttention(torch.nn.Module):
    """Minimal stand-in for Ideogram4Attention: the projections and norms the
    processor calls, and nothing else."""

    def __init__(self, dim=16, num_heads=2, head_dim=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.to_q = torch.nn.Linear(dim, inner, bias=False)
        self.to_k = torch.nn.Linear(dim, inner, bias=False)
        self.to_v = torch.nn.Linear(dim, inner, bias=False)
        self.norm_q = torch.nn.Identity()
        self.norm_k = torch.nn.Identity()
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(inner, dim, bias=False)])
        self.processor = attn_processor.Ideogram4VarlenAttnProcessor()


@pytest.fixture
def recorder(monkeypatch):
    """Replace both kernels with recorders returning correctly shaped zeros."""
    calls = []

    def fake_varlen(q, k, v, cu, max_seqlen, *, deterministic=False):
        calls.append(
            {
                "path": "varlen",
                "cu_seqlens": cu.tolist(),
                "max_seqlen": max_seqlen,
                "deterministic": deterministic,
                "tokens": q.shape[0],
            }
        )
        return torch.zeros_like(q)

    def fake_dense(q, k, v, *, deterministic=False):
        calls.append({"path": "dense", "deterministic": deterministic})
        return torch.zeros_like(q)

    monkeypatch.setattr(varlen_utils, "varlen_flash_attention", fake_varlen)
    monkeypatch.setattr(varlen_utils, "dense_flash_attention", fake_dense)
    return calls


def rotary_for(batch, seq_len, head_dim):
    """Identity rotary embedding, so it cannot perturb what is being measured."""
    cos = torch.ones(batch, seq_len, head_dim)
    sin = torch.zeros(batch, seq_len, head_dim)
    return cos, sin


def mask_for(text_lengths, max_text, num_image):
    ids = torch.tensor([[0] * (max_text - n) + [1] * (n + num_image) for n in text_lengths])
    return (ids[:, :, None] == ids[:, None, :]).unsqueeze(1)


def run(processor, attn, batch, seq_len, dim=16, mask=None):
    hidden = torch.randn(batch, seq_len, dim)
    return processor(attn, hidden, mask, rotary_for(batch, seq_len, attn.head_dim))


class TestPathSelection:
    def test_a_provided_packing_takes_the_varlen_path(self, recorder):
        attn = FakeAttention()
        packing = cu_seqlens.build_cu_seqlens([3, 5], max_text_tokens=8, num_image_tokens=4)
        attn.register_buffer(packing_buffer.PACKING_ATTR, packing, persistent=False)
        setattr(attn, packing_buffer.BOUND_ATTR, cu_seqlens.static_max_seqlen(8, 4))

        run(attn.processor, attn, batch=2, seq_len=12)
        assert [c["path"] for c in recorder] == ["varlen"]
        assert recorder[0]["cu_seqlens"] == packing.tolist()
        assert recorder[0]["max_seqlen"] == 11

    def test_a_provided_packing_ignores_the_mask_entirely(self, recorder):
        """The model still materializes a mask; on this path it is dead weight, and
        reading it would reintroduce the host synchronization the packing exists to
        avoid."""
        attn = FakeAttention()
        packing = cu_seqlens.build_cu_seqlens([3, 5], max_text_tokens=8, num_image_tokens=4)
        attn.register_buffer(packing_buffer.PACKING_ATTR, packing, persistent=False)

        # A mask that disagrees with the packing. The packing must win.
        wrong_mask = torch.ones(2, 1, 12, 12, dtype=torch.bool)
        run(attn.processor, attn, batch=2, seq_len=12, mask=wrong_mask)
        assert recorder[0]["path"] == "varlen"
        assert recorder[0]["cu_seqlens"] == packing.tolist()

    def test_an_explicit_argument_beats_the_buffer(self, recorder):
        """So a future kwargs route could override the transport without the
        processor changing."""
        attn = FakeAttention()
        buffered = cu_seqlens.build_cu_seqlens([1, 1], max_text_tokens=8, num_image_tokens=4)
        attn.register_buffer(packing_buffer.PACKING_ATTR, buffered, persistent=False)
        explicit = cu_seqlens.build_cu_seqlens([3, 5], max_text_tokens=8, num_image_tokens=4)

        hidden = torch.randn(2, 12, 16)
        attn.processor(attn, hidden, None, rotary_for(2, 12, 8), cu_seqlens=explicit, max_seqlen=11)
        assert recorder[0]["cu_seqlens"] == explicit.tolist()

    def test_a_padded_batch_with_no_packing_derives_it_from_the_mask(self, recorder):
        """It must NOT fall through to dense: dense over a padded row lets padding
        attend to real tokens, which corrupts training with no error."""
        attn = FakeAttention()
        mask = mask_for([3, 5], max_text=8, num_image=4)
        run(attn.processor, attn, batch=2, seq_len=12, mask=mask)
        assert recorder[0]["path"] == "varlen"
        expected = cu_seqlens.build_cu_seqlens([3, 5], 8, 4)
        assert recorder[0]["cu_seqlens"] == expected.tolist()

    def test_a_trivial_mask_takes_dense(self, recorder):
        """One full segment per row means the mask says nothing, so packing would
        be busywork."""
        attn = FakeAttention()
        mask = torch.ones(2, 1, 12, 12, dtype=torch.bool)
        run(attn.processor, attn, batch=2, seq_len=12, mask=mask)
        assert recorder[0]["path"] == "dense"

    def test_no_mask_takes_dense(self, recorder):
        attn = FakeAttention()
        run(attn.processor, attn, batch=2, seq_len=12, mask=None)
        assert recorder[0]["path"] == "dense"

    def test_assume_dense_skips_the_mask_analysis(self, recorder, monkeypatch):
        attn = FakeAttention()
        monkeypatch.setattr(attn.processor, "assume_dense", True)
        mask = mask_for([3, 5], max_text=8, num_image=4)
        run(attn.processor, attn, batch=2, seq_len=12, mask=mask)
        assert recorder[0]["path"] == "dense"

    def test_a_provided_packing_outranks_assume_dense(self, recorder, monkeypatch):
        """assume_dense is exact only for unpadded batches, so a real packing must
        take precedence over it rather than the other way round."""
        attn = FakeAttention()
        monkeypatch.setattr(attn.processor, "assume_dense", True)
        packing = cu_seqlens.build_cu_seqlens([3, 5], max_text_tokens=8, num_image_tokens=4)
        attn.register_buffer(packing_buffer.PACKING_ATTR, packing, persistent=False)
        run(attn.processor, attn, batch=2, seq_len=12)
        assert recorder[0]["path"] == "varlen"

    def test_the_non_deterministic_backward_is_used(self, recorder):
        attn = FakeAttention()
        run(attn.processor, attn, batch=2, seq_len=12, mask=None)
        assert recorder[0]["deterministic"] is False


class TestAdditiveMask:
    def test_it_defers_to_the_original_dispatch(self, monkeypatch, recorder):
        """An additive mask carries magnitudes rather than boundaries, so it has no
        var-len form and must not be reinterpreted as one."""
        dispatched = []

        def fake_dispatch(q, k, v, attn_mask=None, backend=None, parallel_config=None):
            dispatched.append(attn_mask)
            return torch.zeros_like(q)

        module = pytest.importorskip("types")  # build a stand-in diffusers dispatch module
        import sys

        fake_module = module.ModuleType("diffusers.models.attention_dispatch")
        fake_module.dispatch_attention_fn = fake_dispatch
        monkeypatch.setitem(sys.modules, "diffusers.models.attention_dispatch", fake_module)

        attn = FakeAttention()
        additive = torch.zeros(2, 1, 12, 12)  # float, not bool
        run(attn.processor, attn, batch=2, seq_len=12, mask=additive)

        assert len(dispatched) == 1
        assert recorder == [], "no flash kernel should have been called"

    def test_it_warns_once(self, monkeypatch, caplog, recorder):
        import sys
        import types

        fake_module = types.ModuleType("diffusers.models.attention_dispatch")
        fake_module.dispatch_attention_fn = lambda q, k, v, **kw: torch.zeros_like(q)
        monkeypatch.setitem(sys.modules, "diffusers.models.attention_dispatch", fake_module)
        monkeypatch.setattr(attn_processor, "_warned", set())

        attn = FakeAttention()
        additive = torch.zeros(2, 1, 12, 12)
        with caplog.at_level("WARNING"):
            run(attn.processor, attn, batch=2, seq_len=12, mask=additive)
            run(attn.processor, attn, batch=2, seq_len=12, mask=additive)

        warnings = [r for r in caplog.records if "non-boolean" in r.getMessage()]
        assert len(warnings) == 1


class TestStalePackingGuard:
    """The buffer outlives the step that published it, so this turns a silent
    corruption into an error. It compares only static shapes, so it is free."""

    def test_a_packing_for_a_different_batch_size_is_refused(self, recorder):
        attn = FakeAttention()
        # Published for four rows, then run with two.
        stale = cu_seqlens.build_cu_seqlens([2, 2, 2, 2], max_text_tokens=8, num_image_tokens=4)
        attn.register_buffer(packing_buffer.PACKING_ATTR, stale, persistent=False)

        with pytest.raises(ValueError, match="2B\\+1"):
            run(attn.processor, attn, batch=2, seq_len=12)

    def test_the_error_says_what_to_do(self, recorder):
        attn = FakeAttention()
        stale = cu_seqlens.build_cu_seqlens([2] * 4, max_text_tokens=8, num_image_tokens=4)
        attn.register_buffer(packing_buffer.PACKING_ATTR, stale, persistent=False)

        with pytest.raises(ValueError) as excinfo:
            run(attn.processor, attn, batch=2, seq_len=12)
        assert "clear_packing" in str(excinfo.value)

    def test_a_matching_packing_passes(self, recorder):
        attn = FakeAttention()
        good = cu_seqlens.build_cu_seqlens([3, 5], max_text_tokens=8, num_image_tokens=4)
        attn.register_buffer(packing_buffer.PACKING_ATTR, good, persistent=False)
        run(attn.processor, attn, batch=2, seq_len=12)
        assert recorder[0]["path"] == "varlen"


class TestPackingShapes:
    def test_the_packed_token_count_matches_the_batch(self, recorder):
        attn = FakeAttention()
        packing = cu_seqlens.build_cu_seqlens([3, 5], max_text_tokens=8, num_image_tokens=4)
        attn.register_buffer(packing_buffer.PACKING_ATTR, packing, persistent=False)
        run(attn.processor, attn, batch=2, seq_len=12)
        assert recorder[0]["tokens"] == 2 * 12

    def test_the_output_keeps_the_input_shape(self, recorder):
        attn = FakeAttention()
        out = run(attn.processor, attn, batch=2, seq_len=12, mask=None)
        assert out.shape == (2, 12, 16)


class TestDtypeHandling:
    def test_rotary_promotion_is_cast_back(self, monkeypatch):
        """Rotary multiplies q and k by a float32 cos and sin, promoting them.
        Torch SDPA is autocast-aware and downcasts at its boundary; the flash op is
        not, so the cast has to happen in the processor to match the path it
        replaced. Without it the kernel receives mismatched dtypes."""
        seen = {}

        def fake_dense(q, k, v, *, deterministic=False):
            seen["q"] = q.dtype
            seen["k"] = k.dtype
            seen["v"] = v.dtype
            return torch.zeros_like(q)

        monkeypatch.setattr(varlen_utils, "dense_flash_attention", fake_dense)

        attn = FakeAttention().to(torch.bfloat16)
        hidden = torch.randn(2, 12, 16, dtype=torch.bfloat16)
        # float32 rotary, as the model produces.
        cos = torch.ones(2, 12, 8, dtype=torch.float32)
        sin = torch.zeros(2, 12, 8, dtype=torch.float32)
        attn.processor(attn, hidden, None, (cos, sin))

        assert seen["q"] == seen["v"] == torch.bfloat16
        assert seen["k"] == torch.bfloat16


class TestEnvGates:
    def test_varlen_is_off_by_default(self):
        assert attn_processor.is_varlen_attn_enabled() is False

    def test_precompute_defaults_on(self):
        assert attn_processor.precompute_cu_seqlens_enabled() is True

    def test_precompute_is_inactive_without_the_processor(self, monkeypatch):
        """Both switches have to be on. Precomputing for a run whose processor was
        never installed costs a build and a reserved token position for nothing,
        because the stock processor has no packing parameter."""
        assert attn_processor.precompute_cu_seqlens_enabled() is True
        assert attn_processor.precompute_cu_seqlens_active() is False

    def test_both_switches_on_activates_it(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        assert attn_processor.precompute_cu_seqlens_active() is True

    def test_the_kill_switch_deactivates_it(self, monkeypatch):
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        monkeypatch.setenv("PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS", "0")
        assert attn_processor.precompute_cu_seqlens_active() is False

    def test_assume_dense_is_off_by_default(self):
        assert attn_processor.assume_dense_enabled() is False


class TestProcessorSignature:
    def test_the_packing_parameters_are_named(self):
        """diffusers filters forwarded kwargs against this signature, so naming
        them is what would let a kwargs route work at all -- a **kwargs-only
        processor would silently receive nothing."""
        import inspect

        params = inspect.signature(attn_processor.Ideogram4VarlenAttnProcessor.__call__).parameters
        assert "cu_seqlens" in params
        assert "max_seqlen" in params

    def test_the_diffusers_discovery_attributes_are_present(self):
        cls = attn_processor.Ideogram4VarlenAttnProcessor
        assert hasattr(cls, "_attention_backend")
        assert hasattr(cls, "_parallel_config")


class TestInstall:
    def test_it_is_a_no_op_when_not_requested(self):
        assert attn_processor.install() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
