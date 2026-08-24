###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU unit tests for the ``use_turbo_flex_attention`` routing layer.

Everything here runs without a GPU and without a real Primus-Turbo build: the
compat-layer entry and ``create_block_mask`` are stubbed, so what is under test is
the Primus-side wiring -- config validation, mask selection, the BlockMask cache,
argument passthrough, and the explicit rejections.
"""

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.extensions import turbo_flex_attention as tfa


def _dummy_mask_mod(b, h, q_idx, kv_idx):
    """Module-level callable used to exercise dotted-path resolution."""
    return q_idx >= kv_idx


NOT_CALLABLE = 42


class _Recorder:
    """Stands in for ``flex_attention_bshd``; records what it was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, **kwargs):
        self.calls.append({"q": q, "k": k, "v": v, **kwargs})
        return torch.zeros_like(q)

    @property
    def last(self):
        return self.calls[-1]


@pytest.fixture
def stub_backend(monkeypatch):
    """Install a recording compat-layer entry + a cheap BlockMask factory."""
    recorder = _Recorder()
    made = []

    def fake_create_block_mask(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None, device=None):
        obj = SimpleNamespace(mask_mod=mask_mod, shape=(Q_LEN, KV_LEN), device=device)
        made.append(obj)
        return obj

    monkeypatch.setattr(tfa, "turbo_flex_attention", recorder)
    monkeypatch.setattr(tfa, "turbo_flex_attention_bshd", recorder)
    monkeypatch.setattr(tfa, "turbo_create_block_mask", fake_create_block_mask)
    monkeypatch.setattr(tfa, "_FLEX_IMPORT_ERROR", None)
    tfa.clear_turbo_flex_block_mask_cache()
    yield SimpleNamespace(recorder=recorder, made=made)
    tfa.clear_turbo_flex_block_mask_cache()


def _qkv(b=2, s=16, h=4, d=8, hkv=None):
    hkv = h if hkv is None else hkv
    return (
        torch.randn(b, s, h, d),
        torch.randn(b, s, hkv, d),
        torch.randn(b, s, hkv, d),
    )


def _args(**overrides):
    base = dict(
        use_turbo_flex_attention=True,
        enable_turbo_attention_float8=False,
        turbo_flex_attention_mask_mod=None,
        turbo_flex_attention_score_mod=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _config(**overrides):
    base = dict(context_parallel_size=1)
    base.update(overrides)
    return SimpleNamespace(**base)


# =============================================================================
# dotted-path resolution
# =============================================================================


class TestResolveDottedCallable:
    def test_resolves_module_attribute(self):
        fn = tfa.resolve_dotted_callable(
            "tests.unit_tests.backends.megatron.test_turbo_flex_attention:_dummy_mask_mod",
            what="mask_mod",
        )
        assert fn is _dummy_mask_mod

    def test_missing_colon_rejected(self):
        with pytest.raises(ValueError, match="package.module:attribute"):
            tfa.resolve_dotted_callable("some.module.attr", what="mask_mod")

    def test_non_string_rejected(self):
        with pytest.raises(ValueError):
            tfa.resolve_dotted_callable(_dummy_mask_mod, what="mask_mod")

    def test_unknown_module_reports_clearly(self):
        with pytest.raises(ImportError, match="could not import module"):
            tfa.resolve_dotted_callable("primus_no_such_module_xyz:fn", what="mask_mod")

    def test_unknown_attribute_reports_clearly(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            tfa.resolve_dotted_callable(
                "tests.unit_tests.backends.megatron.test_turbo_flex_attention:nope",
                what="mask_mod",
            )

    def test_non_callable_rejected(self):
        with pytest.raises(TypeError, match="not callable"):
            tfa.resolve_dotted_callable(
                "tests.unit_tests.backends.megatron.test_turbo_flex_attention:NOT_CALLABLE",
                what="mask_mod",
            )


# =============================================================================
# mask_mod semantics
# =============================================================================


class TestMaskMods:
    def test_causal_mask_mod(self):
        q = torch.arange(4).view(-1, 1)
        kv = torch.arange(4).view(1, -1)
        assert torch.equal(tfa._causal_mask_mod(0, 0, q, kv), q >= kv)

    def test_window_causal_mask_mod(self):
        q = torch.arange(8).view(-1, 1)
        kv = torch.arange(8).view(1, -1)
        got = tfa._make_window_causal_mask_mod(3)(0, 0, q, kv)
        expected = (q >= kv) & ((q - kv) <= 3)
        assert torch.equal(got, expected)
        # row 7 keeps exactly the 4 positions 4..7
        assert int(got[7].sum()) == 4


# =============================================================================
# BlockMask cache
# =============================================================================


class TestBlockMaskCache:
    def test_same_shape_reuses_one_mask(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        q, k, v = _qkv()
        for _ in range(5):
            attn(q, k, v, causal=True)
        assert len(stub_backend.made) == 1
        masks = {id(c["block_mask"]) for c in stub_backend.recorder.calls}
        assert len(masks) == 1

    def test_different_seqlen_builds_a_second_mask(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        attn(*_qkv(s=16), causal=True)
        attn(*_qkv(s=32), causal=True)
        assert len(stub_backend.made) == 2

    def test_window_and_causal_are_distinct_entries(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        q, k, v = _qkv()
        attn(q, k, v, causal=True)
        attn(q, k, v, causal=True, window_size=(4, 0))
        attn(q, k, v, causal=True, window_size=(8, 0))
        assert len(stub_backend.made) == 3

    def test_clear_cache_forces_a_rebuild(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        q, k, v = _qkv()
        attn(q, k, v, causal=True)
        tfa.clear_turbo_flex_block_mask_cache()
        attn(q, k, v, causal=True)
        assert len(stub_backend.made) == 2


# =============================================================================
# dispatch behaviour
# =============================================================================


class TestDispatch:
    def test_no_mask_sends_no_block_mask(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        attn(*_qkv(), causal=False)
        assert stub_backend.recorder.last["block_mask"] is None
        assert stub_backend.made == []

    def test_causal_sends_a_causal_block_mask(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        attn(*_qkv(), causal=True)
        bm = stub_backend.recorder.last["block_mask"]
        assert bm is not None
        q = torch.arange(4).view(-1, 1)
        kv = torch.arange(4).view(1, -1)
        assert torch.equal(bm.mask_mod(0, 0, q, kv), q >= kv)

    def test_window_size_becomes_a_window_mask(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        attn(*_qkv(s=16), causal=True, window_size=(4, 0))
        bm = stub_backend.recorder.last["block_mask"]
        q = torch.arange(16).view(-1, 1)
        kv = torch.arange(16).view(1, -1)
        expected = (q >= kv) & ((q - kv) <= 4)
        assert torch.equal(bm.mask_mod(0, 0, q, kv), expected)

    def test_user_mask_mod_overrides_causal(self, stub_backend):
        attn = tfa.TurboFlexAttention(mask_mod=_dummy_mask_mod, mask_mod_key="dummy")
        attn(*_qkv(), causal=True, window_size=(4, 0))
        assert stub_backend.recorder.last["block_mask"].mask_mod is _dummy_mask_mod

    def test_score_mod_is_forwarded(self, stub_backend):
        def score_mod(score, b, h, q_idx, kv_idx):
            return score

        attn = tfa.TurboFlexAttention(score_mod=score_mod)
        attn(*_qkv(), causal=True)
        assert stub_backend.recorder.last["score_mod"] is score_mod

    def test_enable_gqa_follows_head_counts(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        attn(*_qkv(h=8, hkv=8), causal=True)
        assert stub_backend.recorder.last["enable_gqa"] is False
        attn(*_qkv(h=8, hkv=2), causal=True)
        assert stub_backend.recorder.last["enable_gqa"] is True

    def test_optional_arguments_are_forwarded(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        sink = torch.zeros(4)
        alibi = torch.ones(4)
        bias = torch.zeros(2, 4, 16, 16)
        attn(
            *_qkv(),
            dropout_p=0.1,
            softmax_scale=0.25,
            causal=True,
            bias=bias,
            alibi_slopes=alibi,
            sink=sink,
        )
        call = stub_backend.recorder.last
        assert call["scale"] == 0.25
        assert call["dropout_p"] == 0.1
        assert call["sink"] is sink
        assert call["alibi_slopes"] is alibi
        assert call["bias"] is bias

    def test_bshd_entry_receives_tensors_uncopied(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        q, k, v = _qkv()
        attn(q, k, v, causal=True)
        call = stub_backend.recorder.last
        assert call["q"].data_ptr() == q.data_ptr()
        assert call["k"].data_ptr() == k.data_ptr()
        assert call["v"].data_ptr() == v.data_ptr()

    def test_legacy_bhsd_entry_round_trips_the_layout(self, monkeypatch, stub_backend):
        """Older Turbo builds without flex_attention_bshd still get bshd out."""
        monkeypatch.setattr(tfa, "turbo_flex_attention_bshd", None)
        attn = tfa.TurboFlexAttention()
        assert attn._entry_is_bshd is False
        q, k, v = _qkv(b=2, s=16, h=4, d=8)
        out = attn(q, k, v, causal=True)
        seen_q = stub_backend.recorder.last["q"]
        assert seen_q.shape == (2, 4, 16, 8)  # bhsd on the way in
        assert out.shape == (2, 16, 4, 8)  # bshd on the way out

    def test_return_lse_tuple_is_preserved(self, monkeypatch, stub_backend):
        monkeypatch.setattr(tfa, "turbo_flex_attention_bshd", None)

        def entry(q, k, v, **kwargs):
            return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[1], q.shape[2])

        monkeypatch.setattr(tfa, "turbo_flex_attention", entry)
        attn = tfa.TurboFlexAttention()
        out, lse = attn(*_qkv(b=2, s=16, h=4, d=8), causal=True, return_lse=True)
        assert out.shape == (2, 16, 4, 8)
        assert lse.shape == (2, 4, 16)


# =============================================================================
# explicit rejections (never a silent fallback)
# =============================================================================


class TestRejections:
    def test_thd_input_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        t = torch.randn(32, 4, 8)
        with pytest.raises(NotImplementedError, match="bshd"):
            attn(t, t, t, causal=True)

    def test_return_attn_probs_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match="return_attn_probs"):
            attn(*_qkv(), causal=True, return_attn_probs=True)

    def test_deterministic_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match="deterministic"):
            attn(*_qkv(), causal=True, deterministic=True)

    def test_context_parallel_groups_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match="ulysses_group"):
            attn(*_qkv(), causal=True, ulysses_group=object())

    def test_window_without_causal_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match="sliding window"):
            attn(*_qkv(), causal=False, window_size=(4, 0))


# =============================================================================
# build-time validation
# =============================================================================


class TestBuild:
    def test_builds_with_defaults(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        assert isinstance(attn, tfa.TurboFlexAttention)
        assert attn.mask_mod is None and attn.score_mod is None

    def test_resolves_configured_hooks(self, stub_backend):
        path = "tests.unit_tests.backends.megatron.test_turbo_flex_attention:_dummy_mask_mod"
        attn = tfa.build_turbo_flex_attention(
            args=_args(turbo_flex_attention_mask_mod=path, turbo_flex_attention_score_mod=path),
            config=_config(),
        )
        assert attn.mask_mod is _dummy_mask_mod
        assert attn.score_mod is _dummy_mask_mod

    def test_context_parallel_rejected(self, stub_backend):
        with pytest.raises(NotImplementedError, match="context parallel"):
            tfa.build_turbo_flex_attention(args=_args(), config=_config(context_parallel_size=2))

    def test_float8_rejected(self, stub_backend):
        with pytest.raises(NotImplementedError, match="float8"):
            tfa.build_turbo_flex_attention(
                args=_args(enable_turbo_attention_float8=True), config=_config()
            )

    def test_missing_turbo_build_reports_clearly(self, monkeypatch, stub_backend):
        monkeypatch.setattr(tfa, "turbo_flex_attention", None)
        monkeypatch.setattr(
            tfa, "_FLEX_IMPORT_ERROR", ModuleNotFoundError("No module named 'primus_turbo'")
        )
        with pytest.raises(RuntimeError, match="does not provide"):
            tfa.build_turbo_flex_attention(args=_args(), config=_config())


# =============================================================================
# the config key itself
# =============================================================================


class TestConfigSchema:
    def test_registered_and_defaults_to_false(self):
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[4]
        cfg = yaml.safe_load(
            (root / "primus/configs/modules/megatron/primus_turbo.yaml").read_text()
        )
        assert cfg["use_turbo_flex_attention"] is False
        assert cfg["turbo_flex_attention_mask_mod"] is None
        assert cfg["turbo_flex_attention_score_mod"] is None
