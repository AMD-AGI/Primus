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

import warnings
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


class _VarlenRecorder:
    """Stands in for ``flex_attention_varlen``; records positional + keyword args."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        self.calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "cu_seqlens_q": cu_q,
                "cu_seqlens_k": cu_k,
                "max_seqlen_q": max_q,
                "max_seqlen_k": max_k,
                **kwargs,
            }
        )
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

    varlen = _VarlenRecorder()
    monkeypatch.setattr(tfa, "turbo_flex_attention", recorder)
    monkeypatch.setattr(tfa, "turbo_flex_attention_bshd", recorder)
    monkeypatch.setattr(tfa, "turbo_flex_attention_varlen", varlen)
    monkeypatch.setattr(tfa, "turbo_create_block_mask", fake_create_block_mask)
    monkeypatch.setattr(tfa, "_FLEX_IMPORT_ERROR", None)
    tfa.clear_turbo_flex_block_mask_cache()
    yield SimpleNamespace(recorder=recorder, made=made, varlen=varlen)
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
        deterministic_mode=False,
        reset_attention_mask=False,
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
        # No window here on purpose: this asserts mask_mod beats the built-in *causal*
        # mask. mask_mod combined with a window is now a rejected conflict, covered by
        # TestRejections.test_user_mask_mod_plus_window_rejected.
        attn = tfa.TurboFlexAttention(mask_mod=_dummy_mask_mod, mask_mod_key="dummy")
        attn(*_qkv(), causal=True)
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
        assert call["deterministic"] is False

    @pytest.mark.parametrize("flag", [True, False])
    def test_deterministic_is_forwarded(self, stub_backend, flag):
        attn = tfa.TurboFlexAttention()
        attn(*_qkv(), causal=True, deterministic=flag)
        assert stub_backend.recorder.last["deterministic"] is flag

    @pytest.mark.parametrize("flag", [True, False])
    def test_deterministic_is_forwarded_on_the_packed_path(self, stub_backend, flag):
        attn = tfa.TurboFlexAttention()
        q = torch.randn(24, 4, 8)
        cu = torch.tensor([0, 8, 16, 24], dtype=torch.int32)
        attn(q, q.clone(), q.clone(), causal=True, cu_seqlens_q=cu, deterministic=flag)
        assert stub_backend.varlen.last["deterministic"] is flag

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
    def test_thd_without_boundaries_rejected(self, stub_backend):
        """THD is supported now (see TestPackedTHD), but only with cu_seqlens. Bare 3D
        input has no recoverable document structure, so it must still be refused rather
        than treated as one long sequence."""
        attn = tfa.TurboFlexAttention()
        t = torch.randn(32, 4, 8)
        with pytest.raises(NotImplementedError, match="cu_seqlens"):
            attn(t, t, t, causal=True)

    def test_2d_input_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        t = torch.randn(32, 8)
        with pytest.raises(NotImplementedError, match="THD"):
            attn(t, t, t, causal=True)

    def test_return_attn_probs_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match="return_attn_probs"):
            attn(*_qkv(), causal=True, return_attn_probs=True)

    def test_deterministic_rejected_on_an_older_turbo(self, stub_backend, monkeypatch):
        # The flag is threaded through now; it is only refused when the installed
        # compat layer has no parameter to put it in. A stub with an explicit
        # signature (no ``**kwargs``) stands in for that older build.
        def old_entry(
            q,
            k,
            v,
            score_mod=None,
            block_mask=None,
            scale=None,
            enable_gqa=False,
            return_lse=False,
            alibi_slopes=None,
            dropout_p=0.0,
            sink=None,
            bias=None,
        ):
            return torch.zeros_like(q)

        monkeypatch.setattr(tfa, "turbo_flex_attention_bshd", old_entry)
        monkeypatch.setattr(tfa, "turbo_flex_attention", old_entry)
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match="deterministic"):
            attn(*_qkv(), causal=True, deterministic=True)
        # ...and it is a *conditional* refusal: the default still runs.
        assert attn(*_qkv(), causal=True, deterministic=False) is not None

    def test_context_parallel_groups_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match="ulysses_group"):
            attn(*_qkv(), causal=True, ulysses_group=object())

    def test_window_without_causal_rejected(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match="sliding window"):
            attn(*_qkv(), causal=False, window_size=(4, 0))

    def test_forward_looking_window_rejected(self, stub_backend):
        """window_size[1] > 0 asks each query to see tokens ahead of it. Every mask_mod
        the dense path can build is causal, so the right bound cannot be honoured -- and
        it used to be dropped on the floor while the left bound was applied, which
        silently trains a different model than the config asked for."""
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError, match=r"window_size\[1\]"):
            attn(*_qkv(s=16), causal=True, window_size=(4, 4))

    def test_symmetric_window_is_not_silently_truncated(self, stub_backend):
        """The specific regression: (4, 4) must not be treated as (4, 0)."""
        attn = tfa.TurboFlexAttention()
        with pytest.raises(NotImplementedError):
            attn(*_qkv(s=16), causal=True, window_size=(4, 4))
        assert stub_backend.recorder.calls == [], "the backend was reached despite the rejection"

    def test_unbounded_right_window_still_accepted(self, stub_backend):
        """(-1) and 0 both mean 'no forward bound' and must keep working."""
        attn = tfa.TurboFlexAttention()
        attn(*_qkv(s=16), causal=True, window_size=(4, 0))
        attn(*_qkv(s=16), causal=True, window_size=(4, -1))
        assert len(stub_backend.recorder.calls) == 2

    def test_mask_mod_plus_sink_window_rejected_at_build(self, stub_backend):
        """The same conflict, but reachable from a plain config file -- so it has to be
        caught when the model is built, not on the first forward pass."""
        with pytest.raises(NotImplementedError, match="sink_sliding_window"):
            tfa.build_turbo_flex_attention(
                args=_args(
                    turbo_flex_attention_mask_mod="mod:fn",
                    sink_sliding_window=512,
                ),
                config=_config(),
            )

    def test_user_mask_mod_plus_window_rejected(self, stub_backend):
        """A user mask_mod replaces the window rather than composing with it. Resolving
        that silently would drop whichever one lost, so the conflict is an error."""
        attn = tfa.TurboFlexAttention(mask_mod=lambda b, h, q, kv: q >= kv)
        with pytest.raises(NotImplementedError, match="mask_mod"):
            attn(*_qkv(s=16), causal=True, window_size=(4, 0))


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

    def test_deterministic_mode_accepted_on_a_current_turbo(self, stub_backend):
        # The compat layer threads ``deterministic`` now, so deterministic_mode=true is
        # no longer a build-time rejection -- it is simply forwarded.
        attn = tfa.build_turbo_flex_attention(args=_args(deterministic_mode=True), config=_config())
        attn(*_qkv(), causal=True, deterministic=True)
        assert stub_backend.recorder.last["deterministic"] is True

    def test_deterministic_mode_rejected_at_build_on_an_older_turbo(self, stub_backend, monkeypatch):
        def old_entry(
            q,
            k,
            v,
            score_mod=None,
            block_mask=None,
            scale=None,
            enable_gqa=False,
            return_lse=False,
            alibi_slopes=None,
            dropout_p=0.0,
            sink=None,
            bias=None,
        ):
            return torch.zeros_like(q)

        monkeypatch.setattr(tfa, "turbo_flex_attention_bshd", old_entry)
        monkeypatch.setattr(tfa, "turbo_flex_attention", old_entry)
        with pytest.raises(NotImplementedError, match="deterministic"):
            tfa.build_turbo_flex_attention(args=_args(deterministic_mode=True), config=_config())

    def test_float8_rejected(self, stub_backend):
        with pytest.raises(NotImplementedError, match="float8"):
            tfa.build_turbo_flex_attention(args=_args(enable_turbo_attention_float8=True), config=_config())

    def test_missing_turbo_build_reports_clearly(self, monkeypatch, stub_backend):
        monkeypatch.setattr(tfa, "turbo_flex_attention", None)
        monkeypatch.setattr(tfa, "_FLEX_IMPORT_ERROR", ModuleNotFoundError("No module named 'primus_turbo'"))
        with pytest.raises(RuntimeError, match="does not provide"):
            tfa.build_turbo_flex_attention(args=_args(), config=_config())


# =============================================================================
# the config key itself
# =============================================================================
# packed sequences (THD) -> flex_attention_varlen
# =============================================================================


def _thd(total=24, h=4, d=8, lens=(8, 10, 6)):
    q = torch.randn(total, h, d)
    k = torch.randn(total, h, d)
    v = torch.randn(total, h, d)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    return q, k, v, cu, lens


class TestPackedTHD:
    """THD input must reach flex_attention_varlen with the boundaries intact.

    Before this path existed, packed sequences either hit a NotImplementedError or --
    on the direct flash_attn_func binding -- silently lost cu_seqlens and let tokens
    attend across documents. Both are covered here.
    """

    def test_thd_routes_to_varlen(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        out = attn(q, k, v, causal=True, cu_seqlens_q=cu, max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
        assert out.shape == q.shape
        assert len(stub_backend.varlen.calls) == 1
        assert len(stub_backend.recorder.calls) == 0, "must not go through the dense entry"

    def test_boundaries_are_forwarded_verbatim(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        attn(q, k, v, causal=True, cu_seqlens_q=cu, max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
        call = stub_backend.varlen.last
        assert torch.equal(call["cu_seqlens_q"], cu)
        assert torch.equal(call["cu_seqlens_k"], cu), "kv defaults to q when omitted"
        assert call["max_seqlen_q"] == max(lens)
        assert call["causal"] is True

    def test_max_seqlen_derived_when_missing(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        attn(q, k, v, causal=True, cu_seqlens_q=cu)
        call = stub_backend.varlen.last
        assert call["max_seqlen_q"] == max(lens)
        assert call["max_seqlen_k"] == max(lens)

    def test_separate_kv_boundaries(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        cu_kv = cu.clone()
        attn(
            q,
            k,
            v,
            causal=False,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=max(lens),
            max_seqlen_kv=max(lens),
        )
        assert stub_backend.varlen.last["cu_seqlens_k"] is cu_kv

    def test_no_block_mask_is_built(self, stub_backend):
        """Packing carries its boundaries explicitly -- nothing to probe or classify,
        so the BlockMask cache must stay empty (this is why THD does not thrash it)."""
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        attn(q, k, v, causal=True, cu_seqlens_q=cu)
        assert stub_backend.made == []
        assert len(tfa._BLOCK_MASK_CACHE) == 0

    def test_sink_is_cast_on_the_packed_path_too(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        sink = torch.zeros(4, dtype=torch.bfloat16)
        attn(q, k, v, causal=True, cu_seqlens_q=cu, sink=sink)
        assert stub_backend.varlen.last["sink"].dtype == torch.float32

    def test_3d_without_cu_seqlens_is_refused(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        with pytest.raises(NotImplementedError, match="cu_seqlens"):
            attn(q, k, v, causal=True)

    def test_cu_seqlens_with_4d_is_refused(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v = _qkv()
        cu = torch.tensor([0, 8, 16], dtype=torch.int32)
        with pytest.raises(NotImplementedError, match="THD expects"):
            attn(q, k, v, causal=True, cu_seqlens_q=cu)

    def test_user_mask_mod_is_refused_on_packed(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(
            args=_args(
                turbo_flex_attention_mask_mod=(
                    "tests.unit_tests.backends.megatron." "test_turbo_flex_attention:_dummy_mask_mod"
                )
            ),
            config=_config(),
        )
        q, k, v, cu, lens = _thd()
        with pytest.raises(NotImplementedError, match="mask_mod"):
            attn(q, k, v, causal=True, cu_seqlens_q=cu)

    def test_bias_is_refused_on_packed(self, stub_backend):
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        with pytest.raises(NotImplementedError, match="bias"):
            attn(q, k, v, causal=True, cu_seqlens_q=cu, bias=torch.zeros(1))

    def test_missing_varlen_entry_reports_clearly(self, stub_backend, monkeypatch):
        monkeypatch.setattr(tfa, "turbo_flex_attention_varlen", None)
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v, cu, lens = _thd()
        with pytest.raises(RuntimeError, match="flex_attention_varlen"):
            attn(q, k, v, causal=True, cu_seqlens_q=cu)

    def test_dense_path_is_unaffected(self, stub_backend):
        """The packed branch must not perturb the ordinary bshd call."""
        attn = tfa.build_turbo_flex_attention(args=_args(), config=_config())
        q, k, v = _qkv()
        attn(q, k, v, causal=True)
        assert len(stub_backend.recorder.calls) == 1
        assert len(stub_backend.varlen.calls) == 0


# =============================================================================


class TestSinkDtype:
    """Megatron allocates ``self.sinks`` in bfloat16; the compat layer demands fp32."""

    def test_bf16_sink_is_cast_to_fp32(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        q, k, v = _qkv(h=4)
        sink = torch.zeros(4, dtype=torch.bfloat16)
        attn(q, k, v, causal=True, sink=sink)
        assert stub_backend.recorder.last["sink"].dtype == torch.float32

    def test_fp32_sink_is_passed_through_untouched(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        q, k, v = _qkv(h=4)
        sink = torch.zeros(4, dtype=torch.float32)
        attn(q, k, v, causal=True, sink=sink)
        assert stub_backend.recorder.last["sink"] is sink

    def test_none_sink_stays_none(self, stub_backend):
        attn = tfa.TurboFlexAttention()
        q, k, v = _qkv()
        attn(q, k, v, causal=True, sink=None)
        assert stub_backend.recorder.last["sink"] is None

    def test_cast_keeps_the_parameter_trainable(self, stub_backend):
        """The cast must be differentiable, or the sink Parameter stops learning."""
        sink = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        cast = tfa._coerce_sink(sink)
        assert cast.dtype == torch.float32
        cast.sum().backward()
        assert sink.grad is not None
        assert sink.grad.dtype == torch.bfloat16


class TestBoundedCache:
    def test_cache_is_bounded_and_evicts(self, stub_backend, monkeypatch):
        monkeypatch.setattr(tfa, "_BLOCK_MASK_CACHE_MAX", 4)
        attn = tfa.TurboFlexAttention()
        for s_len in range(8, 8 + 10):
            q, k, v = _qkv(s=s_len)
            attn(q, k, v, causal=True)
        assert len(tfa._BLOCK_MASK_CACHE) <= 4

    def test_eviction_warns_once(self, stub_backend, monkeypatch):
        monkeypatch.setattr(tfa, "_BLOCK_MASK_CACHE_MAX", 2)
        attn = tfa.TurboFlexAttention()
        with pytest.warns(RuntimeWarning, match="BlockMask cache"):
            for s_len in range(8, 14):
                q, k, v = _qkv(s=s_len)
                attn(q, k, v, causal=True)
        # A second burst must stay quiet -- one warning per process, not per miss.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for s_len in range(20, 26):
                q, k, v = _qkv(s=s_len)
                attn(q, k, v, causal=True)

    def test_hot_shape_survives_a_cold_tail(self, stub_backend, monkeypatch):
        """LRU, not FIFO: the shape used every step must not be evicted by the tail."""
        monkeypatch.setattr(tfa, "_BLOCK_MASK_CACHE_MAX", 4)
        attn = tfa.TurboFlexAttention()
        hot = _qkv(s=64)
        attn(*hot, causal=True)
        first = stub_backend.recorder.last["block_mask"]
        for s_len in range(8, 20):
            attn(*_qkv(s=s_len), causal=True)
            attn(*hot, causal=True)  # touched every step, so it stays hot
        attn(*hot, causal=True)
        assert stub_backend.recorder.last["block_mask"] is first


class TestBuildTimeRejections:
    """These combinations must fail at model build, not on the first forward.

    ``deterministic_mode`` used to be one of them and no longer is: the compat layer
    threads ``deterministic`` through, so the build only rejects it on a Turbo whose
    entries predate the parameter. Both halves of that narrower contract live in
    ``TestBuild`` -- ``test_deterministic_mode_accepted_on_a_current_turbo`` and
    ``test_deterministic_mode_rejected_at_build_on_an_older_turbo``.
    """

    def test_reset_attention_mask_rejected_at_build(self, stub_backend):
        with pytest.raises(NotImplementedError, match="reset_attention_mask"):
            tfa.build_turbo_flex_attention(args=_args(reset_attention_mask=True), config=_config())

    def test_defaults_still_build(self, stub_backend):
        assert tfa.build_turbo_flex_attention(args=_args(), config=_config()) is not None


class TestConfigSchema:
    def test_registered_and_defaults_to_false(self):
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[4]
        cfg = yaml.safe_load((root / "primus/configs/modules/megatron/primus_turbo.yaml").read_text())
        assert cfg["use_turbo_flex_attention"] is False
        assert cfg["turbo_flex_attention_mask_mod"] is None
        assert cfg["turbo_flex_attention_score_mod"] is None


class TestRejectResetAttentionMask:
    """The direct-path guard for the mask tensor ``forward()`` never reads.

    Before this guard, ``PrimusTurboAttention.forward`` accepted ``attention_mask`` and
    never looked at it. With ``reset_attention_mask=true`` that silently trained a
    different model than the config asked for: the per-document boundaries were dropped
    and tokens attended across documents, with no error and no warning anywhere.
    """

    def test_off_is_a_noop(self):
        tfa.reject_reset_attention_mask(_args(reset_attention_mask=False), is_flex=False, where="X")

    def test_missing_attribute_is_a_noop(self):
        # Older arg namespaces predate the flag; absence must not raise.
        tfa.reject_reset_attention_mask(SimpleNamespace(), is_flex=False, where="X")

    def test_on_and_direct_path_raises(self):
        with pytest.raises(NotImplementedError, match="reset_attention_mask"):
            tfa.reject_reset_attention_mask(_args(reset_attention_mask=True), is_flex=False, where="X")

    def test_error_names_the_caller_and_the_supported_alternative(self):
        with pytest.raises(NotImplementedError) as excinfo:
            tfa.reject_reset_attention_mask(
                _args(reset_attention_mask=True), is_flex=False, where="MyAttention"
            )
        msg = str(excinfo.value)
        assert "MyAttention" in msg
        assert "thd" in msg

    def test_flex_path_defers_to_its_own_message(self):
        # build_turbo_flex_attention raises for itself with a message that names the THD
        # alternative in flex terms; this helper must not pre-empt it.
        tfa.reject_reset_attention_mask(_args(reset_attention_mask=True), is_flex=True, where="X")

    def test_flex_build_still_rejects_it(self, stub_backend):
        # The deferral above is only sound while the flex side actually raises.
        with pytest.raises(NotImplementedError, match="reset_attention_mask"):
            tfa.build_turbo_flex_attention(args=_args(reset_attention_mask=True), config=_config())


class TestSinkKwargsFor:
    """The three-way answer to "can this entry take a sink?".

    Primus bound one of several attention entries and then passed ``sink=`` to all of
    them. Only ``flash_attn_func`` has the parameter: the context-parallel, fp8 and
    packed-varlen entries do not. So every ``context_parallel_size > 1`` run died on a
    bare ``TypeError: flash_attn_usp_func() got an unexpected keyword argument 'sink'``
    -- even with no sink configured, and with nothing in the message to say why
    (measured on 2x MI355, TASK_PROGRESS EXP21 case B; the entry inventory is EXP22).
    """

    @staticmethod
    def _takes_sink(q, k, v, sink=None):
        return None

    @staticmethod
    def _no_sink(q, k, v):
        return None

    @staticmethod
    def _kwargs_catch_all(q, k, v, **kw):
        return None

    def test_forwards_when_the_entry_takes_one(self):
        sink = torch.zeros(4)
        got = tfa.sink_kwargs_for(self._takes_sink, sink, where="X")
        assert list(got) == ["sink"]
        assert got["sink"] is sink

    def test_forwards_none_when_the_entry_takes_one(self):
        # None is the parameter's own default; passing it explicitly is harmless and
        # keeps the forwarding branch free of a second special case.
        assert tfa.sink_kwargs_for(self._takes_sink, None, where="X") == {"sink": None}

    def test_omits_when_the_entry_cannot_take_one_and_there_is_nothing_to_lose(self):
        # The bug this fixes: no sink configured, yet the call still died on the kwarg.
        assert tfa.sink_kwargs_for(self._no_sink, None, where="X") == {}

    def test_raises_when_a_sink_would_be_dropped(self):
        # Omitting it here would change the softmax denominator of every query and
        # train a different model in silence -- so this must be loud, not lenient.
        with pytest.raises(NotImplementedError, match="attention sink"):
            tfa.sink_kwargs_for(self._no_sink, torch.zeros(4), where="X")

    def test_error_names_the_entry_and_the_configuration(self):
        with pytest.raises(NotImplementedError) as excinfo:
            tfa.sink_kwargs_for(self._no_sink, torch.zeros(4), where="context_parallel_size=2")
        msg = str(excinfo.value)
        assert "_no_sink" in msg
        assert "context_parallel_size=2" in msg

    def test_unbound_entry_with_a_sink_raises_rather_than_calling_none(self):
        # attn_varlen is None when no varlen entry was bound; a sink plus that
        # combination is a configuration error, not an AttributeError later on.
        with pytest.raises(NotImplementedError):
            tfa.sink_kwargs_for(None, torch.zeros(4), where="X")

    def test_kwargs_catch_all_is_treated_as_accepting(self):
        # A **kwargs entry can genuinely take a sink, and refusing it would ground
        # wrappers that forward everything. The cost is that a permissive test double
        # also passes -- acceptable, since a real entry that ignores **kwargs is a bug
        # on its own side.
        assert tfa.sink_kwargs_for(self._kwargs_catch_all, None, where="X") == {"sink": None}
