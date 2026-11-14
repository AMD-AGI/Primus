###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU tests for MuseSparkModel (soft-cap, embedding norm, the dropout rejection, the soft-cap
in the chunked linear+CE) and the
gated attention output (per-head o_norm, then the sigmoid gate)."""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")  # Megatron-LM (third_party) is not on the path: skip

from megatron.core.models.common.embeddings.language_model_embedding import (  # noqa: E402
    LanguageModelEmbedding,
)
from megatron.core.models.gpt.gpt_model import GPTModel  # noqa: E402
from megatron.core.transformer.attention import SelfAttention  # noqa: E402

from primus.backends.megatron.core.models.muse_spark import (  # noqa: E402
    muse_spark_model as msm,
)
from primus.backends.megatron.core.transformer.muse_spark.attention import (  # noqa: E402
    MuseSparkSelfAttention,
)

CAP, MULT = 20.0, 0.17677669529663687  # the released config's values


def _ref_rms(x, eps):  # HF: x * rsqrt(mean(x^2) + eps) in fp32
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# ---------------------------------------------------------------- soft-cap
def test_softcap_logits_matches_the_formula():
    x = torch.linspace(-2000.0, 2000.0, 101, dtype=torch.float64)
    got = msm.softcap_logits(x, MULT, CAP)
    torch.testing.assert_close(got, CAP * torch.tanh(x * MULT / CAP), rtol=0, atol=0)
    assert got.abs().max() < CAP  # bounded by the cap
    assert got.abs().max() > CAP - 1e-6  # and saturates at it


def test_softcap_logits_keeps_the_input_and_its_gradient():
    # the in-place div_ / tanh_ work on the x * m temporary: the input is unchanged and
    # the gradient is the out-of-place formula's
    x = torch.randn(4, 7, dtype=torch.float64) * 300
    x0 = x.clone()
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    g = torch.randn(4, 7, dtype=torch.float64)
    msm.softcap_logits(a, MULT, CAP).backward(g)
    (CAP * torch.tanh(b * MULT / CAP)).backward(g)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert torch.equal(a.detach(), x0)


@pytest.mark.parametrize("cap", [None, 0, 0.0, "0", "0.0"])
def test_softcap_logits_without_a_cap_only_scales(cap):
    x = torch.randn(3, 5, dtype=torch.float64)
    torch.testing.assert_close(msm.softcap_logits(x, MULT, cap), x * MULT, rtol=0, atol=0)


def test_softcap_logits_accepts_a_numeric_string_cap():
    # a quoted yaml value ("20") caps like the float, no TypeError in forward
    x = torch.linspace(-500.0, 500.0, 11, dtype=torch.float64)
    torch.testing.assert_close(msm.softcap_logits(x, MULT, "20"), msm.softcap_logits(x, MULT, CAP))
    assert msm.softcap_value("20") == 20.0 and msm.softcap_value("0") is None
    assert msm.softcap_value(None) is None


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -5.0, "-20", True, "abc", 10**400])
def test_softcap_value_rejects_a_bad_cap(bad):
    # inf gives NaN logits, a negative cap flips every logit's sign, a bool is no cap value
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        msm.softcap_value(bad)
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        msm.softcap_logits(torch.ones(2), MULT, bad)


def test_softcap_value_rejects_a_numpy_bool_and_normalizes_negative_zero():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        msm.softcap_value(np.True_)
    from primus.backends.megatron.core.models.muse_spark.number_utils import (
        parse_softcap,
    )

    v = parse_softcap("-0")
    assert v == 0.0 and math.copysign(1.0, v) == 1.0
    assert msm.softcap_value("-0") is None


def _model(mup=False, cap=CAP, mult=MULT):
    m = object.__new__(msm.MuseSparkModel)  # no Megatron parallel state needed
    object.__setattr__(
        m,
        "config",
        SimpleNamespace(
            use_mup=mup, mup_output_mult=1.0, output_logit_multiplier=mult, final_logit_softcapping=cap
        ),
    )
    return m


def test_scale_logits_applies_the_multiplier_then_the_cap():
    x = torch.randn(4, 2, 8, dtype=torch.float64) * 300
    got = _model()._scale_logits(x)
    torch.testing.assert_close(got, CAP * torch.tanh(x * MULT / CAP), rtol=0, atol=0)


def test_scale_logits_calls_the_parent_first(monkeypatch):
    seen = []

    def parent(self, logits):
        seen.append(logits.clone())
        return logits * 2  # stands in for the MuP scale

    monkeypatch.setattr(GPTModel, "_scale_logits", parent)
    x = torch.randn(6, dtype=torch.float64)
    got = _model()._scale_logits(x)
    assert len(seen) == 1 and torch.equal(seen[0], x)
    torch.testing.assert_close(got, CAP * torch.tanh(2 * x * MULT / CAP), rtol=0, atol=0)


# ---------------------------------------------------------------- embedding norm
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_normed_embedding_forward_is_a_weightless_fp32_rms_norm(monkeypatch, dtype):
    eps = 1e-5
    raw = (torch.randn(5, 2, 16) * 3).to(dtype)
    calls = []

    def parent_forward(self, *args, **kwargs):
        calls.append((args, kwargs))
        return raw

    monkeypatch.setattr(LanguageModelEmbedding, "forward", parent_forward)
    emb = object.__new__(msm.MuseSparkNormedEmbedding)
    torch.nn.Module.__init__(emb)
    emb.muse_embed_norm_eps = eps
    out = emb(torch.zeros(2, 5, dtype=torch.long), position_ids=None)
    assert len(calls) == 1 and "position_ids" in calls[0][1]  # arguments passed through
    assert out.dtype == dtype and out.shape == raw.shape
    torch.testing.assert_close(out, _ref_rms(raw, eps).to(dtype), rtol=0, atol=0)
    # weightless: the subclass only overrides forward (no __init__, no parameter or buffer of
    # its own), so the built embedding has exactly LanguageModelEmbedding's parameters
    own = {k for k, v in vars(msm.MuseSparkNormedEmbedding).items() if not k.startswith("__")}
    assert own == {"forward"} and "__init__" not in vars(msm.MuseSparkNormedEmbedding)
    # unit RMS per token (up to eps and the output dtype)
    rms = out.float().pow(2).mean(-1).sqrt()
    torch.testing.assert_close(
        rms, torch.ones_like(rms), rtol=2e-2 if dtype == torch.bfloat16 else 1e-4, atol=0
    )


# ---------------------------------------------------------------- build / forward rejections
def _fake_gpt_init(self, config, *args, **kwargs):
    torch.nn.Module.__init__(self)
    self.config = config
    self.pre_process = True
    self.embedding = object.__new__(LanguageModelEmbedding)
    torch.nn.Module.__init__(self.embedding)


@pytest.mark.parametrize("dropout", [0.1, 1e-6])
def test_model_rejects_hidden_dropout_with_the_embedding_norm(monkeypatch, dropout):
    monkeypatch.setattr(GPTModel, "__init__", _fake_gpt_init)
    cfg = SimpleNamespace(embedding_norm=True, hidden_dropout=dropout, layernorm_epsilon=1e-5)
    with pytest.raises(ValueError, match="hidden_dropout must be 0"):
        msm.MuseSparkModel(cfg)


@pytest.mark.parametrize("dropout", [0.0, None])
def test_model_swaps_in_the_normed_embedding_at_zero_dropout(monkeypatch, dropout):
    monkeypatch.setattr(GPTModel, "__init__", _fake_gpt_init)
    cfg = SimpleNamespace(embedding_norm=True, hidden_dropout=dropout, layernorm_epsilon=1e-6)
    m = msm.MuseSparkModel(cfg)
    assert type(m.embedding) is msm.MuseSparkNormedEmbedding
    assert m.embedding.muse_embed_norm_eps == 1e-6


def test_model_without_embedding_norm_ignores_dropout(monkeypatch):
    monkeypatch.setattr(GPTModel, "__init__", _fake_gpt_init)
    m = msm.MuseSparkModel(SimpleNamespace(embedding_norm=False, hidden_dropout=0.1, layernorm_epsilon=1e-5))
    assert type(m.embedding) is LanguageModelEmbedding


def test_model_does_not_override_postprocess():
    # no fused-CE refusal: the chunked linear+CE calls _scale_logits itself
    assert "_postprocess" not in vars(msm.MuseSparkModel)


def test_chunked_linear_ce_applies_the_soft_cap():
    from primus.backends.megatron.patches.fused_linear_ce_patches import (
        _ChunkedLinearCE,
    )

    torch.manual_seed(0)
    s, b, h, v = 5, 2, 4, 7
    model = _model()
    model.compute_language_model_loss = lambda labels, logits: torch.nn.functional.cross_entropy(
        logits.transpose(0, 1).reshape(-1, v), labels.reshape(-1), reduction="none"
    ).view(labels.shape)
    hidden = (torch.randn(s, b, h, dtype=torch.float64) * 30).requires_grad_(True)
    weight = (torch.randn(v, h, dtype=torch.float64) * 30).requires_grad_(True)
    labels = torch.randint(0, v, (b, s))

    got = _ChunkedLinearCE.apply(hidden, weight, labels, model, 2)
    g_h, g_w = torch.autograd.grad(got.sum(), (hidden, weight))

    logits = CAP * torch.tanh(torch.matmul(hidden, weight.t()) * MULT / CAP)
    want = model.compute_language_model_loss(labels, logits)
    w_h, w_w = torch.autograd.grad(want.sum(), (hidden, weight))
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(g_h, w_h)
    torch.testing.assert_close(g_w, w_w)
    uncapped = model.compute_language_model_loss(labels, torch.matmul(hidden, weight.t()) * MULT)
    assert not torch.allclose(got, uncapped)


# ---------------------------------------------------------------- attention output gate
def _attn(hn, eps):
    a = object.__new__(MuseSparkSelfAttention)
    a.hidden_size_per_attention_head = hn
    a.o_norm_eps = eps
    a.o_norm_fused = True  # the CPU tensors here take the eager path anyway
    return a


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_gate_is_per_head_rms_then_sigmoid_gate(dtype):
    sq, b, nh, hn, eps = 3, 2, 4, 8, 1e-5
    # heads with very different scales: a whole-row norm would give a different result
    scale = torch.tensor([0.01, 1.0, 10.0, 300.0]).repeat_interleave(hn)
    x = (torch.randn(sq, b, nh * hn) * scale).to(dtype)
    gate = torch.randn(sq, b, nh * hn).to(dtype)
    got = _attn(hn, eps)._apply_output_gate(x, gate)

    normed = _ref_rms(x.view(sq, b, nh, hn), eps).to(dtype).view(sq, b, nh * hn)
    want = (normed * torch.sigmoid(gate.float())).to(dtype)
    assert got.dtype == dtype and got.shape == x.shape
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    whole_row = (_ref_rms(x, eps).to(dtype) * torch.sigmoid(gate.float())).to(dtype)
    assert not torch.allclose(got.float(), whole_row.float(), atol=1e-3)  # per head, not per row


def test_output_gate_uses_o_norm_eps_and_head_size():
    hn = 4
    # two heads at very different scales, and no constant run inside head 0: a wrong head
    # size (a split inside a head, or the whole row) changes the RMS of the result
    head0 = 1e-3 * torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    head1 = torch.full((hn,), 5.0, dtype=torch.float64)
    x = torch.cat([head0, head1]).view(1, 1, 2 * hn)
    gate = torch.zeros_like(x)  # sigmoid(0) = 1/2
    for eps in (1e-6, 1.0):
        got = _attn(hn, eps)._apply_output_gate(x, gate).view(2 * hn)
        want0 = 0.5 * head0 / math.sqrt(float((head0**2).mean()) + eps)
        want1 = 0.5 * head1 / math.sqrt(25.0 + eps)
        torch.testing.assert_close(got[:hn], want0)  # head 0 alone
        torch.testing.assert_close(got[hn:], want1)  # head 1 alone
    # with a small eps, another head size (a split inside a head, or the whole row) differs
    got = _attn(hn, 1e-6)._apply_output_gate(x, gate)
    for wrong in (hn // 2, 2 * hn):
        assert not torch.allclose(_attn(wrong, 1e-6)._apply_output_gate(x, gate), got)


def test_output_gate_calls_megatron_gate_with_the_normed_input(monkeypatch):
    seen = {}

    def parent(self, x, gate):
        seen.update(x=x.clone(), gate=gate)
        return x

    monkeypatch.setattr(SelfAttention, "_apply_output_gate", parent)
    x = torch.randn(2, 1, 6)
    gate = torch.randn(2, 1, 6)
    _attn(3, 1e-5)._apply_output_gate(x, gate)
    torch.testing.assert_close(seen["x"], _ref_rms(x.view(2, 1, 2, 3), 1e-5).view(2, 1, 6))
    assert seen["gate"] is gate
