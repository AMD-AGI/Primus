###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU equivalence: the chunked linear cross-entropy (``fused_ce_mode`` 3) vs Megatron's, and its Mamba wiring.

Stock path (TP=1): ``logits = x @ W^T`` in bf16 ->
``tensor_parallel.vocab_parallel_cross_entropy`` (fp32 upcast inside).
ZAYA1's wiring is tested in ``test_zaya1_model.py``.
"""

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from primus.backends.megatron.core.fusions import chunked_linear_cross_entropy as mod
from primus.backends.megatron.core.fusions.chunked_linear_cross_entropy import (
    chunked_ce_blocker,
    chunked_linear_cross_entropy,
)
from primus.backends.megatron.patches import mamba_fused_ce_patches as mamba


@pytest.fixture(autouse=True)
def _fresh_compile_caches():
    # Dynamo keeps a function's compiled graphs for the whole process, so the
    # shapes of earlier tests would use up the per-chunk helpers' recompile
    # limit and send every later test to eager.
    torch._dynamo.reset()
    yield


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def tp_group():
    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        dist.init_process_group("gloo", rank=0, world_size=1)
        created = True
    group = dist.new_group([0], backend="gloo")
    yield group
    if created:
        dist.destroy_process_group()


def _stock(hidden, weight, labels, tp_group, logit_scale=None):
    from megatron.core import tensor_parallel

    logits = torch.matmul(hidden, weight.t())  # [s, b, V] bf16
    if logit_scale is not None:
        logits = logits * logit_scale
    loss = tensor_parallel.vocab_parallel_cross_entropy(
        logits, labels.transpose(0, 1).contiguous(), tp_group=tp_group
    )
    return loss.transpose(0, 1).contiguous()


def _run(fn, hidden0, weight0, labels, mask):
    hidden = hidden0.clone().requires_grad_(True)
    weight = weight0.clone().requires_grad_(True)
    loss = fn(hidden, weight, labels)
    # loss_mask handled outside the model, as in pretrain_gpt.loss_func.
    total = (loss.float() * mask).sum() / mask.sum()
    total.backward()
    return loss.detach(), hidden.grad.float(), weight.grad.float()


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("chunk", [7, 64, 1000])
@pytest.mark.parametrize("logit_scale", [None, 0.5])
def test_chunked_ce_matches_stock(tp_group, chunk, logit_scale, compiled):
    torch.manual_seed(0)
    s, b, h, vocab = 16, 6, 64, 1531
    hidden0 = (torch.randn(s, b, h) * 0.5).to(torch.bfloat16)
    weight0 = (torch.randn(vocab, h) * 0.05).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (b, s))
    mask = (torch.rand(b, s) > 0.2).float()

    ref = _run(lambda x, w, y: _stock(x, w, y, tp_group, logit_scale), hidden0, weight0, labels, mask)
    out = _run(
        lambda x, w, y: chunked_linear_cross_entropy(
            x, w, y, chunk_tokens=chunk, logit_scale=logit_scale, compiled=compiled
        ),
        hidden0,
        weight0,
        labels,
        mask,
    )

    assert out[0].shape == ref[0].shape == (b, s)
    assert out[0].dtype == torch.float32
    torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=1e-5)
    # dx: identical math per element modulo fp32 summation order before the
    # bf16 rounding of dlogits; dW: fp32 accumulation across chunks vs one GEMM.
    assert _rel(out[1], ref[1]) < 1e-2
    assert _rel(out[2], ref[2]) < 1e-2
    torch.testing.assert_close(out[1], ref[1], rtol=2e-2, atol=2e-4)
    torch.testing.assert_close(out[2], ref[2], rtol=2e-2, atol=2e-4)


@pytest.mark.parametrize("compiled", [False, True])
def test_chunked_ce_out_of_range_labels_match_stock(tp_group, compiled):
    torch.manual_seed(1)
    s, b, h, vocab = 8, 3, 32, 257
    hidden0 = torch.randn(s, b, h).to(torch.bfloat16)
    weight0 = (torch.randn(vocab, h) * 0.1).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (b, s))
    labels[0, 0] = -100
    labels[1, 3] = vocab + 5
    mask = torch.ones(b, s)
    ref = _run(lambda x, w, y: _stock(x, w, y, tp_group), hidden0, weight0, labels, mask)
    out = _run(
        lambda x, w, y: chunked_linear_cross_entropy(x, w, y, chunk_tokens=5, compiled=compiled),
        hidden0,
        weight0,
        labels,
        mask,
    )
    torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=1e-5)
    assert _rel(out[1], ref[1]) < 1e-2
    assert _rel(out[2], ref[2]) < 1e-2


def test_chunked_ce_fused_kernels_compile_and_match_eager():
    """The fused per-chunk CE math matches the eager formulas (and does compile)."""
    torch.manual_seed(2)
    n, vocab = 37, 1031
    logits = (torch.randn(n, vocab) * 3).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (n,))
    valid = torch.rand(n) > 0.1
    g = torch.rand(n)
    fwd, bwd = mod._fused_kernels()
    lse, loss = fwd(logits, labels, valid)
    d = bwd(logits, lse, labels, valid, g)
    assert fwd.compiled_ok() and bwd.compiled_ok()
    lf = logits.float()
    ref_lse = torch.logsumexp(lf, dim=-1)
    m = lf.amax(dim=-1)
    tgt = torch.where(valid, lf.gather(1, labels[:, None]).squeeze(1), m)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss, ref_lse - tgt, rtol=1e-5, atol=1e-5)
    p = torch.softmax(lf, dim=-1)
    p[torch.arange(n)[valid], labels[valid]] -= 1
    torch.testing.assert_close(d.float(), (p * g[:, None]).to(torch.bfloat16).float(), rtol=1e-2, atol=1e-6)


def _config(**overrides):
    values = dict(
        cross_entropy_loss_fusion=False, cross_entropy_fusion_impl="native", sequence_parallel=False
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_blocker_rules():
    assert chunked_ce_blocker(_config(), 1) is None
    assert chunked_ce_blocker(_config(), 1, mode=3) is None
    # A fused cross-entropy that was asked for runs, whatever fused_ce_mode says.
    fused = _config(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="te")
    for mode in (None, 0, 3):
        assert chunked_ce_blocker(fused, 1, mode) == "cross_entropy_loss_fusion is set (te)"
    assert chunked_ce_blocker(_config(), 1, None) == "fused_ce_mode is unset"
    assert chunked_ce_blocker(_config(), 1, 0) == "fused_ce_mode is 0"
    for mode in (1, 2):
        assert chunked_ce_blocker(_config(), 1, mode) == (
            f"fused_ce_mode {mode} selects FLA's kernel, which this model doesn't wire up"
        )
    assert chunked_ce_blocker(_config(), 2) == "it needs TP=1 without sequence parallel"
    assert chunked_ce_blocker(_config(sequence_parallel=True), 1) == "it needs TP=1 without sequence parallel"


def _mamba_model(tp=1, **config):
    values = dict(tensor_model_parallel_size=tp, use_mup=False, mup_output_mult=1.0)
    values.update(config)
    return SimpleNamespace(config=_config(**values))


def test_mamba_mode_3_setup():
    model = _mamba_model()
    assert mamba._setup_chunked_ce(model, SimpleNamespace()) == 3
    assert (model._fused_ce_chunk_tokens, model._fused_ce_compile) == (4096, False)
    args = SimpleNamespace(fused_ce_chunk_tokens=128, fused_ce_compile=True)
    assert mamba._setup_chunked_ce(model, args) == 3
    assert (model._fused_ce_chunk_tokens, model._fused_ce_compile) == (128, True)
    assert mamba._setup_chunked_ce(model, SimpleNamespace(fused_ce_compile=1)) == 3
    assert model._fused_ce_compile is True
    # Megatron's cross-entropy runs where the chunked one can't, as in ZAYA1.
    assert mamba._setup_chunked_ce(_mamba_model(tp=2), args) == 0
    assert mamba._setup_chunked_ce(_mamba_model(cross_entropy_loss_fusion=True), args) == 0
    with pytest.raises(ValueError, match="fused_ce_chunk_tokens"):
        mamba._setup_chunked_ce(model, SimpleNamespace(fused_ce_chunk_tokens=0))
    with pytest.raises(ValueError, match="fused_ce_compile must be true or false"):
        mamba._setup_chunked_ce(model, SimpleNamespace(fused_ce_compile="off"))


@pytest.mark.parametrize("mup_output_mult", [1.0, 0.5])
def test_mamba_mode_3_matches_megatrons_loss(tp_group, mup_output_mult):
    """Per-token loss and gradients as the stock MambaModel path computes them, MuP scale included."""
    torch.manual_seed(3)
    s, b, h, vocab = 12, 4, 32, 523
    hidden0 = (torch.randn(s, b, h) * 0.5).to(torch.bfloat16)
    weight0 = (torch.randn(vocab, h) * 0.05).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (b, s))
    mask = (torch.rand(b, s) > 0.2).float()
    model = _mamba_model(use_mup=True, mup_output_mult=mup_output_mult)
    model._fused_ce_mode = mamba._setup_chunked_ce(model, SimpleNamespace(fused_ce_chunk_tokens=7))

    def loss(x, w, y):
        model.output_layer = SimpleNamespace(weight=w)
        return mamba._fused_cross_entropy_loss(model, x, y, None)

    scale = None if mup_output_mult == 1.0 else mup_output_mult
    ref = _run(lambda x, w, y: _stock(x, w, y, tp_group, scale), hidden0, weight0, labels, mask)
    out = _run(loss, hidden0, weight0, labels, mask)
    assert out[0].shape == (b, s)
    torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=1e-5)
    assert _rel(out[1], ref[1]) < 1e-2
    assert _rel(out[2], ref[2]) < 1e-2


@pytest.mark.parametrize("mode", [0, 3])
def test_mamba_init_text_sets_up_mode_3(monkeypatch, mode):
    """The source the patch splices into MambaModel.__init__ runs mode 3's setup, and leaves 0 alone."""
    import megatron.training as training

    namespace = {}
    exec("def init(self):\n        " + mamba._INIT_NEW, namespace)
    monkeypatch.setattr(
        training, "get_args", lambda: SimpleNamespace(fused_ce_mode=mode, fused_ce_chunk_tokens=64)
    )
    model = _mamba_model()
    model.setup_embeddings_and_output_layer = lambda: None
    namespace["init"](model)
    assert model._fused_ce_mode == mode
    assert model._use_fused_cross_entropy is (mode == 3)
    assert getattr(model, "_fused_ce_chunk_tokens", None) == (64 if mode == 3 else None)
