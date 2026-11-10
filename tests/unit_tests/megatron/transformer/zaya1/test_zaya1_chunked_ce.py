###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU equivalence: chunked output-projection + CE vs the stock ZAYA1 path.

Stock path (TP=1): ``logits = x @ W^T`` in bf16 ->
``tensor_parallel.vocab_parallel_cross_entropy`` (fp32 upcast inside).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist

from primus.backends.megatron.core.models.zaya1.zaya1_chunked_ce import (
    chunked_ce_enabled,
    chunked_linear_cross_entropy,
)


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


@pytest.mark.parametrize("compile_flag", ["0", "1"])
@pytest.mark.parametrize("chunk", [7, 64, 1000])
@pytest.mark.parametrize("logit_scale", [None, 0.5])
def test_chunked_ce_matches_stock(tp_group, chunk, logit_scale, compile_flag, monkeypatch):
    monkeypatch.setenv("ZAYA_CHUNKED_CE_COMPILE", compile_flag)
    torch.manual_seed(0)
    s, b, h, vocab = 16, 6, 64, 1531
    hidden0 = (torch.randn(s, b, h) * 0.5).to(torch.bfloat16)
    weight0 = (torch.randn(vocab, h) * 0.05).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (b, s))
    mask = (torch.rand(b, s) > 0.2).float()

    ref = _run(lambda x, w, y: _stock(x, w, y, tp_group, logit_scale), hidden0, weight0, labels, mask)
    out = _run(
        lambda x, w, y: chunked_linear_cross_entropy(x, w, y, chunk_tokens=chunk, logit_scale=logit_scale),
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


@pytest.mark.parametrize("compile_flag", ["0", "1"])
def test_chunked_ce_out_of_range_labels_match_stock(tp_group, compile_flag, monkeypatch):
    monkeypatch.setenv("ZAYA_CHUNKED_CE_COMPILE", compile_flag)
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
        lambda x, w, y: chunked_linear_cross_entropy(x, w, y, chunk_tokens=5), hidden0, weight0, labels, mask
    )
    torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=1e-5)
    assert _rel(out[1], ref[1]) < 1e-2
    assert _rel(out[2], ref[2]) < 1e-2


def test_chunked_ce_fused_kernels_compile_and_match_eager():
    """The fused per-chunk CE math matches the eager formulas (and does compile)."""
    import warnings

    from primus.backends.megatron.core.models.zaya1 import zaya1_chunked_ce as mod

    torch.manual_seed(2)
    n, vocab = 37, 1031
    logits = (torch.randn(n, vocab) * 3).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (n,))
    valid = torch.rand(n) > 0.1
    g = torch.rand(n)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # _maybe_compile warns on eager fallback
        fwd, bwd = mod._fused_kernels()
        lse, loss = fwd(logits, labels, valid)
        d = bwd(logits, lse, labels, valid, g)
    lf = logits.float()
    ref_lse = torch.logsumexp(lf, dim=-1)
    m = lf.amax(dim=-1)
    tgt = torch.where(valid, lf.gather(1, labels[:, None]).squeeze(1), m)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss, ref_lse - tgt, rtol=1e-5, atol=1e-5)
    p = torch.softmax(lf, dim=-1)
    p[torch.arange(n)[valid], labels[valid]] -= 1
    torch.testing.assert_close(d.float(), (p * g[:, None]).to(torch.bfloat16).float(), rtol=1e-2, atol=1e-6)


def test_chunked_ce_env_toggle(monkeypatch):
    monkeypatch.delenv("ZAYA_CHUNKED_CE", raising=False)
    assert chunked_ce_enabled()
    monkeypatch.setenv("ZAYA_CHUNKED_CE", "0")
    assert not chunked_ce_enabled()
    monkeypatch.setenv("ZAYA_CHUNKED_CE", "1")
    assert chunked_ce_enabled()
