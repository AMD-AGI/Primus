###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Chunked output-projection + cross-entropy for ZAYA1 (TP=1).

The stock path materializes ``logits = x @ W^T`` for every token
([s*b, 262272] in bf16), then ``vocab_parallel_cross_entropy`` upcasts it to
fp32 and keeps an fp32 softmax for backward. At mbs 248 x seq 128 that is
~16.6 GB bf16 + ~33 GB fp32 + temporaries.

``ChunkedLinearCrossEntropy`` walks the tokens in chunks. Forward keeps only
the fp32 log-sum-exp per token. Backward recomputes each chunk's logits,
forms ``g * (softmax - onehot)`` in fp32, rounds it to the activation dtype
(as autograd does on the stock path), and runs the dx / dW GEMMs per chunk.
dW is accumulated in fp32 and rounded once at the end. No tensor larger than
``chunk x vocab`` is ever live.

Semantics match ``LanguageModule.compute_language_model_loss`` with
``vocab_parallel_cross_entropy`` at TP=1 and label_smoothing=0, including
out-of-range labels (their max-shifted predicted logit is 0, so the loss is
``lse - max``, exactly as the stock kernel computes it).
Returns per-token loss ``[b, s]``. The caller applies ``loss_mask``.
"""

import os
from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "ChunkedLinearCrossEntropy",
    "chunked_linear_cross_entropy",
    "chunked_ce_enabled",
    "chunked_ce_chunk_tokens",
    "chunked_ce_compile_enabled",
]


def chunked_ce_enabled() -> bool:
    """``ZAYA_CHUNKED_CE`` (default 1). Set 0 to restore the stock logits + CE path."""
    return os.environ.get("ZAYA_CHUNKED_CE", "1").strip().lower() not in ("0", "false", "off", "no", "")


def chunked_ce_chunk_tokens() -> int:
    """``ZAYA_CHUNKED_CE_TOKENS`` (default 4096): tokens per chunk."""
    try:
        n = int(os.environ.get("ZAYA_CHUNKED_CE_TOKENS", "4096"))
    except ValueError:
        n = 4096
    return max(1, n)


def chunked_ce_compile_enabled() -> bool:
    """``ZAYA_CHUNKED_CE_COMPILE`` (default 1): fuse each chunk's CE math with torch.compile."""
    return os.environ.get("ZAYA_CHUNKED_CE_COMPILE", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
        "",
    )


def _chunk_logits(x_c: Tensor, weight: Tensor, logit_scale: Optional[float]) -> Tensor:
    # Same rounding as the stock path: bf16 GEMM output, optional MuP scale on
    # the bf16 logits.
    logits = torch.matmul(x_c, weight.t())
    if logit_scale is not None:
        logits = logits * logit_scale
    return logits


def _chunk_logits_fp32(x_c: Tensor, weight: Tensor, logit_scale: Optional[float]) -> Tensor:
    return _chunk_logits(x_c, weight, logit_scale).float()


# Fused per-chunk CE math. Each reads the bf16 chunk logits once or twice and
# writes only [chunk] fp32 stats (forward) or the bf16 dlogits (backward),
# instead of ~6 eager passes over a [chunk, vocab] fp32 tensor.
def _ce_fwd_stats(logits: Tensor, labels: Tensor, valid: Tensor):
    lf = logits.float()
    m = lf.amax(dim=-1, keepdim=True)
    shifted = lf - m
    log_sum = shifted.exp().sum(dim=-1).log()
    tgt = shifted.gather(1, labels.unsqueeze(1)).squeeze(1)
    tgt = torch.where(valid, tgt, torch.zeros_like(tgt))
    return log_sum + m.squeeze(1), log_sum - tgt


def _ce_bwd_dlogits(logits: Tensor, lse: Tensor, labels: Tensor, valid: Tensor, g: Tensor):
    p = (logits.float() - lse.unsqueeze(1)).exp()
    cols = torch.arange(logits.shape[1], device=logits.device)
    onehot = (cols.unsqueeze(0) == labels.unsqueeze(1)) & valid.unsqueeze(1)
    d = (p - onehot.to(p.dtype)) * g.unsqueeze(1)
    return d.to(logits.dtype)


_FUSED = {}


def _fused_kernels():
    """(fwd_stats, bwd_dlogits) compiled once per process; eager fallback inside."""
    if "fns" not in _FUSED:
        from primus.backends.megatron.core.models.zaya1.zaya1_modules import (
            _maybe_compile,
        )

        _FUSED["fns"] = (_maybe_compile(_ce_fwd_stats, True), _maybe_compile(_ce_bwd_dlogits, True))
    return _FUSED["fns"]


class ChunkedLinearCrossEntropy(torch.autograd.Function):
    """loss[t] = logsumexp(x[t] @ W^T) - (x[t] @ W^T)[label[t]] in fp32."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, labels: Tensor, chunk: int, logit_scale: Optional[float]):
        # x: [N, H] activation dtype, weight: [V, H], labels: [N] int64.
        n_tok = x.shape[0]
        vocab = weight.shape[0]
        valid = (labels >= 0) & (labels < vocab)
        safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

        loss = torch.empty(n_tok, dtype=torch.float32, device=x.device)
        lse = torch.empty(n_tok, dtype=torch.float32, device=x.device)
        fused = chunked_ce_compile_enabled()
        ctx.fused = fused
        if fused:
            fwd_stats, _ = _fused_kernels()
            for start in range(0, n_tok, chunk):
                end = min(start + chunk, n_tok)
                logits = _chunk_logits(x[start:end], weight, logit_scale)
                lse_c, loss_c = fwd_stats(logits, safe_labels[start:end], valid[start:end])
                lse[start:end] = lse_c
                loss[start:end] = loss_c
                del logits
            ctx.save_for_backward(x, weight, safe_labels, valid, lse)
            ctx.chunk = chunk
            ctx.logit_scale = logit_scale
            return loss

        for start in range(0, n_tok, chunk):
            end = min(start + chunk, n_tok)
            lf = _chunk_logits_fp32(x[start:end], weight, logit_scale)
            # Same op order as VocabParallelCrossEntropy: shift by the row max
            # in place, take the target from the shifted logits (0 for an
            # out-of-range label), exp in place, sum, log.
            m = lf.amax(dim=-1)
            lf.sub_(m.unsqueeze(1))
            tgt = lf.gather(1, safe_labels[start:end].unsqueeze(1)).squeeze(1)
            tgt = torch.where(valid[start:end], tgt, torch.zeros_like(tgt))
            log_sum = lf.exp_().sum(dim=-1).log_()
            lse[start:end] = log_sum + m
            loss[start:end] = log_sum - tgt
            del lf

        ctx.save_for_backward(x, weight, safe_labels, valid, lse)
        ctx.chunk = chunk
        ctx.logit_scale = logit_scale
        return loss

    @staticmethod
    def backward(ctx, grad_loss: Tensor):
        x, weight, safe_labels, valid, lse = ctx.saved_tensors
        chunk, logit_scale = ctx.chunk, ctx.logit_scale
        n_tok = x.shape[0]
        grad_loss = grad_loss.float()

        need_dx = ctx.needs_input_grad[0]
        need_dw = ctx.needs_input_grad[1]
        dx = torch.empty_like(x) if need_dx else None
        dw_acc = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_dw else None

        for start in range(0, n_tok, chunk):
            end = min(start + chunk, n_tok)
            x_c = x[start:end]
            g = grad_loss[start:end]
            if ctx.fused:
                _, bwd_dlogits = _fused_kernels()
                logits = _chunk_logits(x_c, weight, logit_scale)
                d = bwd_dlogits(logits, lse[start:end], safe_labels[start:end], valid[start:end], g)
                del logits
                if logit_scale is not None:
                    d = d * logit_scale
                if need_dx:
                    torch.matmul(d, weight, out=dx[start:end])
                if need_dw:
                    dw_acc.add_(torch.matmul(d.t(), x_c))
                del d
                continue
            # softmax - onehot, scaled by the upstream grad (fp32, in place).
            p = _chunk_logits_fp32(x_c, weight, logit_scale)
            p.sub_(lse[start:end].unsqueeze(1)).exp_()
            sub = valid[start:end].to(p.dtype).unsqueeze(1)
            p.scatter_add_(1, safe_labels[start:end].unsqueeze(1), -sub)
            p.mul_(g.unsqueeze(1))
            if logit_scale is not None:
                # d(scaled logits)/d(raw logits); stock autograd does this in bf16.
                d = p.to(x.dtype) * logit_scale
            else:
                d = p.to(x.dtype)
            del p
            if need_dx:
                torch.matmul(d, weight, out=dx[start:end])
            if need_dw:
                dw_acc.add_(torch.matmul(d.t(), x_c))
            del d

        dw = dw_acc.to(weight.dtype) if need_dw else None
        return dx, dw, None, None, None


def chunked_linear_cross_entropy(
    hidden_states: Tensor,
    weight: Tensor,
    labels: Tensor,
    chunk_tokens: Optional[int] = None,
    logit_scale: Optional[float] = None,
) -> Tensor:
    """hidden_states [s, b, h], weight [V, h], labels [b, s] -> loss [b, s] (fp32).

    Tokens are processed in ``[s, b]`` order so ``hidden_states`` is a free
    view; only the small label / loss tensors are transposed.
    """
    s, b, h = hidden_states.shape
    x = hidden_states.reshape(s * b, h)
    labels_sb = labels.transpose(0, 1).reshape(s * b)
    chunk = chunk_tokens if chunk_tokens is not None else chunked_ce_chunk_tokens()
    loss = ChunkedLinearCrossEntropy.apply(x, weight, labels_sb, int(chunk), logit_scale)
    return loss.view(s, b).transpose(0, 1).contiguous()
