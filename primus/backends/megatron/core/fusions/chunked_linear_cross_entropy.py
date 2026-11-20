###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Chunked output projection + cross-entropy (TP=1), ``fused_ce_mode`` 3.

The stock path materializes ``logits = x @ W^T`` for every token
([s*b, vocab] in bf16), then ``vocab_parallel_cross_entropy`` upcasts it to
fp32 and keeps an fp32 softmax for backward. For ZAYA1's 262,272-token
vocabulary at mbs 248 x seq 128 that is ~16.6 GB bf16 + ~33 GB fp32 +
temporaries.

``ChunkedLinearCrossEntropy`` walks the tokens in chunks. Forward keeps only
the fp32 log-sum-exp per token. Backward recomputes each chunk's logits,
forms ``g * (softmax - onehot)`` in fp32, rounds it to the activation dtype
(as autograd does on the stock path), and runs the dx / dW GEMMs per chunk.
Each chunk's dW leaves its GEMM in the activation dtype and is summed into an
fp32 buffer, which is rounded to the weight dtype once at the end (the stock
path's single dW GEMM rounds once). No tensor larger than ``chunk x vocab`` is
ever live.

Semantics match ``LanguageModule.compute_language_model_loss`` with
``vocab_parallel_cross_entropy`` at TP=1 and label_smoothing=0, including
out-of-range labels (their max-shifted predicted logit is 0, so the loss is
``lse - max``, exactly as the stock kernel computes it).
Returns per-token loss ``[b, s]``. The caller applies ``loss_mask``.

ZAYA1 and the Mamba models select it the same way. ``fused_ce_mode`` is the
Mamba models' loss switch (``fla_runtime_patches.py``), where 1 and 2 are
FLA's kernels; 3 is this one, in ``mamba_fused_ce_patches.py`` and in
``zaya1_model.py``, and unset or 0 is Megatron's cross-entropy.
``fused_ce_chunk_tokens`` (default 4096) is the tokens per chunk, and
``fused_ce_compile`` (default off) fuses each chunk's CE math with
torch.compile. ``chunked_ce_blocker`` holds the rules both models apply.
"""

from typing import Optional

import torch
from torch import Tensor

from primus.backends.megatron.core.utils import maybe_compile

__all__ = [
    "CHUNKED_CE_MODE",
    "DEFAULT_CHUNK_TOKENS",
    "ChunkedLinearCrossEntropy",
    "check_chunk_tokens",
    "chunked_ce_blocker",
    "chunked_linear_cross_entropy",
    "mup_logit_scale",
]

CHUNKED_CE_MODE = 3
DEFAULT_CHUNK_TOKENS = 4096


def chunked_ce_blocker(config, tp_size: int, mode: Optional[int] = CHUNKED_CE_MODE) -> Optional[str]:
    """Why ``fused_ce_mode`` ``mode`` leaves the loss to Megatron's cross-entropy, or None when this one runs."""
    if getattr(config, "cross_entropy_loss_fusion", False):
        # A fused cross-entropy was asked for: Megatron's native or TE kernel,
        # or Turbo's, which only runs with this flag set.
        impl = getattr(config, "cross_entropy_fusion_impl", "native")
        return f"cross_entropy_loss_fusion is set ({impl})"
    if mode != CHUNKED_CE_MODE:
        if mode in (1, 2):
            return f"fused_ce_mode {mode} selects FLA's kernel, which this model doesn't wire up"
        return "fused_ce_mode is unset" if mode is None else f"fused_ce_mode is {mode}"
    if tp_size != 1 or getattr(config, "sequence_parallel", False):
        return "it needs TP=1 without sequence parallel"
    return None


def check_chunk_tokens(tokens) -> int:
    if type(tokens) is not int or tokens < 1:
        raise ValueError(f"fused_ce_chunk_tokens must be a positive integer, got {tokens!r}.")
    return tokens


def mup_logit_scale(config) -> Optional[float]:
    """The MuP output scale ``LanguageModule._scale_logits`` puts on the logits, or None."""
    if getattr(config, "use_mup", False) and config.mup_output_mult != 1.0:
        return float(config.mup_output_mult)
    return None


def _chunk_logits(x_c: Tensor, weight: Tensor, logit_scale: Optional[float]) -> Tensor:
    # Same rounding as the stock path: bf16 GEMM output, optional MuP scale on
    # the bf16 logits.
    logits = torch.matmul(x_c, weight.t())
    if logit_scale is not None:
        logits = logits * logit_scale
    return logits


# Per-chunk CE math. Compiled (fused_ce_compile), each reads the bf16
# chunk logits once or twice and writes only [chunk] fp32 stats (forward) or
# the bf16 dlogits (backward), instead of ~6 eager passes over a
# [chunk, vocab] fp32 tensor.
def _ce_fwd_stats(logits: Tensor, labels: Tensor, valid: Tensor):
    # Same op order as VocabParallelCrossEntropy: shift by the row max, take the
    # target from the shifted logits (0 for an out-of-range label), exp, sum, log.
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
        _FUSED["fns"] = (maybe_compile(_ce_fwd_stats), maybe_compile(_ce_bwd_dlogits))
    return _FUSED["fns"]


class ChunkedLinearCrossEntropy(torch.autograd.Function):
    """loss[t] = logsumexp(x[t] @ W^T) - (x[t] @ W^T)[label[t]] in fp32."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        labels: Tensor,
        chunk: int,
        logit_scale: Optional[float],
        compiled: bool,
    ):
        # x: [N, H] activation dtype, weight: [V, H], labels: [N] int64.
        n_tok = x.shape[0]
        vocab = weight.shape[0]
        valid = (labels >= 0) & (labels < vocab)
        safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

        loss = torch.empty(n_tok, dtype=torch.float32, device=x.device)
        lse = torch.empty(n_tok, dtype=torch.float32, device=x.device)
        fwd_stats = _fused_kernels()[0] if compiled else _ce_fwd_stats
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
        ctx.compiled = compiled
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
        bwd_dlogits = _fused_kernels()[1] if ctx.compiled else _ce_bwd_dlogits

        for start in range(0, n_tok, chunk):
            end = min(start + chunk, n_tok)
            x_c = x[start:end]
            g = grad_loss[start:end]
            # softmax - onehot, scaled by the upstream grad, in fp32, then
            # rounded to the activation dtype.
            logits = _chunk_logits(x_c, weight, logit_scale)
            d = bwd_dlogits(logits, lse[start:end], safe_labels[start:end], valid[start:end], g)
            del logits
            if logit_scale is not None:
                # d(scaled logits)/d(raw logits); stock autograd does this in bf16.
                d = d * logit_scale
            if need_dx:
                torch.matmul(d, weight, out=dx[start:end])
            if need_dw:
                dw_acc.add_(torch.matmul(d.t(), x_c))
            del d

        dw = dw_acc.to(weight.dtype) if need_dw else None
        return dx, dw, None, None, None, None


def chunked_linear_cross_entropy(
    hidden_states: Tensor,
    weight: Tensor,
    labels: Tensor,
    chunk_tokens: int = 4096,
    logit_scale: Optional[float] = None,
    compiled: bool = False,
) -> Tensor:
    """hidden_states [s, b, h], weight [V, h], labels [b, s] -> loss [b, s] (fp32).

    Tokens are processed in ``[s, b]`` order so ``hidden_states`` is a free
    view; only the small label / loss tensors are transposed. ``chunk_tokens``
    and ``compiled`` are ``fused_ce_chunk_tokens`` and ``fused_ce_compile``.
    """
    s, b, h = hidden_states.shape
    x = hidden_states.reshape(s * b, h)
    labels_sb = labels.transpose(0, 1).reshape(s * b)
    loss = ChunkedLinearCrossEntropy.apply(x, weight, labels_sb, int(chunk_tokens), logit_scale, compiled)
    return loss.view(s, b).transpose(0, 1).contiguous()
