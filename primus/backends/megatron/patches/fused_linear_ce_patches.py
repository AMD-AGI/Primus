###############################################################################
# Chunked linear cross-entropy for Megatron GPTModel (avoid full-logits OOM).
#
# PROBLEM: Megatron's stock GPTModel._postprocess materializes the FULL logits
# tensor [seq, batch, vocab] before cross-entropy. For a large vocab (Qwen
# 151936) and long sequence (e.g. 64k), that single tensor is tens of GB and
# OOMs -- even though a 1.5B model's weights/activations are tiny. This is the
# exact wall we hit validating ODC/LB-Mini on DeepSeek-R1-Distill-Qwen-1.5B.
#
# FIX (verl/Liger spirit): split along the sequence dim and compute
# logits+CE per chunk under activation checkpointing, so the full [seq, vocab]
# logits is NEVER resident -- peak logits memory is one chunk only. This lets
# TP=1 / DP=8 (the layout ODC requires) run 64k-token sequences without OOM.
#
# ZERO IMPACT on stock Megatron:
#   * Pure monkey-patch in the Primus layer; third-party Megatron source is NOT
#     touched.
#   * Gated by enable_fused_linear_ce (yaml) or FUSED_LINEAR_CE=1 (env).
#     DEFAULT OFF -> the original full-logits _postprocess runs byte-for-byte.
#   * Even when ON, only the training-with-labels path is intercepted;
#     inference / no-labels (generation) / MTP / non-post-process stages all
#     fall through to the original implementation.
#   * Numerically equivalent: per-token CE is independent across positions, so
#     chunk-then-concatenate yields the identical [b, s] per-token loss.
###############################################################################

import os

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

_PATCHED = False


def _fused_ce_enabled(args) -> bool:
    """Switch can come from the YAML arg or, for experiments, the env var."""
    return bool(getattr(args, "enable_fused_linear_ce", False)) or (
        os.environ.get("FUSED_LINEAR_CE", "0") == "1"
    )


def _chunk_size() -> int:
    """Per-chunk sequence length. Smaller = less peak logits mem, more recompute."""
    return int(os.environ.get("FUSED_CE_CHUNK", "0") or 0) or 4096


class _ChunkedLinearCE(torch.autograd.Function):
    """verl-style fused linear cross-entropy.

    forward: compute the LM loss in sequence chunks WITHOUT building an autograd
        graph and WITHOUT saving per-chunk logits -> the full [s, vocab] logits
        is never materialized (peak = one chunk).
    backward: recompute each chunk's logits and obtain grads via a SINGLE
        torch.autograd.grad call per chunk, inside ONE backward invocation.

    Why not torch.utils.checkpoint: its recompute is driven by an unpack hook
    that runs INTERLEAVED with FSDP2/ODC's backward communication. Under ODC the
    ranks run DIFFERENT micro-batch counts, so that interleaving desyncs and
    DEADLOCKS (the iter-3 hang). Doing the recompute here -- a plain local
    autograd.grad with NO collective -- keeps each rank independent in backward.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, model, chunk):
        ctx.model = model
        ctx.chunk = chunk
        ctx.save_for_backward(hidden, weight, labels)
        seq = hidden.size(0)
        parts = []
        with torch.no_grad():
            for i in range(0, seq, chunk):
                logits = model._scale_logits(torch.matmul(hidden[i : i + chunk], weight.t()))
                parts.append(
                    model.compute_language_model_loss(labels[:, i : i + chunk].contiguous(), logits)
                )
        return torch.cat(parts, dim=1)  # [b, s]

    @staticmethod
    def backward(ctx, grad_out):  # grad_out [b, s]
        hidden, weight, labels = ctx.saved_tensors
        model = ctx.model
        chunk = ctx.chunk
        seq = hidden.size(0)
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight)
        for i in range(0, seq, chunk):
            h_c = hidden[i : i + chunk].detach().requires_grad_(True)
            w = weight.detach().requires_grad_(True)
            with torch.enable_grad():
                logits = model._scale_logits(torch.matmul(h_c, w.t()))
                loss_c = model.compute_language_model_loss(
                    labels[:, i : i + chunk].contiguous(), logits
                )
            g_h, g_w = torch.autograd.grad(loss_c, (h_c, w), grad_out[:, i : i + chunk].contiguous())
            grad_hidden[i : i + chunk] = g_h
            grad_weight = grad_weight + g_w
        return grad_hidden, grad_weight, None, None, None


def _chunked_lm_loss(model, hidden_states, labels):
    """Chunked linear+CE that never materializes the full [s, b, vocab] logits.

    hidden_states: [s, b, h]   labels: [b, s]   -> per-token loss [b, s]
    """
    # PREFERRED PATH (ODC + DiffMicro): use the full output weight that ODC's
    # train-loop hook already all-gathered ONCE at the minibatch boundary
    # (pre_minibatch_start). Calling full_tensor() here -- inside the per-micro
    # batch forward -- would issue a DTensor collective whose call-count differs
    # across ranks under DiffMicro, and deadlock. The cached tensor is a leaf
    # with requires_grad=True; its .grad accumulates across all this rank's
    # micro-batches and is reduce-scattered back to the sharded param at
    # pre_optimizer_step (see odc_torch_fsdp2_patches._odc_reduce_output_grad).
    cached = getattr(model, "_odc_cached_output_weight", None)
    if cached is not None:
        return _ChunkedLinearCE.apply(hidden_states, cached, labels, model, _chunk_size())

    # FALLBACK (no ODC, or SameMicro): same-count collectives are safe.
    if model.share_embeddings_and_output_weights:
        ow = model.shared_embedding_or_output_weight()
    else:
        ow = model.output_layer.weight
    if hasattr(ow, "full_tensor"):
        ow_full = ow.full_tensor()
    else:
        ow_full = ow.clone()
    return _ChunkedLinearCE.apply(hidden_states, ow_full, labels, model, _chunk_size())


def _install_fused_ce_patch():
    global _PATCHED
    if _PATCHED:
        return
    import megatron.core.models.gpt.gpt_model as gpt_mod

    GPTModel = gpt_mod.GPTModel
    if getattr(GPTModel._postprocess, "_fused_ce_hooked", False):
        _PATCHED = True
        return
    orig_postprocess = GPTModel._postprocess

    def postprocess_with_fused_ce(self, *args, **kwargs):
        # forward() calls _postprocess() with all-kwargs, but support positional
        # too: signature is (hidden_states, input_ids, position_ids, labels, ...).
        hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        labels = kwargs.get("labels", args[3] if len(args) > 3 else None)
        inference_context = kwargs.get("inference_context")
        in_inference = inference_context is not None and not self.training

        if (
            labels is not None
            and hidden_states is not None
            and not in_inference
            and getattr(self, "post_process", True)
            and not getattr(self.config, "mtp_num_layers", 0)
        ):
            return _chunked_lm_loss(self, hidden_states, labels)
        # Everything else -> stock path, unchanged.
        return orig_postprocess(self, *args, **kwargs)

    postprocess_with_fused_ce._fused_ce_hooked = True
    GPTModel._postprocess = postprocess_with_fused_ce
    _PATCHED = True
    log_rank_0(
        f"[FusedCE] patched GPTModel._postprocess: chunked linear+CE "
        f"(chunk={_chunk_size()}); full [seq, vocab] logits no longer materialized."
    )


@register_patch(
    "megatron.fused_linear_ce",
    backend="megatron",
    phase="before_train",
    description="Chunked linear cross-entropy for GPTModel to avoid full-logits OOM (large vocab + long seq).",
    condition=lambda ctx: _fused_ce_enabled(get_args(ctx)),
)
def patch_fused_linear_ce(ctx: PatchContext):
    log_rank_0(
        "[FusedCE] enable_fused_linear_ce=true -> installing chunked linear+CE "
        "(avoids materializing full [seq, vocab] logits; numerically equivalent)."
    )
    _install_fused_ce_patch()


__all__ = ["patch_fused_linear_ce"]
