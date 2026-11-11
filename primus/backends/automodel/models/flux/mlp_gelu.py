###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX MLP's up-projection and GELU as one op, with the GELU written by one FlyDSL pass.

WHY:
  Both FLUX block types run their MLP as an up-projection GEMM followed by a separate tanh-GELU
  kernel that reads the pre-activation back from memory, and the single-stream block then copies
  the attention output and the GELU output into one buffer for ``proj_out``. In the backward,
  autograd computes the GELU gradient in one pass and sums it for the bias gradient in another.

WHAT:
  ``primus_flux::gelu_linear(x, w, b, prefix)`` returns ``cat([prefix, gelu_tanh(x @ w.T + b)])`` and
  the pre-activation. The GEMM runs as ``torch.addmm``; one FlyDSL pass (``gelu_flydsl.py``) then
  copies ``prefix`` and writes the GELU of the pre-activation straight into the concatenated buffer,
  so the single-stream block's ``torch.cat([attn_output, mlp_hidden_states])`` needs no pass of its
  own. Shapes the kernel does not cover compute the same thing unfused. The backward is the one
  autograd runs for the unfused graph: ``gelu_backward`` on the saved pre-activation, the two GEMMs
  and the bias sum. The single-stream block's forward and the dual-stream blocks' ``FeedForward``
  call it; anything that does not match (another activation, dtype or dropout) keeps diffusers'
  forward.

  With ``flux_fused_gelu_bias_grad`` the backward's ``gelu_backward`` is a Triton kernel that also
  sums each tile's columns, so the bias gradient does not read the GELU gradient back. It evaluates
  Inductor's ``gelu_backward`` expression in the same order and sums the BF16-rounded gradient, as
  the unfused graph does; only the order of the FP32 column sums differs.

REQUIRES:
  FlyDSL; without it the patch is not applied. The single-stream forward re-implements diffusers
  0.40.0's ``FluxSingleTransformerBlock.forward`` (checked by source hash; any other source keeps
  diffusers' forward there).

Settings (module config, ``primus_perf:`` section):
    flux_fused_gelu: false            apply this patch
    flux_fused_gelu_bias_grad: false  compute the bias gradient in the GELU-backward kernel
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import logging
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_fused_gelu"
# sha256 of diffusers 0.40.0's FluxSingleTransformerBlock.forward, which _install_single_block
# re-implements with the MLP fused; any other source keeps diffusers' forward.
SINGLE_FORWARD_SHA256 = "57350a6dab842b40e0d677b9c42794c17e45907ad45991d36dba69d89096a37b"

_BIAS_BLOCK_R = 8
_BIAS_BLOCK_C = 256
_BIAS_PARTS = 128

_state = {"bias_grad": False}


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fused_gelu")


def installed() -> bool:
    """Whether ``install()`` has run: the dual-stream ``FeedForward`` calls ``gelu_linear``."""
    from diffusers.models.attention import FeedForward

    return getattr(FeedForward, _PATCHED_ATTR, False)


def fusable(x: torch.Tensor, linear: torch.nn.Linear, act_is_tanh_gelu: bool) -> bool:
    """Whether ``gelu_linear`` serves ``act(linear(x))``."""
    return (
        act_is_tanh_gelu
        and linear.bias is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and linear.weight.dtype == torch.bfloat16
        and linear.bias.dtype == torch.bfloat16
    )


# --- the op --------------------------------------------------------------------------------------


@torch.library.custom_op("primus_flux::gelu_linear", mutates_args=(), device_types="cuda")
def gelu_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, prefix: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cat([prefix, gelu_tanh(x @ weight.T + bias)], dim=1), x @ weight.T + bias) for 2-D BF16 x."""
    from primus.backends.automodel.models.flux import gelu_flydsl

    x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
    m, n = x.shape[0], weight.shape[0]
    width = 0 if prefix is None else prefix.shape[1]
    aux = torch.addmm(bias, x, weight.t())
    if gelu_flydsl.supported(m, width, n):
        return gelu_flydsl.gelu_into_cat(aux, prefix), aux
    out = torch.empty((m, width + n), device=x.device, dtype=x.dtype)
    if prefix is not None:
        out[:, :width].copy_(prefix)
    out[:, width:].copy_(F.gelu(aux, approximate="tanh"))
    return out, aux


@gelu_linear.register_fake
def _(x, weight, bias, prefix):
    width = 0 if prefix is None else prefix.shape[1]
    return x.new_empty((x.shape[0], width + weight.shape[0])), x.new_empty((x.shape[0], weight.shape[0]))


def _setup_context(ctx, inputs, output):
    x, weight, _, prefix = inputs
    _, aux = output
    # The pre-activation is returned only so the backward can save it; nothing downstream reads it.
    ctx.mark_non_differentiable(aux)
    ctx.save_for_backward(x, weight, aux)
    ctx.width = 0 if prefix is None else prefix.shape[1]
    ctx.has_prefix = prefix is not None


@triton.jit
def _gelu_bwd_bias_kernel(
    g_ptr,
    h_ptr,
    dh_ptr,
    part_ptr,
    n_rows,
    n_cols,
    g_stride,
    rows_per_prog,
    h_stride,
    dh_stride,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < n_cols
    row_end = tl.minimum((pid_r + 1) * rows_per_prog, n_rows)
    acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
    for row0 in range(pid_r * rows_per_prog, row_end, BLOCK_R):
        rows = (row0 + tl.arange(0, BLOCK_R)).to(tl.int64)
        mask = (rows < row_end)[:, None] & col_mask[None, :]
        g = tl.load(g_ptr + rows[:, None] * g_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        h = tl.load(h_ptr + rows[:, None] * h_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        # Inductor's gelu_backward(approximate="tanh"), operation for operation.
        h2 = h * h
        t = libdevice.tanh((h + h2 * h * 0.044715) * 0.7978845608028654)
        dh = g * ((t + 1.0) * 0.5 + (h * 0.5) * (1.0 - t * t) * ((h2 * 0.134145 + 1.0) * 0.7978845608028654))
        dh = dh.to(dh_ptr.dtype.element_ty)
        tl.store(dh_ptr + rows[:, None] * dh_stride + cols[None, :], dh, mask=mask)
        acc += dh.to(tl.float32)
    tl.store(part_ptr + pid_r * n_cols + cols, tl.sum(acc, axis=0), mask=col_mask)


@triton.jit
def _colsum_kernel(src_ptr, out_ptr, n_rows, n_cols, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    # out[c] = sum_r src[r, c] for the FP32 partials; faster here than torch's column sum.
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < n_cols
    acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
    for r0 in range(0, n_rows, BLOCK_R):
        rows = (r0 + tl.arange(0, BLOCK_R)).to(tl.int64)
        acc += tl.load(
            src_ptr + rows[:, None] * n_cols + cols[None, :],
            mask=(rows < n_rows)[:, None] & col_mask[None, :],
            other=0.0,
        )
    tl.store(out_ptr + cols, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty), mask=col_mask)


@torch.library.custom_op("primus_flux::gelu_backward_bias", mutates_args=(), device_types="cuda")
def gelu_backward_bias(grad: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(gelu_backward(grad, aux, "tanh"), its column sums) for a 2-D BF16 grad with unit column stride."""
    if grad.stride(1) != 1:
        grad = grad.contiguous()
    aux = aux.contiguous()
    grad_h = torch.empty_like(aux)
    return grad_h, gelu_backward_bias_into(grad, aux, grad_h)


def gelu_backward_bias_into(grad: torch.Tensor, aux: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    """Writes gelu_backward(grad, aux, "tanh") into ``grad_h`` and returns its column sums. All three are
    2-D BF16 with unit column stride; their row strides may differ (column slices of wider buffers)."""
    n_rows, n_cols = aux.shape
    rows_per_prog = triton.cdiv(triton.cdiv(n_rows, _BIAS_PARTS), _BIAS_BLOCK_R) * _BIAS_BLOCK_R
    n_parts = triton.cdiv(n_rows, rows_per_prog)
    parts = torch.empty((n_parts, n_cols), device=aux.device, dtype=torch.float32)
    _gelu_bwd_bias_kernel[(n_parts, triton.cdiv(n_cols, _BIAS_BLOCK_C))](
        grad,
        aux,
        grad_h,
        parts,
        n_rows,
        n_cols,
        grad.stride(0),
        rows_per_prog,
        aux.stride(0),
        grad_h.stride(0),
        BLOCK_R=_BIAS_BLOCK_R,
        BLOCK_C=_BIAS_BLOCK_C,
        num_warps=8,
    )
    grad_bias = torch.empty(n_cols, device=aux.device, dtype=aux.dtype)
    _colsum_kernel[(triton.cdiv(n_cols, 64),)](
        parts, grad_bias, n_parts, n_cols, BLOCK_R=32, BLOCK_C=64, num_warps=4
    )
    return grad_bias


@gelu_backward_bias.register_fake
def _(grad, aux):
    return torch.empty_like(aux), aux.new_empty((aux.shape[1],))


def _backward(ctx, grad_out, _grad_aux):
    x, weight, aux = ctx.saved_tensors
    grad = grad_out[:, ctx.width :]
    if _state["bias_grad"]:
        grad_h, grad_bias = gelu_backward_bias(grad, aux)
    else:
        grad_h = torch.ops.aten.gelu_backward(grad, aux, approximate="tanh")
        grad_bias = grad_h.sum(0)
    grad_prefix = grad_out[:, : ctx.width] if ctx.has_prefix else None
    return grad_h @ weight, grad_h.t() @ x, grad_bias, grad_prefix


gelu_linear.register_autograd(_backward, setup_context=_setup_context)


# --- FLUX modules -------------------------------------------------------------------------------


def _install_feed_forward() -> None:
    from diffusers.models.activations import GELU
    from diffusers.models.attention import FeedForward

    orig = FeedForward.forward

    @functools.wraps(orig)
    def forward(self, hidden_states, *args, **kwargs):
        net = self.net
        act = net[0] if len(net) == 3 else None
        if (
            args
            or kwargs
            or not isinstance(act, GELU)
            or not isinstance(net[1], torch.nn.Dropout)
            or (self.training and net[1].p > 0)
            or not isinstance(net[2], torch.nn.Linear)
            or not fusable(hidden_states, act.proj, act.approximate == "tanh")
        ):
            return orig(self, hidden_states, *args, **kwargs)
        lead = hidden_states.shape[:-1]
        mlp, _ = gelu_linear(
            hidden_states.reshape(-1, hidden_states.shape[-1]), act.proj.weight, act.proj.bias, None
        )
        return net[2](mlp.view(*lead, mlp.shape[-1]))

    FeedForward.forward = forward


def _install_single_block() -> bool:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    orig = FluxSingleTransformerBlock.forward
    digest = hashlib.sha256(inspect.getsource(orig).encode()).hexdigest()
    if digest != SINGLE_FORWARD_SHA256:
        logger.warning(
            "[PrimusFluxFusedGelu] FluxSingleTransformerBlock.forward is not diffusers 0.40.0's; "
            "single-stream blocks keep the unfused MLP."
        )
        return False

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        act = self.act_mlp
        if not fusable(
            hidden_states, self.proj_mlp, isinstance(act, torch.nn.GELU) and act.approximate == "tanh"
        ):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        # diffusers 0.40.0's forward, with act_mlp(proj_mlp(x)) and the cat fused into one op.
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        batch, seq, dim = norm_hidden_states.shape
        concat, _ = gelu_linear(
            norm_hidden_states.reshape(batch * seq, dim),
            self.proj_mlp.weight,
            self.proj_mlp.bias,
            attn_output.reshape(batch * seq, -1),
        )
        hidden_states = gate.unsqueeze(1) * self.proj_out(concat.view(batch, seq, -1))
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states

    FluxSingleTransformerBlock.forward = forward
    return True


def install() -> bool:
    from diffusers.models.attention import FeedForward

    if getattr(FeedForward, _PATCHED_ATTR, False):
        return True
    if importlib.util.find_spec("flydsl") is None:
        logger.warning("[PrimusFluxFusedGelu] FlyDSL is not installed; not applied.")
        return False
    _state["bias_grad"] = options.flag(f"{_SECTION}.flux_fused_gelu_bias_grad")
    _install_feed_forward()
    single = _install_single_block()
    setattr(FeedForward, _PATCHED_ATTR, True)
    logger.info(
        f"[PrimusFluxFusedGelu] FlyDSL GELU (+ concat) after the up-projection GEMM: dual-stream FeedForward"
        f"{' and single-stream blocks' if single else ''}"
        f"{'; bias gradient in the GELU-backward kernel' if _state['bias_grad'] else ''}."
    )
    return True
