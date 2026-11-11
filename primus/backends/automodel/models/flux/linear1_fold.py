###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX single-stream block's QKV and MLP up-projections as one 21504-wide GEMM (BFL's ``linear1``).

WHY:
  Every single-stream block projects its normalized tokens twice: the fused QKV GEMM (9216 wide)
  and ``proj_mlp`` (12288 wide). The forward GEMMs cost the same split or folded, but the weight
  gradient of one 21504-wide GEMM runs faster than the two narrower ones under sustained,
  power-capped load, and with one GEMM reading the normalized tokens their gradient needs no sum.
  Folding only pays if the gradient of the folded output is one contiguous buffer, so the q/k
  norm + RoPE backward and the GELU backward must write into column slices of it, which autograd
  cannot arrange across two separate consumers.

WHAT:
  - At model build, right after AutoModel's QKV fusion and before FSDP2 wraps the model, every
    single-stream block's ``attn.to_qkv`` and ``proj_mlp`` are replaced by one ``linear1``
    (their weights and biases concatenated, so the model is unchanged). The last block keeps its
    own projections when ``last_block.py`` gives it the image-only path.
  - ``primus_flux::single_core(z, bias, w_q, w_k, cos, sin, eps, heads, qkv_width)`` takes the
    ``linear1`` output and runs, on its column slices, ``qk_norm_rope.py``'s q/k RMSNorm + RoPE
    kernel, the aiter attention Primus-Turbo calls, and ``gelu_flydsl.py``'s GELU into the
    concat that ``proj_out`` reads. Its backward writes the GELU gradient and the q/k norm + RoPE
    gradient straight into one ``[B, S, 21504]`` buffer and returns the bias gradient from both
    kernels' partial sums, so ``linear1``'s two backward GEMMs run once, 21504 wide.

NUMERICS:
  The same kernels on the same values: outputs and gradients differ from the unfolded block only
  where the folded GEMMs pick other kernels (BF16 rounding).

REQUIRES:
  ``model.fuse_qkv_projections`` with ``model.compact_fused_qkv_projections``, and ``mlp_gelu.py``
  with ``qk_norm_rope.py``, on diffusers 0.40.0; uses ``adaln_flydsl.py``'s adaLN op when it is
  installed. Installs at priority 76, before ``block_outputs.py`` and ``last_block.py``. The three
  21504-wide GEMMs run at their tuned speed only with TunableOp entries for them.

Settings (module config, ``primus_perf:`` section):
    flux_fold_linear1: false     apply this patch
"""
from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_fold_linear1"
_QKV_WIDTH_ATTR = "_primus_qkv_width"
_HEAD_DIM = 128


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fold_linear1")


# --- the model --------------------------------------------------------------------------------


def _foldable(block) -> bool:
    from primus.backends.automodel.models.flux import qk_norm_rope

    attn, mlp, act = block.attn, getattr(block, "proj_mlp", None), getattr(block, "act_mlp", None)
    qkv = getattr(attn, "to_qkv", None)
    return (
        getattr(attn, "fused_projections", False)
        and not hasattr(attn, "to_q")
        and attn.head_dim == _HEAD_DIM
        and qkv is not None
        and qk_norm_rope._plain_linear(qkv)
        and mlp is not None
        and qk_norm_rope._plain_linear(mlp)
        and qkv.weight.dtype == mlp.weight.dtype
        and qkv.weight.device == mlp.weight.device
        and isinstance(act, torch.nn.GELU)
        and act.approximate == "tanh"
        and qk_norm_rope._norm_eps([attn.norm_q, attn.norm_k]) is not None
    )


def fold_blocks(transformer) -> int:
    """Replace each single-stream block's ``attn.to_qkv`` and ``proj_mlp`` by one ``linear1``."""
    from primus.backends.automodel.models.flux import last_block

    blocks = getattr(transformer, "single_transformer_blocks", None) or []
    folded = 0
    for i, block in enumerate(blocks):
        if last_block.is_enabled() and i == len(blocks) - 1:
            continue
        if not _foldable(block):
            continue
        qkv, mlp = block.attn.to_qkv, block.proj_mlp
        # skip_init: an initializing constructor would advance the RNG and so change every later random
        # draw (the noise and timesteps of every step), not just these weights, which are copied over.
        linear1 = torch.nn.utils.skip_init(
            torch.nn.Linear,
            qkv.in_features,
            qkv.out_features + mlp.out_features,
            bias=True,
            device=qkv.weight.device,
            dtype=qkv.weight.dtype,
        )
        with torch.no_grad():
            linear1.weight.copy_(torch.cat([qkv.weight, mlp.weight]))
            linear1.bias.copy_(torch.cat([qkv.bias, mlp.bias]))
        linear1.weight.requires_grad_(qkv.weight.requires_grad)
        linear1.bias.requires_grad_(qkv.bias.requires_grad)
        block.linear1 = linear1
        setattr(block, _QKV_WIDTH_ATTR, qkv.out_features)
        del block.attn.to_qkv
        del block.proj_mlp
        folded += 1
    return folded


def _install_fold() -> bool:
    from nemo_automodel._diffusers import auto_diffusion_pipeline as pipeline

    orig = pipeline._fuse_transformer_qkv_projections

    @functools.wraps(orig)
    def fuse(module, module_name, *, compact=False):
        fused = orig(module, module_name, compact=compact)
        if compact and hasattr(module, "single_transformer_blocks"):
            logger.info(
                "[PrimusFluxLinear1] folded proj_mlp into to_qkv in %d single-stream blocks of %s",
                fold_blocks(module),
                module_name,
            )
        return fused

    pipeline._fuse_transformer_qkv_projections = fuse
    return True


# --- the op -----------------------------------------------------------------------------------


def _attention_args(seq: int):
    # Primus-Turbo's flash_attn_func as the attention override calls it: non-causal, no dropout,
    # deterministic=False, its env-selected dQ atomics, and how_v3_bf16_cvt=0 on gfx950.
    return dict(
        dropout_p=0.0,
        softmax_scale=_HEAD_DIM**-0.5,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        bias=None,
        alibi_slopes=None,
    )


@torch.library.custom_op("primus_flux::single_core", mutates_args=(), device_types="cuda")
def single_core(
    z: torch.Tensor,
    bias: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    heads: int,
    qkv_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``cat([attention(qk_norm_rope(z[..., :qkv_width])), gelu_tanh(z[..., qkv_width:])], -1)`` as
    ``[B * S, H * D + MLP]`` for the ``linear1`` output ``z`` (its bias already added; ``bias`` is
    taken for its gradient). Also returns q, k, v, the attention output, its LSE and RNG state,
    only for the backward."""
    from primus.backends.automodel.models.flux import gelu_flydsl, qk_norm_rope

    batch, seq, width = z.shape
    n_tok, hd = batch * seq, heads * _HEAD_DIM
    shape = (batch, seq, heads, _HEAD_DIM)
    q, k, v = (torch.empty(shape, device=z.device, dtype=z.dtype) for _ in range(3))
    cos, sin = cos.contiguous(), sin.contiguous()
    grid = (
        qk_norm_rope.triton_cdiv(n_tok, qk_norm_rope._FWD_TOKENS),
        qk_norm_rope.triton_cdiv(heads, qk_norm_rope._BLOCK_H),
    )
    qk_norm_rope._get_kernels()[0][grid](
        z,
        q,
        k,
        v,
        w_q.contiguous(),
        w_k.contiguous(),
        cos,
        sin,
        n_tok,
        seq,
        seq,
        0,
        eps,
        width,
        H=heads,
        D=_HEAD_DIM,
        BLOCK_H=qk_norm_rope._BLOCK_H,
        TOKENS=qk_norm_rope._FWD_TOKENS,
        num_warps=4,
    )
    out, lse, _, rng = torch.ops.primus_turbo.attention_aiter_forward_impl(
        q,
        k,
        v,
        return_lse=True,
        return_softmax=False,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        sink=None,
        qkv_format="bshd",
        **_attention_args(seq),
    )
    concat = gelu_flydsl.gelu_into_cat(z.view(n_tok, width)[:, qkv_width:], out.view(n_tok, hd))
    return concat, q, k, v, out, lse, rng


@single_core.register_fake
def _(z, bias, w_q, w_k, cos, sin, eps, heads, qkv_width):
    batch, seq, width = z.shape
    shape = (batch, seq, heads, _HEAD_DIM)
    q, k, v = z.new_empty(shape), z.new_empty(shape), z.new_empty(shape)
    out, lse, _, rng = torch.ops.primus_turbo.attention_aiter_forward_impl(
        q,
        k,
        v,
        return_lse=True,
        return_softmax=False,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        sink=None,
        qkv_format="bshd",
        **_attention_args(seq),
    )
    return z.new_empty((batch * seq, heads * _HEAD_DIM + width - qkv_width)), q, k, v, out, lse, rng


@torch.library.custom_op("primus_flux::single_core_backward", mutates_args=(), device_types="cuda")
def single_core_backward(
    d_concat: torch.Tensor,
    z: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    rng: torch.Tensor,
    eps: float,
    heads: int,
    qkv_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(gradient of z, of the bias, of w_q, of w_k). Both kernels write into one contiguous gradient of z."""
    from primus_turbo.pytorch.ops.attention.attention_utils import (
        _resolve_is_v3_atomic_fp32_from_env,
    )

    from primus.backends.automodel.models.flux import (
        mlp_gelu,
        qk_norm_rope,
        qk_norm_rope_flydsl,
    )

    batch, seq, width = z.shape
    n_tok, hd = batch * seq, heads * _HEAD_DIM
    dz = torch.empty_like(z)
    d_concat = d_concat.view(n_tok, -1)
    db_mlp = mlp_gelu.gelu_backward_bias_into(
        d_concat[:, hd:], z.view(n_tok, width)[:, qkv_width:], dz.view(n_tok, width)[:, qkv_width:]
    )
    # As Primus-Turbo's AiterFlashAttnFunc.backward allocates them (bshd).
    dq, dk, dv = torch.ones_like(q), torch.empty_like(k), torch.empty_like(v)
    torch.ops.primus_turbo.attention_aiter_backward_impl(
        d_concat[:, :hd].view(batch, seq, heads, _HEAD_DIM),
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        deterministic=False,
        rng_state=rng,
        is_v3_atomic_fp32=_resolve_is_v3_atomic_fp32_from_env(),
        how_v3_bf16_cvt=0,
        qkv_format="bshd",
        **_attention_args(seq),
    )
    src, dsrc = z[..., :qkv_width], dz[..., :qkv_width]
    cos, sin = cos.contiguous(), sin.contiguous()
    colsum = qk_norm_rope._get_kernels()[2]
    if qk_norm_rope_flydsl.supported(src, None, cos, heads):
        dwq, dwk, db_qkv = qk_norm_rope_flydsl.backward_segment(
            dq, dk, dv, src, dsrc, w_q, w_k, cos, sin, 0, eps, heads, True, colsum
        )
    else:
        bwd = qk_norm_rope._get_kernels()[1]
        per = qk_norm_rope.triton_cdiv(n_tok, min(qk_norm_rope._BWD_PROGRAMS_BIAS, n_tok))
        n_prog, n_h = qk_norm_rope.triton_cdiv(n_tok, per), qk_norm_rope.triton_cdiv(
            heads, qk_norm_rope._BLOCK_H
        )
        part = torch.empty((n_prog * n_h, 2, _HEAD_DIM), device=z.device, dtype=torch.float32)
        bpart = torch.empty((n_prog, qkv_width), device=z.device, dtype=torch.float32)
        bwd[(n_prog, n_h)](
            src,
            dsrc,
            dq,
            dk,
            dv,
            w_q.contiguous(),
            w_k.contiguous(),
            cos,
            sin,
            part,
            bpart,
            n_tok,
            seq,
            seq,
            0,
            eps,
            per,
            width,
            H=heads,
            D=_HEAD_DIM,
            BLOCK_H=qk_norm_rope._BLOCK_H,
            BIAS=True,
            num_warps=4,
        )
        dw = part.sum(0)
        dwq, dwk = dw[0].to(w_q.dtype), dw[1].to(w_k.dtype)
        db_qkv = torch.empty(qkv_width, device=z.device, dtype=z.dtype)
        colsum[(qk_norm_rope.triton_cdiv(qkv_width, qk_norm_rope._COLSUM_BLOCK_C),)](
            bpart,
            db_qkv,
            n_prog,
            qkv_width,
            BLOCK_R=qk_norm_rope._COLSUM_BLOCK_R,
            BLOCK_C=qk_norm_rope._COLSUM_BLOCK_C,
            num_warps=4,
        )
    return dz, torch.cat([db_qkv, db_mlp]), dwq, dwk


@single_core_backward.register_fake
def _(d_concat, z, w_q, w_k, cos, sin, q, k, v, out, lse, rng, eps, heads, qkv_width):
    return torch.empty_like(z), z.new_empty((z.shape[-1],)), torch.empty_like(w_q), torch.empty_like(w_k)


def _setup_context(ctx, inputs, output):
    z, bias, w_q, w_k, cos, sin, eps, heads, qkv_width = inputs
    _, q, k, v, out, lse, rng = output
    ctx.mark_non_differentiable(q, k, v, out, lse, rng)
    ctx.save_for_backward(z, w_q, w_k, cos, sin, q, k, v, out, lse, rng)
    ctx.eps, ctx.heads, ctx.qkv_width, ctx.bias_dtype = eps, heads, qkv_width, bias.dtype


def _backward(ctx, d_concat, *_):
    z, w_q, w_k, cos, sin, q, k, v, out, lse, rng = ctx.saved_tensors
    dz, db, dwq, dwk = single_core_backward(
        d_concat, z, w_q, w_k, cos, sin, q, k, v, out, lse, rng, ctx.eps, ctx.heads, ctx.qkv_width
    )
    return dz, db.to(ctx.bias_dtype), dwq, dwk, None, None, None, None, None


single_core.register_autograd(_backward, setup_context=_setup_context)


# --- the single-stream block ------------------------------------------------------------------


def make_forward(orig):
    """The single-stream forward for blocks with ``linear1``; ``orig`` serves the others."""
    from primus.backends.automodel.models.flux import adaln_flydsl, qk_norm_rope

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        linear1 = getattr(self, "linear1", None)
        if linear1 is None:
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        if (
            joint_attention_kwargs
            or not isinstance(image_rotary_emb, (tuple, list))
            or hidden_states.dtype != torch.bfloat16
        ):
            raise NotImplementedError(
                "[PrimusFluxLinear1] a folded single-stream block needs BF16 tokens and the "
                "rotary embedding, and takes no joint_attention_kwargs"
            )
        attn, norm = self.attn, self.norm
        text_seq_len = encoder_hidden_states.shape[1]
        emb = norm.linear(norm.silu(temb))
        shift, scale, gate = emb.chunk(3, dim=1)
        split_residual = getattr(type(self), adaln_flydsl._PATCHED_ATTR, False) and adaln_flydsl.supported(
            encoder_hidden_states, hidden_states, scale, shift
        )
        if split_residual:
            r_txt, r_img, norm_hidden_states, _ = adaln_flydsl.adaln_single(
                encoder_hidden_states, hidden_states, scale, shift, norm.norm.eps
            )
        else:
            residual = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            norm_hidden_states = norm.norm(residual) * (1 + scale[:, None]) + shift[:, None]
        batch, seq, _ = norm_hidden_states.shape
        # The op returns the bias gradient from its kernels' partial sums.
        z = F.linear(norm_hidden_states, linear1.weight, linear1.bias.detach())
        cos, sin = image_rotary_emb
        qkv_width = getattr(self, _QKV_WIDTH_ATTR)
        concat = single_core(
            z,
            linear1.bias,
            attn.norm_q.weight,
            attn.norm_k.weight,
            cos,
            sin,
            qk_norm_rope._norm_eps([attn.norm_q, attn.norm_k]),
            attn.heads,
            qkv_width,
        )[0]
        gated = gate.unsqueeze(1) * self.proj_out(concat.view(batch, seq, -1))
        if split_residual:
            return r_txt + gated[:, :text_seq_len], r_img + gated[:, text_seq_len:]
        hidden_states = residual + gated
        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]

    return forward


def install() -> bool:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    from primus.backends.automodel.models.flux import mlp_gelu, qk_norm_rope

    if getattr(FluxSingleTransformerBlock, _PATCHED_ATTR, False):
        return True
    if not (mlp_gelu.installed() and qk_norm_rope.is_enabled()):
        logger.warning("[PrimusFluxLinear1] needs flux_fused_gelu and flux_fused_qk_norm_rope; not applied.")
        return False
    _install_fold()
    FluxSingleTransformerBlock.forward = make_forward(FluxSingleTransformerBlock.forward)
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxLinear1] single-stream blocks: to_qkv and proj_mlp as one linear1, with q/k norm + RoPE, "
        "attention and GELU in one op whose backward writes one contiguous linear1 gradient."
    )
    return True
