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
  - Primus-Turbo's ``flux_single_core(z, bias, w_q, w_k, cos, sin, eps, heads, qkv_width)`` takes the
    ``linear1`` output and runs, on its column slices, the q/k RMSNorm + RoPE of
    ``flux_qk_norm_rope``, aiter's flash attention as the attention override calls it, and the GELU of
    ``flux_gelu_linear`` into the concat that ``proj_out`` reads. Its backward writes the GELU
    gradient and the q/k norm + RoPE gradient straight into one ``[B, S, 21504]`` buffer and returns
    the bias gradient from both kernels' partial sums, so ``linear1``'s two backward GEMMs run once,
    21504 wide.

NUMERICS:
  The same kernels on the same values: outputs and gradients differ from the unfolded block only
  where the folded GEMMs pick other kernels (BF16 rounding).

REQUIRES:
  Primus-Turbo with its FLUX ops. ``model.fuse_qkv_projections`` with
  ``model.compact_fused_qkv_projections``, and ``mlp_gelu.py``
  with ``qk_norm_rope.py``, on diffusers 0.40.0; uses ``adaln_flydsl.py``'s adaLN op when it is
  installed. Installs at priority 76, before ``block_outputs.py`` and ``last_block.py``. The three
  21504-wide GEMMs run at their tuned speed only with TunableOp entries for them.
  The folded blocks make the attention call of ``primus_turbo.nondeterministic_attention``
  themselves, so the fold needs that override and declines where its attention would differ from
  the unfolded blocks': with the FP8 attention override, and under context parallelism, which
  the folded blocks' direct kernel call cannot follow. It also declines when a linear swap
  (``model.transformer_engine_linear``) would replace ``linear1`` after the fold.

Settings (module config, ``primus_perf:`` section):
    flux_fold_linear1: false     apply this patch
"""
from __future__ import annotations

import functools
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from primus.backends.automodel import options
from primus.backends.automodel.models.flux.plain_linear import is_plain_linear
from primus.backends.automodel.models.flux.turbo_ops import flux_ops

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
        and is_plain_linear(qkv)
        and mlp is not None
        and is_plain_linear(mlp)
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


# --- the single-stream block ------------------------------------------------------------------


def make_forward(orig, ops):
    """The single-stream forward for blocks with ``linear1``; ``orig`` serves the others."""
    from primus_turbo.pytorch.kernels.flux.adaln_impl import adaln_single_shape_error

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
        if not is_plain_linear(linear1):
            raise NotImplementedError(
                f"[PrimusFluxLinear1] linear1 was replaced by a {type(linear1).__name__} after the fold; "
                "the folded block computes it from its weight and would skip that layer's forward"
            )
        attn, norm = self.attn, self.norm
        text_seq_len = encoder_hidden_states.shape[1]
        emb = norm.linear(norm.silu(temb))
        shift, scale, gate = emb.chunk(3, dim=1)
        split_residual = (
            getattr(type(self), adaln_flydsl._PATCHED_ATTR, False)
            and adaln_single_shape_error(encoder_hidden_states, hidden_states, scale, shift) is None
        )
        if split_residual:
            r_txt, r_img, norm_hidden_states = ops.flux_adaln_single(
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
        concat = ops.flux_single_core(
            z,
            linear1.bias,
            attn.norm_q.weight,
            attn.norm_k.weight,
            cos,
            sin,
            qk_norm_rope._norm_eps([attn.norm_q, attn.norm_k]),
            attn.heads,
            qkv_width,
        )
        gated = gate.unsqueeze(1) * self.proj_out(concat.view(batch, seq, -1))
        if split_residual:
            return r_txt + gated[:, :text_seq_len], r_img + gated[:, text_seq_len:]
        hidden_states = residual + gated
        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]

    return forward


def unsupported_reason(*, context_parallel: bool, linear_swap: bool) -> Optional[str]:
    """Why the fold cannot serve this run, or None if it can (see REQUIRES above)."""
    from primus.backends.automodel.attention import fp8, nondeterministic

    if not nondeterministic.is_enabled() or fp8.is_enabled():
        return (
            "needs primus_turbo.nondeterministic_attention without primus_turbo.fp8_attention: "
            "the folded blocks make that override's attention call"
        )
    if context_parallel:
        return "context parallelism is on, and the folded blocks call the attention kernel on local tokens"
    if linear_swap:
        return (
            "model.transformer_engine_linear is on, and its linear swap would replace linear1 after the fold"
        )
    return None


def install(*, context_parallel: bool = False, linear_swap: bool = False) -> bool:
    """``context_parallel`` and ``linear_swap`` say what the run's config builds."""
    reason = unsupported_reason(context_parallel=context_parallel, linear_swap=linear_swap)
    if reason is not None:
        logger.warning("[PrimusFluxLinear1] %s; not applied.", reason)
        return False

    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    from primus.backends.automodel.models.flux import mlp_gelu, qk_norm_rope

    if getattr(FluxSingleTransformerBlock, _PATCHED_ATTR, False):
        return True
    if not (mlp_gelu.installed() and qk_norm_rope.is_enabled()):
        logger.warning("[PrimusFluxLinear1] needs flux_fused_gelu and flux_fused_qk_norm_rope; not applied.")
        return False
    ops = flux_ops("PrimusFluxLinear1")
    if ops is None:
        return False
    _install_fold()
    FluxSingleTransformerBlock.forward = make_forward(FluxSingleTransformerBlock.forward, ops)
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxLinear1] single-stream blocks: to_qkv and proj_mlp as one linear1, with q/k norm + RoPE, "
        "attention and GELU in one op whose backward writes one contiguous linear1 gradient."
    )
    return True
