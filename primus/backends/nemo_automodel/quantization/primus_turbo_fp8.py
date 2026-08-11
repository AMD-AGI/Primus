###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus-Turbo low-precision (FP8 / MXFP4) linear swap for the NeMo AutoModel
diffusion recipe.

Why:
  Config-only Transformer Engine FP8 does not train Wan2.2 on rocm/primus:v26.3
  (TE 2.12.0.dev0, ROCm 7.2): per-tensor delayed/current has no hipBLASLt fp8
  backward GEMM on gfx950, and MXFP8 has no bias support. Primus-Turbo's
  ``Float8Linear`` (aiter-backed ``gemm_fp8``, autograd fwd+bwd, bias applied
  OUTSIDE the fp8 GEMM) sidesteps both gaps and is the SAME library that powers
  the verified FLUX fp8 diffusion training on MI355X. The MXFP4 path reuses the
  same seam via ``primus_turbo.pytorch.ops.gemm_fp4`` (autograd fwd+bwd,
  MX_BLOCKWISE / E2M1 / block-32, HIPBLASLT backend) wrapped in a local
  ``Float4Linear`` (Primus-Turbo 0.2.0 ships no Float4Linear module).

What this does (NO Automodel fork):
  Monkeypatches ``nemo_automodel._diffusers.auto_diffusion_pipeline
  ._replace_linear_with_transformer_engine`` so that the existing config seam
  (``model.transformer_engine_linear=true``) swaps ``torch.nn.Linear`` ->
  ``primus_turbo.pytorch.modules.Float8Linear`` (FP8) or the local
  ``Float4Linear`` (MXFP4) instead of TE Linear. The swap runs on the built
  transformer BEFORE FSDP2 wrapping, mirroring the TE path (weight/bias copy,
  requires_grad preserved). We ALWAYS apply AutoModel's
  ``_is_fp8_training_safe_linear`` skip-list for stability (keep time/text embed,
  norm-modulation, final proj, and non-16-aligned Linears in bf16); the MXFP4
  path adds a stricter 128-alignment guard (see ``_is_fp4_training_safe_linear``).

Activation & tuning (env, so no Automodel source or config schema changes):
  # --- FP8 (Float8Linear, aiter gemm_fp8) ---
  PRIMUS_TURBO_FP8=1                      enable the FP8 swap (default off = no-op)
  PRIMUS_TURBO_FP8_GRANULARITY=TENSORWISE ROWWISE|TENSORWISE|BLOCKWISE|MX_BLOCKWISE
  PRIMUS_TURBO_FP8_FORMAT=E4M3            E4M3|E5M2|HYBRID
  # --- TE-native MXFP4 (te.pytorch.Linear + MXFP4BlockScaling autocast) ------
  # Highest precedence. Requires a TE-native image (TransformerEngine rebuilt
  # with MXFP4 recipe support). TE's recipe applies a Random Hadamard
  # Transform (RHT) + native fused HIP cast/transpose kernel and dispatches the
  # GEMM to AITER a4w4 -- a configuration shown to converge.
  # NOTE: RHT is NOT the differentiator. primus_turbo's gemm_fp4 ALSO applies RHT
  # -- a deterministic 32x32 Hadamard x fixed sign vector (kernels/quantization/
  # hadamard_transform.py), on exactly the two transposed Wgrad operands (a_t,
  # grad_out_t), matching the NVFP4 prescription. Yet the primus_turbo MXFP4
  # Wgrad still explodes on Wan (isolated via PRIMUS_TURBO_FP4_
  # BACKWARD=wgrad_fp8). So the real difference is
  # TE's fused cast/transpose + a4w4 Wgrad SCALE HANDLING vs primus_turbo's, not
  # the presence of RHT. This path tests whether TE's Wgrad avoids the explosion.
  PRIMUS_TE_MXFP4=1                       enable the TE-native MXFP4 swap+autocast
  # --- MXFP4 (Float4Linear, gemm_fp4) --- takes precedence over FP8 if set ---
  PRIMUS_TURBO_FP4=1                      enable the MXFP4 swap (default off = no-op)
  PRIMUS_TURBO_FP4_KEEP_SENSITIVE=1      keep Wan condition_embedder.* (timestep/
                                         text/AdaLN) Linears in bf16 (default on)
  PRIMUS_TURBO_FP4_SR=off                stochastic rounding scope: off|grad|all.
                                         primus_turbo 0.2.0 hardcodes use_sr=False
                                         everywhere (the divergent 'FP4 w/o SR'
                                         regime);
                                         grad = SR on backward gradient quant,
                                         all = SR on fwd (act+weight) + bwd grad.
  # --- MXFP4 convergence scaffolding -----------------------------------------
  # Extra scaffolding from a converging MXFP4 ablation ("NVFP4-aligned"): pure
  # MXFP4 backward + gradient SR + a band of early/late transformer blocks kept
  # in tensorwise FP8. Same primus_turbo kernels -- convergence came from the
  # scaffolding around the kernel, not the kernel.
  PRIMUS_TURBO_FP4_SENSITIVE_LAYERS=0    keep a band of transformer blocks in a
                                         higher precision than MXFP4 (default off)
  PRIMUS_TURBO_FP4_SENSITIVE_START=2     # of leading blocks in the band (first N)
  PRIMUS_TURBO_FP4_SENSITIVE_END=8       # of trailing blocks in the band (last N)
  PRIMUS_TURBO_FP4_SENSITIVE_PRECISION=tw_fp8  band precision: tw_fp8|bf16
                                         (tw_fp8 = primus_turbo Float8Linear
                                          tensorwise; bf16 = keep torch.nn.Linear)
  PRIMUS_TURBO_FP4_BACKWARD=mxfp4        backward GEMM precision:
                                         mxfp4|mxfp4_pad|fp8.
                                         mxfp4 = pure MXFP4 fwd+bwd (default,
                                         matches the converging Run C); fp8 =
                                         hybrid MXFP4 fwd / tensorwise-FP8 bwd
                                         (mirrors mxfp4_backward_precision: fp8;
                                         no activation-memory savings -- saves
                                         bf16 x/W and requantizes to FP8 in bwd).
                                         mxfp4_pad = THE FIX: pure MXFP4 fwd+bwd
                                         with the token/contraction dim zero-padded
                                         to %256 so the AITER a4w4 Wgrad (contracts
                                         over tokens) is numerically correct
                                         instead of returning ~1e7x garbage on
                                         non-%256 token counts (Wan cross-attn
                                         k/v: text_seq*batch). Zeros are exact;
                                         keeps the tuned preshuffled AITER path.
  # (MXFP4 is fixed to MX_BLOCKWISE + E2M1_X2 + block_size 32 + E8M0 scale by the
  #  installed primus_turbo 0.2.0; no granularity/format knobs are exposed.)

Recommended config pairing (keeps TE autocast OFF, drives only the swap):
  model.transformer_engine_linear: true
  model.transformer_engine_fp8: false

MXFP4 kernel constraints (installed primus_turbo 0.2.0, HIPBLASLT FP4 backend):
  The forward + both backward GEMMs are NT-layout, bf16, and (accounting for FP4
  2-values/byte packing and the transposed grad GEMMs) require the logical
  M (tokens), N (out_features) and K (in_features) to ALL be multiples of 128.
  in_features/out_features are enforced at swap time via the skip-list; the token
  dimension M is a runtime property (batch*seq) checked with a one-time warning.
"""
from __future__ import annotations

import logging
import os
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}


def is_enabled() -> bool:
    """Whether the Primus-Turbo FP8 swap should be installed."""
    return os.getenv("PRIMUS_TURBO_FP8", "0") in _TRUTHY


def is_fp4_enabled() -> bool:
    """Whether the Primus-Turbo MXFP4 swap should be installed.

    Takes precedence over FP8 when both are set (see ``install``).
    """
    return os.getenv("PRIMUS_TURBO_FP4", "0") in _TRUTHY


def is_te_mxfp4_enabled() -> bool:
    """Whether the TE-native MXFP4 swap should be installed.

    Highest precedence (see ``install``). Only meaningful on a TE-native image
    where TransformerEngine was rebuilt with MXFP4 recipe support. On a
    baseline image TE exposes the MXFP4BlockScaling *name*
    but none of the execution machinery, so this path fails fast at install.
    """
    return os.getenv("PRIMUS_TE_MXFP4", "0") in _TRUTHY


def _resolve_fp8_config():
    """Build a Float8QuantConfig from env (defaults: TENSORWISE + E4M3 + DYNAMIC)."""
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScalingGranularity,
    )

    gran_name = os.getenv("PRIMUS_TURBO_FP8_GRANULARITY", "TENSORWISE").upper()
    fmt_name = os.getenv("PRIMUS_TURBO_FP8_FORMAT", "E4M3").upper()
    try:
        granularity = getattr(ScalingGranularity, gran_name)
    except AttributeError as exc:
        raise ValueError(
            f"PRIMUS_TURBO_FP8_GRANULARITY={gran_name!r} invalid; expected one of "
            "TENSORWISE, ROWWISE, BLOCKWISE, MX_BLOCKWISE"
        ) from exc
    try:
        fmt = getattr(Format, fmt_name)
    except AttributeError as exc:
        raise ValueError(
            f"PRIMUS_TURBO_FP8_FORMAT={fmt_name!r} invalid; expected one of E4M3, E5M2, HYBRID"
        ) from exc
    return Float8QuantConfig(format=fmt, granularity=granularity)


# --------------------------------------------------------------------------- #
# MXFP4 (Float4Linear on top of primus_turbo.pytorch.ops.gemm_fp4)            #
# --------------------------------------------------------------------------- #
# Alignment required by the installed primus_turbo 0.2.0 HIPBLASLT FP4 GEMM
# (fwd + both bwd GEMMs, NT layout, FP4 packs 2 vals/byte). See module docstring.
_FP4_ALIGN = 128

# Wan-specific precision-sensitive projections kept in bf16 by default under
# MXFP4. AutoModel's FP8-safe predicate uses FLUX names (time_text_embed.,
# norm_out.) that do NOT match Wan's module tree, so without this the timestep
# embedder, text embedder and AdaLN modulation (time_proj) would all be swapped
# to FP4. FP4 is much more aggressive than FP8, and diffusion timestep/text/
# modulation conditioning is the classic "keep in high precision" set; these are
# only 5 of 405 eligible Linears, so the throughput cost is negligible. Toggle
# with PRIMUS_TURBO_FP4_KEEP_SENSITIVE (default on) to A/B full coverage.
_FP4_SENSITIVE_PREFIXES = ("condition_embedder.",)


def _fp4_keep_sensitive_bf16() -> bool:
    return os.getenv("PRIMUS_TURBO_FP4_KEEP_SENSITIVE", "1") in _TRUTHY


_FP4_SR_SCOPES = {"off", "grad", "all"}


def _fp4_sr_scope() -> str:
    """Stochastic-rounding scope for the MXFP4 quantization.

    primus_turbo 0.2.0's ``FP4GemmMXFunction`` hardcodes ``use_sr=False`` at every
    FP4 quant site -- the "FP4 without stochastic rounding" regime that can make
    native-TE MXFP4 diverge, where enabling SR recovers convergence. This
    env opts into SR via our local ``Float4Linear`` (NO primus_turbo/Automodel
    fork):
      off  (default) -> stock behavior, SR nowhere (calls primus_turbo.gemm_fp4)
      grad           -> SR on the backward gradient quant (debias dgrad/wgrad)
      all            -> SR on forward (act+weight) AND backward gradient quant
    """
    scope = os.getenv("PRIMUS_TURBO_FP4_SR", "off").lower()
    if scope in _TRUTHY:
        scope = "grad"
    if scope not in _FP4_SR_SCOPES:
        raise ValueError(
            f"PRIMUS_TURBO_FP4_SR={scope!r} invalid; expected one of off, grad, all"
        )
    return scope


# --------------------------------------------------------------------------- #
# MXFP4 convergence scaffolding                                                #
# --------------------------------------------------------------------------- #
# Backward precision -----------------------------------------------------------
# mxfp4     : pure MXFP4 dgrad + wgrad (the FLUX Run-C recipe).
# fp8       : hybrid -- BOTH dgrad + wgrad in tensorwise FP8 (MXFP4 fwd).
# dgrad_fp8 : ISOLATION -- dgrad (grad wrt input) in FP8, wgrad stays MXFP4.
# wgrad_fp8 : ISOLATION -- wgrad (grad wrt weight) in FP8, dgrad stays MXFP4.
# The two isolation modes pin down which backward GEMM drives the grad explosion.
# Per NVIDIA's NVFP4 paper (arXiv:2509.25149 sec 4 / App E.2), Wgrad carries the
# largest FP4 quant error (esp. late blocks), so `wgrad_fp8` is the prime suspect.
_FP4_BACKWARD_PRECISIONS = {"mxfp4", "mxfp4_pad", "fp8", "dgrad_fp8", "wgrad_fp8"}

# AITER a4w4 f4gemm returns ~1e7x garbage whenever its CONTRACTION dim K is not a
# multiple of 256 (kernel bug; HIPBLASLT is correct but its rocRoller FP4
# solutions require the token OUTPUT dim %256 on fwd/dgrad and hang, which is why
# the stack pins FP4 to AITER). For fwd/dgrad
# K = in/out_features (already %256 for Wan); for the wgrad GEMM K = tokens, which
# for cross-attention key/value projections is text_seq*batch (e.g. 226*8=1808,
# not %256) -> the wgrad explodes. The ``mxfp4_pad`` backward zero-pads the token
# (contraction) dim to this multiple so the wgrad stays in AITER's correct regime;
# zeros are numerically exact (they contribute 0 to grad_out^T @ x).
_AITER_FP4_K_MULTIPLE = 256


def _fp4_backward_precision() -> str:
    """MXFP4 backward GEMM precision (mirrors/extends ``mxfp4_backward_precision``).

    ``mxfp4`` (default) = pure MXFP4 fwd+bwd (the converging Run C recipe).
    ``fp8`` = hybrid: MXFP4 forward, tensorwise-FP8 backward (both dgrad+wgrad).
    ``dgrad_fp8`` / ``wgrad_fp8`` = isolate one backward GEMM in FP8, the other in
    MXFP4 (diagnostic to attribute the gradient explosion to dgrad vs wgrad).
    The FP8 backward keeps bf16 x/W and requantizes at backward time (no
    activation-memory savings for the FP8 leg), matching the local spec.
    """
    p = os.getenv("PRIMUS_TURBO_FP4_BACKWARD", "mxfp4").lower()
    if p not in _FP4_BACKWARD_PRECISIONS:
        raise ValueError(
            f"PRIMUS_TURBO_FP4_BACKWARD={p!r} invalid; expected one of "
            "mxfp4, mxfp4_pad, fp8, dgrad_fp8, wgrad_fp8"
        )
    return p


def _fp8_stage_dtype(fmt, is_fwd_stage: bool):
    """Map a primus_turbo FP8 ``Format`` to the concrete dtype for a GEMM stage.

    HYBRID = E4M3 for the forward-stage operands (act/weight), E5M2 for gradients
    -- the same convention as primus_turbo's ``FP8GemmTensorFunction``.
    """
    from primus_turbo.pytorch.core.low_precision import (
        Format,
        float8_e4m3,
        float8_e5m2,
    )

    if fmt == Format.E4M3:
        return float8_e4m3
    if fmt == Format.E5M2:
        return float8_e5m2
    return float8_e4m3 if is_fwd_stage else float8_e5m2


# Sensitive-layer band ---------------------------------------------------------
_FP4_SENSITIVE_PRECISIONS = {"tw_fp8", "bf16"}
# Matches Wan's ``blocks.N.`` and FLUX's ``transformer_blocks.N.`` module trees.
_FP4_BLOCK_RE = re.compile(r"(?:^|\.)((?:\w+_)*blocks)\.(\d+)(?:\.|$)")


def _fp4_sensitive_enabled() -> bool:
    return os.getenv("PRIMUS_TURBO_FP4_SENSITIVE_LAYERS", "0") in _TRUTHY


def _fp4_sensitive_start() -> int:
    return int(os.getenv("PRIMUS_TURBO_FP4_SENSITIVE_START", "2"))


def _fp4_sensitive_end() -> int:
    return int(os.getenv("PRIMUS_TURBO_FP4_SENSITIVE_END", "8"))


def _fp4_sensitive_precision() -> str:
    p = os.getenv("PRIMUS_TURBO_FP4_SENSITIVE_PRECISION", "tw_fp8").lower()
    if p not in _FP4_SENSITIVE_PRECISIONS:
        raise ValueError(
            f"PRIMUS_TURBO_FP4_SENSITIVE_PRECISION={p!r} invalid; expected one of tw_fp8, bf16"
        )
    return p


def _block_index(fqn: str):
    """Return (block_list_name, block_index) parsed from a module FQN, or (None, None).

    e.g. ``blocks.7.attn1.to_q`` -> ("blocks", 7); ``proj_out`` -> (None, None).
    """
    m = _FP4_BLOCK_RE.search(fqn)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _count_blocks(module: nn.Module):
    """Map each transformer-block ModuleList prefix -> block count (max idx + 1).

    Walks the module's named Linears so the band ('first N + last M blocks') can
    be resolved without assuming a specific attribute name.
    """
    counts: dict[str, int] = {}
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        list_name, idx = _block_index(name)
        if list_name is not None:
            counts[list_name] = max(counts.get(list_name, 0), idx + 1)
    return counts


def _is_sensitive_block(fqn: str, block_counts: dict) -> bool:
    """Whether ``fqn`` lives in the sensitive band (first START + last END blocks)."""
    list_name, idx = _block_index(fqn)
    if list_name is None or list_name not in block_counts:
        return False
    start = _fp4_sensitive_start()
    end = _fp4_sensitive_end()
    count = block_counts[list_name]
    return idx < start or idx >= (count - end)


def _resolve_fp4_hybrid_fp8_config():
    """Tensorwise-FP8 config for the MXFP4 hybrid backward (HYBRID = E4M3 fwd / E5M2 grad)."""
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScalingGranularity,
    )

    return Float8QuantConfig(format=Format.HYBRID, granularity=ScalingGranularity.TENSORWISE)


class _FP4FwdFP8BwdFunction(torch.autograd.Function):
    """Hybrid MXFP4-forward / tensorwise-FP8-backward GEMM (mirrors the
    ``mxfp4_backward_precision='fp8'`` reference path).

    Forward quantizes activation+weight to MXFP4 and runs ``gemm_fp4_impl`` (same
    E2M1_X2 / block-32 / NT recipe as the pure path), but saves **bf16 copies** of
    the activation and weight. Backward requantizes those bf16 tensors and the
    incoming gradient to tensorwise FP8 and runs ``gemm_fp8_impl`` for dgrad/wgrad
    -- byte-for-byte the recipe primus_turbo's ``FP8GemmTensorFunction`` uses
    (a=x trans_a=False, b=W trans_b=True). Trade-off: no activation-memory savings
    (bf16 saved), but the backward uses the more mature tensorwise-FP8 GEMM.
    """

    @staticmethod
    def forward(ctx, a, b, out_dtype, fp4_config, fp8_config, sr_fwd):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import (
            MXScalingRecipe,
            check_mxfp4_support,
        )
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
            enable_preshuffle,
            gemm_fp4_impl,
        )
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        ok, reason = check_mxfp4_support()
        assert ok, reason
        fp4_dtype = torch.float4_e2m1fn_x2

        # Forward: MXFP4 (row recipes only; the transposed tensors are unused
        # because the backward runs in FP8 from saved bf16 copies).
        a_fp4, a_scale_inv, _, _ = quantize_fp4_with_trans(
            a,
            fp4_dtype,
            fp4_config.granularity,
            block_size=fp4_config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
            ),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=False,
                use_sr=sr_fwd,
                use_rht=True,
                shuffle_scale=enable_preshuffle(),
            ),
        )
        b_fp4, b_scale_inv, _, _ = quantize_fp4_with_trans(
            b,
            fp4_dtype,
            fp4_config.granularity,
            block_size=fp4_config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=True,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
                shuffle_out=enable_preshuffle(),
            ),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=True,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
                shuffle_out=enable_preshuffle(),
            ),
        )
        out = gemm_fp4_impl(
            a_fp4,
            a_scale_inv,
            False,
            b_fp4,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=fp4_config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        # Save bf16 copies for the FP8 backward (dtype-correct, per local spec).
        ctx.save_for_backward(a, b)
        ctx.out_dtype = out_dtype
        ctx.fp8_config = fp8_config
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import (
            Format,
            float8_e4m3,
            float8_e5m2,
        )
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
        from primus_turbo.pytorch.ops.quantization import quantize_fp8

        a, b = ctx.saved_tensors  # bf16 activation (x) and weight (W)
        cfg = ctx.fp8_config
        gran = cfg.granularity

        def _fp8_dtype(is_fwd_stage: bool):
            if cfg.format == Format.E4M3:
                return float8_e4m3
            if cfg.format == Format.E5M2:
                return float8_e5m2
            return float8_e4m3 if is_fwd_stage else float8_e5m2  # HYBRID

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        fwd_dt = _fp8_dtype(True)
        grad_dt = _fp8_dtype(False)
        a_fp8, a_scale_inv = quantize_fp8(a, fwd_dt, gran)
        b_fp8, b_scale_inv = quantize_fp8(b, fwd_dt, gran)
        grad_fp8, grad_scale_inv = quantize_fp8(grad_out, grad_dt, gran)

        # Mirrors FP8GemmTensorFunction.backward with (a=x, trans_a=False,
        # b=W, trans_b=True): out = x @ W^T.
        # grad_a = grad @ W
        grad_a = gemm_fp8_impl(
            grad_fp8,
            grad_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            False,
            ctx.out_dtype,
            False,
            granularity=gran.value,
            default_backend=BackendType.HIPBLASLT.value,
        )
        # grad_b = x^T @ grad  (trans_c=True yields the [out, in] weight grad)
        grad_b = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            True,
            grad_fp8,
            grad_scale_inv,
            False,
            ctx.out_dtype,
            True,
            granularity=gran.value,
            default_backend=BackendType.HIPBLASLT.value,
        )
        return grad_a, grad_b, None, None, None, None


def _gemm_fp4_hybrid(a, b, out_dtype, fp4_config, fp8_config, sr_fwd):
    """MXFP4-forward / tensorwise-FP8-backward GEMM (our local Float4Linear seam)."""
    return _FP4FwdFP8BwdFunction.apply(a, b, out_dtype, fp4_config, fp8_config, sr_fwd)


class _FP4GemmMixedFunction(torch.autograd.Function):
    """MXFP4 forward with a PER-GEMM-choosable backward precision, to isolate
    whether the gradient explosion comes from Dgrad (grad wrt input) or Wgrad
    (grad wrt weight).

    ``dgrad_fp8`` / ``wgrad_fp8`` independently pick tensorwise-FP8 (True) or
    MXFP4 (False) for each backward GEMM. The MXFP4 leg reuses the exact stock
    recipe (E2M1_X2 / block-32 / NT, RHT on the transposed grad, SR per
    ``sr_grad``); the FP8 leg reuses primus_turbo's tensorwise recipe. Forward
    saves BOTH the transposed FP4 tensors (for the MXFP4 leg) AND bf16 x/W (for
    the FP8 leg) -- heavier, but this is a diagnostic path, not the perf path.
    """

    @staticmethod
    def forward(ctx, a, b, out_dtype, fp4_config, fp8_config, sr_fwd, sr_grad, dgrad_fp8, wgrad_fp8):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import (
            MXScalingRecipe,
            check_mxfp4_support,
        )
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
            enable_preshuffle,
            gemm_fp4_impl,
        )
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        ok, reason = check_mxfp4_support()
        assert ok, reason
        fp4_dtype = torch.float4_e2m1fn_x2

        a_fp4, a_scale_inv, a_t_fp4, a_t_scale_inv = quantize_fp4_with_trans(
            a, fp4_dtype, fp4_config.granularity, block_size=fp4_config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False, use_sr=sr_fwd, use_rht=False, shuffle_scale=enable_preshuffle()),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=False, use_sr=sr_fwd, use_rht=True, shuffle_scale=enable_preshuffle()),
        )
        b_fp4, b_scale_inv, b_t_fp4, b_t_scale_inv = quantize_fp4_with_trans(
            b, fp4_dtype, fp4_config.granularity, block_size=fp4_config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=True, use_sr=sr_fwd, use_rht=False,
                shuffle_scale=enable_preshuffle(), shuffle_out=enable_preshuffle()),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=True, use_sr=sr_fwd, use_rht=False,
                shuffle_scale=enable_preshuffle(), shuffle_out=enable_preshuffle()),
        )
        out = gemm_fp4_impl(
            a_fp4, a_scale_inv, False, b_fp4, b_scale_inv, True, out_dtype, False,
            granularity=fp4_config.granularity.value, default_backend=BackendType.HIPBLASLT.value,
        )
        # Save both representations: FP4 transposed (MXFP4 legs) + bf16 (FP8 legs).
        ctx.save_for_backward(a_t_fp4, a_t_scale_inv, b_t_fp4, b_t_scale_inv, a, b)
        ctx.out_dtype = out_dtype
        ctx.fp4_config = fp4_config
        ctx.fp8_config = fp8_config
        ctx.sr_grad = sr_grad
        ctx.dgrad_fp8 = dgrad_fp8
        ctx.wgrad_fp8 = wgrad_fp8
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import MXScalingRecipe
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
            enable_preshuffle,
            gemm_fp4_impl,
        )
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
        from primus_turbo.pytorch.ops.quantization import (
            quantize_fp4_with_trans,
            quantize_fp8,
        )

        a_t_fp4, a_t_scale_inv, b_t_fp4, b_t_scale_inv, a, b = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_out = grad_out.view(grad_out.shape[0], -1)
        fp4_dtype = torch.float4_e2m1fn_x2
        fp4_gran = ctx.fp4_config.granularity
        fp8_gran = ctx.fp8_config.granularity
        HB = BackendType.HIPBLASLT.value

        # --- MXFP4 grad_out quant (row + transposed), shared by whichever leg(s)
        #     stay in FP4. RHT on the transposed recipe (feeds wgrad). ---
        (g_fp4, g_scale, g_t_fp4, g_t_scale) = quantize_fp4_with_trans(
            grad_out, fp4_dtype, fp4_gran, block_size=ctx.fp4_config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False, use_sr=ctx.sr_grad, use_rht=False, shuffle_scale=enable_preshuffle()),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=False, use_sr=ctx.sr_grad, use_rht=True, shuffle_scale=enable_preshuffle()),
        )
        # --- tensorwise FP8 quant of grad_out / a / b, shared by the FP8 leg(s). ---
        g_dt = _fp8_stage_dtype(ctx.fp8_config.format, False)
        fwd_dt = _fp8_stage_dtype(ctx.fp8_config.format, True)
        g_fp8, g_fp8_scale = quantize_fp8(grad_out, g_dt, fp8_gran)

        # dgrad = grad @ W  ([M,in])
        if ctx.dgrad_fp8:
            b_fp8, b_fp8_scale = quantize_fp8(b, fwd_dt, fp8_gran)
            grad_a = gemm_fp8_impl(
                g_fp8, g_fp8_scale, False, b_fp8, b_fp8_scale, False, ctx.out_dtype, False,
                granularity=fp8_gran.value, default_backend=HB)
        else:
            grad_a = gemm_fp4_impl(
                g_fp4, g_scale, False, b_t_fp4, b_t_scale_inv, True, ctx.out_dtype, False,
                granularity=fp4_gran.value, default_backend=HB)

        # wgrad = x^T @ grad  ([out,in])
        if ctx.wgrad_fp8:
            a_fp8, a_fp8_scale = quantize_fp8(a, fwd_dt, fp8_gran)
            grad_b = gemm_fp8_impl(
                a_fp8, a_fp8_scale, True, g_fp8, g_fp8_scale, False, ctx.out_dtype, True,
                granularity=fp8_gran.value, default_backend=HB)
        else:
            grad_b = gemm_fp4_impl(
                g_t_fp4, g_t_scale, False, a_t_fp4, a_t_scale_inv, True, ctx.out_dtype, False,
                granularity=fp4_gran.value, default_backend=HB)

        return grad_a, grad_b, None, None, None, None, None, None, None


def _gemm_fp4_mixed(a, b, out_dtype, fp4_config, fp8_config, sr_fwd, sr_grad, dgrad_fp8, wgrad_fp8):
    """MXFP4 fwd with per-GEMM backward precision (dgrad/wgrad isolation seam)."""
    return _FP4GemmMixedFunction.apply(
        a, b, out_dtype, fp4_config, fp8_config, sr_fwd, sr_grad, dgrad_fp8, wgrad_fp8
    )


class _FP4GemmMXSRFunction(torch.autograd.Function):
    """MXFP4 GEMM (E2M1_X2, block-32, NT layout) mirroring primus_turbo 0.2.0's
    ``FP4GemmMXFunction`` but with stochastic rounding selectable per tensor.

    ONLY the ``use_sr`` flags differ from the stock recipe; every other flag
    (``use_2d_block``, ``use_rht``, ``shuffle_scale``/``shuffle_out``, the NT
    layout, and the HIPBLASLT default backend) is copied verbatim so that
    ``sr_fwd=sr_grad=False`` is numerically the stock path. ``sr_fwd`` applies SR
    to the forward activation+weight quant; ``sr_grad`` to the backward gradient
    quant (both the row and transposed recipes used by dgrad and wgrad).
    """

    @staticmethod
    def forward(ctx, a, b, out_dtype, config, sr_fwd, sr_grad):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import (
            MXScalingRecipe,
            check_mxfp4_support,
        )
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
            enable_preshuffle,
            gemm_fp4_impl,
        )
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        ok, reason = check_mxfp4_support()
        assert ok, reason
        fp4_dtype = torch.float4_e2m1fn_x2

        a_fp4, a_scale_inv, a_t_fp4, a_t_scale_inv = quantize_fp4_with_trans(
            a,
            fp4_dtype,
            config.granularity,
            block_size=config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
            ),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=False,
                use_sr=sr_fwd,
                use_rht=True,
                shuffle_scale=enable_preshuffle(),
                shuffle_out=enable_preshuffle(),
            ),
        )
        b_fp4, b_scale_inv, b_t_fp4, b_t_scale_inv = quantize_fp4_with_trans(
            b,
            fp4_dtype,
            config.granularity,
            block_size=config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=True,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
                shuffle_out=enable_preshuffle(),
            ),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=True,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
                shuffle_out=enable_preshuffle(),
            ),
        )
        out = gemm_fp4_impl(
            a_fp4,
            a_scale_inv,
            False,
            b_fp4,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )

        ctx.save_for_backward(a_t_fp4, a_t_scale_inv, b_t_fp4, b_t_scale_inv)
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.sr_grad = sr_grad
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import MXScalingRecipe
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
            enable_preshuffle,
            gemm_fp4_impl,
        )
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        a_t_fp4, a_t_scale_inv, b_t_fp4, b_t_scale_inv = ctx.saved_tensors
        fp4_dtype = torch.float4_e2m1fn_x2
        grad_out = grad_out.view(grad_out.shape[0], -1)

        (
            grad_out_fp4,
            grad_out_scale_inv,
            grad_out_t_fp4,
            grad_out_t_scale_inv,
        ) = quantize_fp4_with_trans(
            grad_out,
            fp4_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=ctx.sr_grad,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
            ),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=False,
                use_sr=ctx.sr_grad,
                use_rht=True,
                shuffle_scale=enable_preshuffle(),
            ),
        )
        grad_a = gemm_fp4_impl(
            grad_out_fp4,
            grad_out_scale_inv,
            False,
            b_t_fp4,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )
        grad_b = gemm_fp4_impl(
            grad_out_t_fp4,
            grad_out_t_scale_inv,
            False,
            a_t_fp4,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )
        return grad_a, grad_b, None, None, None, None


def _gemm_fp4_sr(a, b, out_dtype, config, sr_fwd, sr_grad):
    """SR-enabled MXFP4 GEMM (our local Float4Linear seam)."""
    return _FP4GemmMXSRFunction.apply(a, b, out_dtype, config, sr_fwd, sr_grad)


class _FP4GemmMXPadFunction(torch.autograd.Function):
    """Pure MXFP4 GEMM (E2M1_X2, block-32, NT) identical to ``_FP4GemmMXSRFunction``
    but zero-pads the token dimension to a multiple of ``_AITER_FP4_K_MULTIPLE``
    (256) so the AITER a4w4 f4gemm Wgrad stays numerically correct.

    THE FIX for the Wan2.2 MXFP4 gradient explosion. The three GEMMs of a Linear
    are, in AITER's ``(M, N, K)`` logical shape (K = contraction):
      * forward  out = x @ W^T            -> K = in_features   (%256 for Wan: ok)
      * dgrad    grad_x = grad_out @ W    -> K = out_features  (%256 for Wan: ok)
      * wgrad    grad_W = grad_out^T @ x  -> K = tokens        (NOT %256: explodes)
    Only the wgrad contracts over tokens, and tokens = seq*batch is not %256 for
    the cross-attn key/value projections (text_seq 226 * batch). AITER returns
    ~1e7x garbage there (content-independent; verified with random operands and by
    zero-padding the real operand: %256 -> A~0.55, %32/%128-but-not-%256 -> A~6e7).
    Padding the token dim to %256 (and slicing fwd/dgrad outputs back) keeps every
    GEMM on the tuned, preshuffled AITER path and is exact. ``sr_fwd``/``sr_grad``
    behave exactly as in ``_FP4GemmMXSRFunction`` (``off`` == stock recipe).
    """

    @staticmethod
    def forward(ctx, a, b, out_dtype, config, sr_fwd, sr_grad):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import (
            MXScalingRecipe,
            check_mxfp4_support,
        )
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
            enable_preshuffle,
            gemm_fp4_impl,
        )
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        ok, reason = check_mxfp4_support()
        assert ok, reason
        fp4_dtype = torch.float4_e2m1fn_x2

        # Zero-pad the token (row / contraction) dim up to a multiple of 256 so the
        # token-contracting Wgrad GEMM stays in AITER's correct regime.
        m = a.shape[0]
        k_mult = _AITER_FP4_K_MULTIPLE
        m_pad = ((m + k_mult - 1) // k_mult) * k_mult
        a_p = torch.nn.functional.pad(a, (0, 0, 0, m_pad - m)) if m_pad != m else a

        a_fp4, a_scale_inv, a_t_fp4, a_t_scale_inv = quantize_fp4_with_trans(
            a_p,
            fp4_dtype,
            config.granularity,
            block_size=config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
            ),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=False,
                use_sr=sr_fwd,
                use_rht=True,
                shuffle_scale=enable_preshuffle(),
                shuffle_out=enable_preshuffle(),
            ),
        )
        b_fp4, b_scale_inv, b_t_fp4, b_t_scale_inv = quantize_fp4_with_trans(
            b,
            fp4_dtype,
            config.granularity,
            block_size=config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=True,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
                shuffle_out=enable_preshuffle(),
            ),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=True,
                use_sr=sr_fwd,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
                shuffle_out=enable_preshuffle(),
            ),
        )
        out = gemm_fp4_impl(
            a_fp4,
            a_scale_inv,
            False,
            b_fp4,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )
        # Drop the padded rows so downstream sees the real token count.
        if m_pad != m:
            out = out[:m].contiguous()

        ctx.save_for_backward(a_t_fp4, a_t_scale_inv, b_t_fp4, b_t_scale_inv)
        ctx.out_dtype = out_dtype
        ctx.config = config
        ctx.sr_grad = sr_grad
        ctx.m = m
        ctx.m_pad = m_pad
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import MXScalingRecipe
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import (
            enable_preshuffle,
            gemm_fp4_impl,
        )
        from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

        a_t_fp4, a_t_scale_inv, b_t_fp4, b_t_scale_inv = ctx.saved_tensors
        fp4_dtype = torch.float4_e2m1fn_x2
        grad_out = grad_out.view(grad_out.shape[0], -1)

        # Pad grad_out's token dim to the SAME multiple used for a_t in forward so
        # the wgrad contraction (over tokens) is %256; slice the dgrad back.
        m, m_pad = ctx.m, ctx.m_pad
        grad_p = (
            torch.nn.functional.pad(grad_out, (0, 0, 0, m_pad - m))
            if m_pad != m
            else grad_out
        )

        (
            grad_out_fp4,
            grad_out_scale_inv,
            grad_out_t_fp4,
            grad_out_t_scale_inv,
        ) = quantize_fp4_with_trans(
            grad_p,
            fp4_dtype,
            ctx.config.granularity,
            block_size=ctx.config.block_size,
            scaling_recipe=MXScalingRecipe(
                use_2d_block=False,
                use_sr=ctx.sr_grad,
                use_rht=False,
                shuffle_scale=enable_preshuffle(),
            ),
            scaling_recipe_for_trans=MXScalingRecipe(
                use_2d_block=False,
                use_sr=ctx.sr_grad,
                use_rht=True,
                shuffle_scale=enable_preshuffle(),
            ),
        )
        grad_a = gemm_fp4_impl(
            grad_out_fp4,
            grad_out_scale_inv,
            False,
            b_t_fp4,
            b_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )
        if m_pad != m:
            grad_a = grad_a[:m].contiguous()
        grad_b = gemm_fp4_impl(
            grad_out_t_fp4,
            grad_out_t_scale_inv,
            False,
            a_t_fp4,
            a_t_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.config.granularity.value,
            default_backend=BackendType.HIPBLASLT.value,
        )
        return grad_a, grad_b, None, None, None, None


def _gemm_fp4_pad(a, b, out_dtype, config, sr_fwd, sr_grad):
    """Token-padded MXFP4 GEMM (AITER Wgrad-correctness fix; local seam)."""
    return _FP4GemmMXPadFunction.apply(a, b, out_dtype, config, sr_fwd, sr_grad)


def _resolve_fp4_config():
    """Build the MXFP4 quant config.

    primus_turbo 0.2.0 only supports MX_BLOCKWISE + E2M1_X2 + block_size 32 +
    E8M0 scale for FP4 (enforced by ``Float4QuantConfig.__post_init__``), so
    there are no env knobs here; the defaults already select that recipe.
    """
    from primus_turbo.pytorch.core.low_precision import Float4QuantConfig

    return Float4QuantConfig()


class Float4Linear(torch.nn.Linear):
    """MXFP4 Linear mirroring ``primus_turbo.Float8Linear`` but using ``gemm_fp4``.

    Forward/backward quantize to MXFP4 (E2M1, block 32) inside the autograd-aware
    ``gemm_fp4``; the bias is applied OUTSIDE the FP4 GEMM (in the compute dtype),
    matching the FP8 module and avoiding any bias-in-GEMM kernel gap.
    """

    def __init__(self, in_features, out_features, bias=True, config=None, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.config = config if config is not None else _resolve_fp4_config()
        self.sr_scope = _fp4_sr_scope()
        self.backward_precision = _fp4_backward_precision()
        # Materialize the FP8 backward config for any mode that uses an FP8 leg.
        self.fp8_bwd_config = (
            _resolve_fp4_hybrid_fp8_config()
            if self.backward_precision in ("fp8", "dgrad_fp8", "wgrad_fp8")
            else None
        )
        self._warned_token_align = False

    def forward(self, x):
        flatten_shapes = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)

        # Token dim M is a runtime property; the HIPBLASLT FP4 GEMM needs M % 128
        # for the (transposed) backward GEMMs. Warn once instead of silently
        # falling back to an untuned AITER path. Skip entirely under torch.compile
        # (the flag mutation + logging would otherwise force a graph break).
        # ``mxfp4_pad`` zero-pads the token/contraction dim to %256 internally, so
        # the misalignment the warning is about is already handled -- skip it.
        _compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)()
        if (
            not _compiling
            and self.backward_precision != "mxfp4_pad"
            and not self._warned_token_align
            and x_2d.shape[0] % _FP4_ALIGN != 0
        ):
            logger.warning(
                "[PrimusTurbo-MXFP4] token dim M=%d is not a multiple of %d; the "
                "HIPBLASLT FP4 backward GEMM may fall back or error for this layer "
                "(in=%d out=%d).",
                x_2d.shape[0],
                _FP4_ALIGN,
                self.in_features,
                self.out_features,
            )
            self._warned_token_align = True

        if self.backward_precision == "mxfp4_pad":
            # THE FIX: pure MXFP4 fwd+bwd, but zero-pad the token/contraction dim
            # to %256 so the AITER a4w4 Wgrad (contracts over tokens) is correct.
            out_2d = _gemm_fp4_pad(
                x_2d,
                self.weight,
                x.dtype,
                self.config,
                self.sr_scope == "all",  # sr_fwd
                self.sr_scope != "off",  # sr_grad
            )
        elif self.backward_precision == "fp8":
            # Hybrid: MXFP4 forward, tensorwise-FP8 backward (mirrors the FLUX
            # mxfp4_backward_precision='fp8'). SR (fwd) still honored.
            out_2d = _gemm_fp4_hybrid(
                x_2d,
                self.weight,
                x.dtype,
                self.config,
                self.fp8_bwd_config,
                self.sr_scope == "all",  # sr_fwd
            )
        elif self.backward_precision in ("dgrad_fp8", "wgrad_fp8"):
            # Isolation: one backward GEMM in FP8, the other in MXFP4.
            out_2d = _gemm_fp4_mixed(
                x_2d,
                self.weight,
                x.dtype,
                self.config,
                self.fp8_bwd_config,
                self.sr_scope == "all",  # sr_fwd
                self.sr_scope != "off",  # sr_grad (grad|all -> SR on the MXFP4 leg)
                self.backward_precision == "dgrad_fp8",  # dgrad_fp8
                self.backward_precision == "wgrad_fp8",  # wgrad_fp8
            )
        elif self.sr_scope == "off":
            # Stock path (SR nowhere) -- bit-identical to primus_turbo baseline.
            from primus_turbo.pytorch.ops import gemm_fp4

            out_2d = gemm_fp4(
                x_2d, self.weight, trans_a=False, trans_b=True, out_dtype=x.dtype, config=self.config
            )
        else:
            # SR-enabled local path: grad -> SR on backward grad quant only;
            # all -> SR on forward (act+weight) too.
            out_2d = _gemm_fp4_sr(
                x_2d,
                self.weight,
                x.dtype,
                self.config,
                self.sr_scope == "all",  # sr_fwd
                True,  # sr_grad (both grad and all enable backward SR)
            )
        if self.bias is not None:
            out_2d = out_2d + self.bias
        return out_2d.view(*flatten_shapes, self.out_features)

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, mxfp4=True, sr={self.sr_scope}, "
            f"backward={self.backward_precision}"
        )


def _is_fp4_training_safe_linear(name: str, linear: nn.Linear) -> bool:
    """Whether a Linear is safe to run in MXFP4.

    Builds on AutoModel's own FP8-safety predicate (time/text embed, norm-out,
    norm-modulation, 16-alignment) and adds the stricter 128-alignment the
    HIPBLASLT FP4 GEMM requires on in_features/out_features.
    """
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        _is_fp8_training_safe_linear,
    )

    if not _is_fp8_training_safe_linear(name, linear):
        return False
    # Wan-aware: keep timestep/text/AdaLN conditioning projections in bf16.
    if _fp4_keep_sensitive_bf16() and name.startswith(_FP4_SENSITIVE_PREFIXES):
        return False
    if linear.weight.shape[0] % _FP4_ALIGN != 0 or linear.weight.shape[1] % _FP4_ALIGN != 0:
        return False
    return True


def _replace_linear_with_primus_turbo(
    module: nn.Module,
    module_name: str,
    *,
    fp8_safe_only: bool = False,  # accepted for signature-compat; we always skip-list
) -> int:
    """Drop-in replacement for AutoModel's TE swap using primus_turbo.Float8Linear."""
    from primus_turbo.pytorch.modules import Float8Linear

    # Reuse AutoModel's own fp8-safety predicate so the kept-in-bf16 set matches
    # the TE path exactly (time/text embed, norm modulation, final proj, non-16).
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        _is_fp8_training_safe_linear,
    )

    cfg = _resolve_fp8_config()
    converted = 0
    skipped = 0

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted, skipped
        for child_name, child in list(parent.named_children()):
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, Float8Linear):
                continue
            if isinstance(child, nn.Linear):
                # Always apply the fp8-safe skip-list (stability). The recipe ties
                # its fp8_safe_only flag to te_fp8, which we keep off, so ignore it.
                if not _is_fp8_training_safe_linear(child_fqn, child):
                    skipped += 1
                    logger.info(
                        "[PrimusTurbo-FP8] Keeping %s.%s as torch.nn.Linear (fp8-unsafe); weight=%s",
                        module_name,
                        child_fqn,
                        tuple(child.weight.shape),
                    )
                    continue
                fl = Float8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    config=cfg,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                fl.train(child.training)
                with torch.no_grad():
                    fl.weight.copy_(child.weight)
                    fl.weight.requires_grad_(child.weight.requires_grad)
                    if child.bias is not None:
                        fl.bias.copy_(child.bias)
                        fl.bias.requires_grad_(child.bias.requires_grad)
                setattr(parent, child_name, fl)
                converted += 1
            else:
                replace_children(child, child_fqn)

    replace_children(module)
    logger.info(
        "[PrimusTurbo-FP8] Replaced %d torch.nn.Linear with primus_turbo.Float8Linear in %s; "
        "skipped=%d (granularity=%s format=%s)",
        converted,
        module_name,
        skipped,
        getattr(cfg.granularity, "name", cfg.granularity),
        getattr(cfg.format, "name", cfg.format),
    )
    return converted


def _copy_linear_params(dst: nn.Linear, src: nn.Linear) -> None:
    """Copy weight/bias + requires_grad + training flag from ``src`` to ``dst``."""
    dst.train(src.training)
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        dst.weight.requires_grad_(src.weight.requires_grad)
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)
            dst.bias.requires_grad_(src.bias.requires_grad)


def _replace_linear_with_primus_turbo_fp4(
    module: nn.Module,
    module_name: str,
    *,
    fp8_safe_only: bool = False,  # accepted for signature-compat; we always skip-list
) -> int:
    """Drop-in replacement for AutoModel's TE swap using the local Float4Linear.

    With ``PRIMUS_TURBO_FP4_SENSITIVE_LAYERS`` on, a band of transformer blocks
    (first START + last END) is kept in a higher precision (tensorwise FP8 via
    primus_turbo.Float8Linear, or bf16) instead of MXFP4 -- mirroring the
    ``sensitive_layers_*`` scaffolding that made the MXFP4 ablation
    converge. Everything else (fp8-safe skip-list + 128-alignment) is unchanged.
    """
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        _is_fp8_training_safe_linear,
    )

    cfg = _resolve_fp4_config()
    sensitive_on = _fp4_sensitive_enabled()
    block_counts = _count_blocks(module) if sensitive_on else {}
    sens_precision = _fp4_sensitive_precision() if sensitive_on else None
    fp8_cfg = None
    Float8Linear = None
    if sensitive_on and sens_precision == "tw_fp8":
        from primus_turbo.pytorch.modules import Float8Linear  # noqa: F811

        fp8_cfg = _resolve_fp4_hybrid_fp8_config()  # tensorwise FP8

    converted = 0  # -> Float4Linear (MXFP4)
    converted_fp8 = 0  # -> Float8Linear (sensitive band, tw_fp8)
    skipped = 0  # -> kept torch.nn.Linear (bf16)

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted, converted_fp8, skipped
        for child_name, child in list(parent.named_children()):
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (Float4Linear,) + ((Float8Linear,) if Float8Linear else ())):
                continue
            if isinstance(child, nn.Linear):
                # Sensitive band: keep first START + last END blocks out of MXFP4.
                if sensitive_on and _is_sensitive_block(child_fqn, block_counts):
                    if sens_precision == "tw_fp8" and _is_fp8_training_safe_linear(child_fqn, child):
                        fl = Float8Linear(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            config=fp8_cfg,
                            device=child.weight.device,
                            dtype=child.weight.dtype,
                        )
                        _copy_linear_params(fl, child)
                        setattr(parent, child_name, fl)
                        converted_fp8 += 1
                        logger.info(
                            "[PrimusTurbo-MXFP4] Sensitive band -> tw_fp8 (Float8Linear): %s.%s weight=%s",
                            module_name,
                            child_fqn,
                            tuple(child.weight.shape),
                        )
                    else:
                        skipped += 1
                        logger.info(
                            "[PrimusTurbo-MXFP4] Sensitive band -> bf16 (kept nn.Linear): %s.%s weight=%s",
                            module_name,
                            child_fqn,
                            tuple(child.weight.shape),
                        )
                    continue
                # FP8-safe skip-list + stricter 128-alignment for the FP4 GEMM.
                if not _is_fp4_training_safe_linear(child_fqn, child):
                    skipped += 1
                    logger.info(
                        "[PrimusTurbo-MXFP4] Keeping %s.%s as torch.nn.Linear "
                        "(fp4-unsafe / not 128-aligned); weight=%s",
                        module_name,
                        child_fqn,
                        tuple(child.weight.shape),
                    )
                    continue
                fl = Float4Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    config=cfg,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                _copy_linear_params(fl, child)
                setattr(parent, child_name, fl)
                converted += 1
            else:
                replace_children(child, child_fqn)

    replace_children(module)
    if sensitive_on:
        band = ", ".join(f"{k}[first {_fp4_sensitive_start()}+last {_fp4_sensitive_end()}/{v}]"
                         for k, v in block_counts.items()) or "none-found"
        logger.info(
            "[PrimusTurbo-MXFP4] Replaced %d Float4Linear + %d %s (sensitive band: %s) in %s; "
            "skipped(bf16)=%d (MX_BLOCKWISE E2M1_X2 block=%d sr=%s backward=%s)",
            converted,
            converted_fp8,
            "Float8Linear(tw_fp8)" if sens_precision == "tw_fp8" else "bf16",
            band,
            module_name,
            skipped,
            getattr(cfg, "block_size", 32),
            _fp4_sr_scope(),
            _fp4_backward_precision(),
        )
    else:
        logger.info(
            "[PrimusTurbo-MXFP4] Replaced %d torch.nn.Linear with Float4Linear in %s; "
            "skipped=%d (MX_BLOCKWISE E2M1_X2 block=%d sr=%s backward=%s)",
            converted,
            module_name,
            skipped,
            getattr(cfg, "block_size", 32),
            _fp4_sr_scope(),
            _fp4_backward_precision(),
        )
    return converted + converted_fp8


# --------------------------------------------------------------------------- #
# TE-native MXFP4 (te.pytorch.Linear under an MXFP4BlockScaling autocast)      #
# --------------------------------------------------------------------------- #
# This path is the reason for the TE-native image. It does two things the
# primus_turbo path cannot on the baseline image:
#   1. swaps nn.Linear -> transformer_engine.pytorch.Linear (128-aligned, same
#      skip-list), and
#   2. wraps each swapped transformer's forward in an MXFP4 autocast so the TE
#      Linears actually quantize (a bare te.Linear runs bf16 outside autocast).
# TE's MXFP4 recipe applies RHT (Hadamard) + the native fused HIP cast/transpose
# kernel and dispatches the GEMM to AITER a4w4 -- a converging configuration.


def _build_te_mxfp4_recipe():
    """Construct the TE MXFP4 recipe.

    Uses ``transformer_engine.common.recipe.MXFP4BlockScaling`` (wired to
    ``Recipe.mxfp4()``). RHT (the convergence lever) is a per-instance
    attribute driven by ``NVTE_MXFP4_USE_HADAMARD`` -- matching the reference
    ``get_fp4_recipe`` (fp4_utils.py), which sets ``recipe.use_hadamard`` the same
    way. Verified on the TE-native image: the recipe builds with use_hadamard=True
    and the fwd/bwd dispatch to AITER a4w4 f4gemm.
    """
    from transformer_engine.common.recipe import MXFP4BlockScaling

    recipe = MXFP4BlockScaling()
    recipe.use_hadamard = os.getenv("NVTE_MXFP4_USE_HADAMARD", "0") in _TRUTHY
    return recipe


def _te_mxfp4_autocast(recipe):
    """Return a TE autocast context for the given MXFP4 recipe.

    Handles TE API drift: newer TE exposes ``te.pytorch.autocast(recipe=...)``;
    the classic API is ``te.pytorch.fp8_autocast(fp8_recipe=...)``. We probe both.
    """
    import transformer_engine.pytorch as tep

    if hasattr(tep, "autocast"):
        try:
            return tep.autocast(enabled=True, recipe=recipe)
        except TypeError:
            pass
    try:
        return tep.fp8_autocast(enabled=True, recipe=recipe)
    except TypeError:
        return tep.fp8_autocast(enabled=True, fp8_recipe=recipe)


def _wrap_forward_with_te_mxfp4_autocast(module: nn.Module) -> None:
    """Wrap ``module.forward`` so every call runs under the MXFP4 autocast.

    Bound on the module INSTANCE before FSDP2 wrapping. FSDP2 (fully_shard) keeps
    the module's ``forward`` and drives it through hooks, so the autocast is
    entered on each real forward. Idempotent (guards against double-wrap).
    """
    if getattr(module, "_te_mxfp4_wrapped", False):
        return
    recipe = _build_te_mxfp4_recipe()
    orig_forward = module.forward

    def forward(*args, **kwargs):
        with _te_mxfp4_autocast(recipe):
            return orig_forward(*args, **kwargs)

    module.forward = forward  # type: ignore[method-assign]
    module._te_mxfp4_wrapped = True  # type: ignore[attr-defined]


# Transformer-block ModuleList attribute names to wrap for autocast (Wan uses
# ``blocks``; FLUX/diffusers use ``transformer_blocks`` AND, for the single-stream
# stack, ``single_transformer_blocks`` -- BOTH must be wrapped or the 38 single
# blocks run their TE Linears outside the MXFP4 autocast).
_TE_BLOCK_LIST_ATTRS = ("blocks", "transformer_blocks", "single_transformer_blocks")


def _wrap_te_mxfp4_autocast_for_ac(module: nn.Module) -> str:
    """Enter the MXFP4 autocast at the *transformer-block* level, not the top.

    CRITICAL for activation checkpointing. AutoModel wraps each block in a
    ``NO_REENTRANT`` ``checkpoint_wrapper`` (parallelizer.py), so on the backward
    recompute the block's forward reruns *standalone*, OUTSIDE any autocast entered
    at the top-level transformer forward. If the autocast is only on the top-level
    forward, the recompute runs in bf16 (not MXFP4) -> the checkpointed region
    saves a different number of tensors on recompute than on the original forward
    -> ``CheckpointError`` (and, if forced through, silently wrong grads).

    Wrapping each block's forward puts the autocast INSIDE the checkpoint boundary,
    so it is re-entered identically on recompute. All TE Linears live inside the
    blocks (the skip-list keeps conditioning/proj Linears in bf16), so per-block
    coverage is complete. Falls back to whole-module wrap if no block list found.
    Returns a short tag describing what was wrapped (for logging).
    """
    tags = []
    for attr in _TE_BLOCK_LIST_ATTRS:
        blocks = getattr(module, attr, None)
        if blocks is not None and len(blocks) > 0:
            for blk in blocks:
                _wrap_forward_with_te_mxfp4_autocast(blk)
            tags.append(f"{attr}[{len(blocks)}]")
    if tags:
        # Wrap ALL block lists (e.g. FLUX has transformer_blocks AND
        # single_transformer_blocks) -- do NOT return after the first match.
        return "per-block:" + "+".join(tags)
    _wrap_forward_with_te_mxfp4_autocast(module)
    return "whole-module (no block list found)"


def _replace_linear_with_te_mxfp4(
    module: nn.Module,
    module_name: str,
    *,
    fp8_safe_only: bool = False,  # accepted for signature-compat; we always skip-list
) -> int:
    """Drop-in replacement for AutoModel's TE swap using native te.pytorch.Linear.

    Swaps 128-aligned, fp8/fp4-safe nn.Linear to ``transformer_engine.pytorch
    .Linear`` (reusing ``_is_fp4_training_safe_linear``), copies weights/bias and
    requires_grad, then wraps the module forward in the MXFP4 autocast.
    """
    from transformer_engine.pytorch import Linear as TELinear

    converted = 0
    skipped = 0

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted, skipped
        for child_name, child in list(parent.named_children()):
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, TELinear):
                continue
            if isinstance(child, nn.Linear):
                if not _is_fp4_training_safe_linear(child_fqn, child):
                    skipped += 1
                    logger.info(
                        "[TE-MXFP4] Keeping %s.%s as torch.nn.Linear "
                        "(fp4-unsafe / not 128-aligned); weight=%s",
                        module_name,
                        child_fqn,
                        tuple(child.weight.shape),
                    )
                    continue
                tl = TELinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                    device=child.weight.device,
                )
                tl.train(child.training)
                with torch.no_grad():
                    tl.weight.copy_(child.weight)
                    tl.weight.requires_grad_(child.weight.requires_grad)
                    if child.bias is not None:
                        tl.bias.copy_(child.bias)
                        tl.bias.requires_grad_(child.bias.requires_grad)
                setattr(parent, child_name, tl)
                converted += 1
            else:
                replace_children(child, child_fqn)

    replace_children(module)
    wrap_tag = "none"
    if converted:
        wrap_tag = _wrap_te_mxfp4_autocast_for_ac(module)
    logger.info(
        "[TE-MXFP4] Replaced %d torch.nn.Linear with te.pytorch.Linear in %s; "
        "skipped=%d (MXFP4BlockScaling autocast wrapped=%s)",
        converted,
        module_name,
        skipped,
        wrap_tag,
    )
    return converted


def install() -> bool:
    """Install the swap by monkeypatching AutoModel's TE swap symbol.

    Precedence: TE-native MXFP4 (``PRIMUS_TE_MXFP4``) > primus_turbo MXFP4
    (``PRIMUS_TURBO_FP4``) > primus_turbo FP8 (``PRIMUS_TURBO_FP8``); else no-op
    (returns False). Modifies NO Automodel source; only rebinds a module
    attribute at runtime. Call this BEFORE building the recipe/pipeline.
    """
    if is_te_mxfp4_enabled():
        # Ensure the running TE can build the MXFP4 recipe. NOTE: we intentionally
        # do NOT hard-depend on ``check_mxfp4_support`` -- on some TE-native
        # builds that symbol is NOT in transformer_engine
        # .pytorch.fp8 (that module is deprecated), yet MXFP4 executes fine
        # (verified: fwd/bwd dispatch to AITER a4w4 f4gemm, finite grads). So the
        # recipe-build is the real gate; the support check is best-effort.
        _build_te_mxfp4_recipe()  # raises if MXFP4BlockScaling is unavailable (baseline TE)
        _reason = None
        for _mod in ("transformer_engine.pytorch", "primus_turbo.pytorch.core.low_precision"):
            try:
                _m = __import__(_mod, fromlist=["check_mxfp4_support"])
                _chk = getattr(_m, "check_mxfp4_support", None)
                if _chk is not None:
                    _ok, _reason = _chk()
                    if not _ok:
                        raise RuntimeError(f"TE/Turbo reports MXFP4 unsupported: {_reason}")
                    break
            except ImportError:
                continue
        if _reason is None:
            logger.warning(
                "[TE-MXFP4] check_mxfp4_support() not found in TE or primus_turbo; "
                "proceeding (recipe builds). Ensure this is the TE-native image (gfx950)."
            )

        import nemo_automodel._diffusers.auto_diffusion_pipeline as adp

        adp._replace_linear_with_transformer_engine = _replace_linear_with_te_mxfp4
        logger.info(
            "[TE-MXFP4] Installed: nn.Linear -> te.pytorch.Linear + MXFP4BlockScaling "
            "autocast (triggered by model.transformer_engine_linear=true)"
        )
        return True

    if is_fp4_enabled():
        # Fail fast if the FP4 op is unavailable so the run errors clearly.
        from primus_turbo.pytorch.ops import gemm_fp4  # noqa: F401

        import nemo_automodel._diffusers.auto_diffusion_pipeline as adp

        adp._replace_linear_with_transformer_engine = _replace_linear_with_primus_turbo_fp4
        logger.info(
            "[PrimusTurbo-MXFP4] Installed: nn.Linear -> Float4Linear (gemm_fp4) swap active "
            "(triggered by model.transformer_engine_linear=true)"
        )
        return True

    if not is_enabled():
        return False

    # Fail fast if primus_turbo is unavailable so the run errors clearly rather
    # than silently falling back to TE.
    import primus_turbo.pytorch.modules  # noqa: F401

    import nemo_automodel._diffusers.auto_diffusion_pipeline as adp

    adp._replace_linear_with_transformer_engine = _replace_linear_with_primus_turbo
    logger.info(
        "[PrimusTurbo-FP8] Installed: nn.Linear -> primus_turbo.Float8Linear swap active "
        "(triggered by model.transformer_engine_linear=true)"
    )
    return True
