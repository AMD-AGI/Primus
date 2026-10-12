"""MXFP4 GEMM for LoRA-frozen base weights, on FlyDSL quantize and FlyDSL GEMM.

Stock ``primus_turbo.pytorch.ops.gemm_fp4`` is written for trainable weights. Per call
it quantizes the activation row- and column-wise and saves the column half, then in
backward quantizes grad_out both ways and runs dgrad *and* wgrad. Under LoRA the base
weight is frozen, so the activation's column half, grad_out's column half and the wgrad
GEMM are all discarded work: 960 GEMMs a step where 640 are needed.

``_FrozenWeightFP4Gemm`` does only what a frozen weight needs:

  forward   x -> FlyDSL quantize (row half used) -> FlyDSL GEMM against the weight's
            row half. Nothing derived from x is saved.
  backward  grad_out -> FlyDSL quantize (row half used) -> FlyDSL GEMM against the
            weight's column half. No wgrad.

The weight pair is quantized once at load by ``pre_quantize_mxfp4`` (1x32, no shuffle).
FlyDSL's only MXFP4 quantize entry is the fused row+col dual, which is bit-exact against
Turbo's HIP quantize and still faster than the HIP row-only kernel on these shapes, so the
unused column half is cheaper to produce and drop than to avoid.
"""

from __future__ import annotations

import os

import torch

_STATE = {"installed": False, "frozen_calls": 0, "stock_calls": 0}

#: Set by turbo_lora_fp4 around one LoRALinear base call. The next frozen GEMM takes it and
#: may return the base output with the adapter already folded in; None means it declined.
LORA_SLOT = [None]


def _log(msg: str) -> None:
    if os.environ.get("TURBO_DEBUG", "0") in ("1", "true", "True"):
        try:
            with open("/results/turbo_decide.log", "a") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass


def _recipes(use_gradient_sr: bool):
    from primus_turbo.pytorch.core.low_precision import ScalingRecipe

    # Row halves feed the GEMMs and match stock gemm_fp4's row recipes exactly. The column
    # halves are dropped, so they get the cheapest recipe (no RHT, no SR).
    act_row = ScalingRecipe(use_2d_block=False, use_sr=False, use_rht=False)
    grad_row = ScalingRecipe(use_2d_block=False, use_sr=use_gradient_sr, use_rht=False)
    dropped = ScalingRecipe(use_2d_block=False, use_sr=False, use_rht=False)
    return act_row, grad_row, dropped


def _quantize_row(x, recipe, dropped, config):
    from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

    q, s, _, _ = quantize_fp4_with_trans(
        x,
        torch.float4_e2m1fn_x2,
        config.granularity,
        block_size=config.block_size,
        scaling_recipe=recipe,
        scaling_recipe_for_trans=dropped,
        scale_rounding_mode=config.scale_rounding_mode,
    )
    return q, s


def _gemm_nt(a, a_s, b, b_s, out_dtype, config):
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl

    return gemm_fp4_impl(
        a,
        a_s,
        False,
        b,
        b_s,
        True,
        out_dtype,
        False,
        granularity=config.granularity.value,
        default_backend=BackendType.FLYDSL.value,
        preshuffled=False,
    )


class _FrozenWeightFP4Gemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_row, w_row_s, w_col, w_col_s, out_dtype, config):
        act_row, grad_row, dropped = _recipes(config.use_gradient_sr)
        a, a_s = _quantize_row(x, act_row, dropped, config)
        out = _gemm_nt(a, a_s, w_row, w_row_s, out_dtype, config)
        ctx.save_for_backward(w_col, w_col_s)
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.grad_row = grad_row
        ctx.dropped = dropped
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w_col, w_col_s = ctx.saved_tensors
        g = grad_out.reshape(grad_out.shape[0], -1).contiguous()
        g_q, g_s = _quantize_row(g, ctx.grad_row, ctx.dropped, ctx.config)
        grad_x = _gemm_nt(g_q, g_s, w_col, w_col_s, ctx.out_dtype, ctx.config)
        return grad_x, None, None, None, None, None, None


_GLU_BUILD = [False]


def _one_tile_per_wg_for_glu():
    """Build the dense glu-quant GEMM with one tile per workgroup.

    At Turbo 30e6de87 the persistent 4-tiles-per-WG build of that kernel drops the first
    256-K block from part of every later tile (rows 0-15 of each 64, half the N bands):
    on the 70B MLP, l1 reads 23.9 dB against the exact product where the plain GEMM reads
    55.6 dB. With 1 or 2 tiles per WG it is bit-identical to the plain GEMM, and 1 is also
    the fastest (1.650 vs 1.722 ms). The tile count is only read at build time and glu
    builds carry their own launch-cache keys, so no other GEMM is affected.
    """
    import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as km

    if getattr(km, "_frozen_glu_tpw_patched", False):
        return
    orig = km._mxfp4_tiles_per_wg

    def tiles_per_wg(n_pids, K, persist):
        return 1 if _GLU_BUILD[0] else orig(n_pids, K, persist)

    km._mxfp4_tiles_per_wg = tiles_per_wg
    km._frozen_glu_tpw_patched = True


def _probs_ones(m, device, _cache={}):
    key = (m, str(device))
    t = _cache.get(key)
    if t is None:
        t = _cache[key] = torch.ones(m, device=device, dtype=torch.float32)
    return t


class _FrozenFusedMLP(torch.autograd.Function):
    """fc1 -> SwiGLU -> fc2 for frozen 1x32 weights, SwiGLU fused into the GEMMs.

    forward   x -> FlyDSL quantize (row) -> fc1 GEMM with a StoreCSwiGLUQuant epilogue,
              which keeps l1 = x @ w1^T in bf16 for the backward and writes the
              activation already MXFP4-quantized -> fc2 GEMM.
    backward  grad_out -> FlyDSL quantize (row) -> fc2 dgrad with a
              StoreCdSwiGLUQuadQuant epilogue (dSwiGLU from l1, then quantize) ->
              fc1 dgrad. No wgrad, so the column halves the epilogues also write go
              unused.
    """

    @staticmethod
    def forward(ctx, x, w1_row, w1_row_s, w1_col, w1_col_s, w2_row, w2_row_s, w2_col, w2_col_s, config):
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_glu_quant_impl

        act_row, grad_row, dropped = _recipes(config.use_gradient_sr)
        a, a_s = _quantize_row(x, act_row, dropped, config)
        probs = _probs_ones(x.shape[0], x.device)
        _one_tile_per_wg_for_glu()
        _GLU_BUILD[0] = True
        try:
            l1, h, h_s, _, _ = gemm_fp4_glu_quant_impl(
                a, a_s, w1_row, w1_row_s, probs, torch.bfloat16, False, False,
                config.scale_rounding_mode, "silu", None,
            )
        finally:
            _GLU_BUILD[0] = False
        out = _gemm_nt(h, h_s, w2_row, w2_row_s, torch.bfloat16, config)
        ctx.save_for_backward(w1_col, w1_col_s, w2_col, w2_col_s, l1, probs)
        ctx.config = config
        ctx.grad_row = grad_row
        ctx.dropped = dropped
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_dglu_quant_impl

        w1_col, w1_col_s, w2_col, w2_col_s, l1, probs = ctx.saved_tensors
        cfg = ctx.config
        g = grad_out.reshape(grad_out.shape[0], -1).contiguous()
        g_q, g_s = _quantize_row(g, ctx.grad_row, ctx.dropped, cfg)
        sr = cfg.use_gradient_sr
        gl, gl_s, _, _ = gemm_fp4_dglu_quant_impl(
            g_q, g_s, w2_col, w2_col_s, l1, probs, torch.bfloat16, sr, sr,
            cfg.scale_rounding_mode, "silu", None,
        )
        grad_x = _gemm_nt(gl, gl_s, w1_col, w1_col_s, torch.bfloat16, cfg)
        return grad_x, None, None, None, None, None, None, None, None, None


def fused_mlp_eligible(x, w1, w2, config) -> bool:
    """Whether ``frozen_fused_mlp`` covers these operands."""
    from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import (
        dense_dglu_epi_quant_supported,
        dense_glu_epi_quant_supported,
    )

    if config is None or getattr(config, "use_preshuffle", False):
        return False
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        return False
    if _frozen_pair(w1) is None or _frozen_pair(w2) is None:
        return False
    M, K = x.shape
    I = w1.shape[0] // 2
    return dense_glu_epi_quant_supported(K, I, M, torch.bfloat16) and dense_dglu_epi_quant_supported(
        w2.shape[0], I, M, torch.bfloat16
    )


def frozen_fused_mlp(x, w1, w2, config):
    """``fc2(silu(gate) * up)`` with w1 = [gate; up] (2I x K) and w2 (K_out x I), both frozen."""
    w1_row, w1_col = _frozen_pair(w1)
    w2_row, w2_col = _frozen_pair(w2)
    _STATE["fused_mlp_calls"] = _STATE.get("fused_mlp_calls", 0) + 1
    if _STATE["fused_mlp_calls"] == 1:
        _log(f"[FrozenFP4] fused MLP: x={tuple(x.shape)} w1={tuple(w1_row.shape)} w2={tuple(w2_row.shape)}")
    return _FrozenFusedMLP.apply(
        x.contiguous(),
        w1_row.qdata, w1_row.scale_inv, w1_col.qdata, w1_col.scale_inv,
        w2_row.qdata, w2_row.scale_inv, w2_col.qdata, w2_col.scale_inv,
        config,
    )


def _frozen_pair(b):
    """The (row, col) QuantizedTensors of a frozen 1x32 weight, else None."""
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor, QuantizedTensorPair

    if isinstance(b, QuantizedTensorPair):
        row, col = b.data, b.data_t
    elif isinstance(b, QuantizedTensor):
        return None
    else:
        pair = getattr(b, "_turbo_fp4_pair", None)
        if pair is None or b.requires_grad:
            return None
        row, col = pair
    if not isinstance(row, QuantizedTensor) or not isinstance(col, QuantizedTensor):
        return None
    if row.requires_grad or col.requires_grad:
        return None
    for q in (row, col):
        r = q.scaling_recipe
        if r is None or r.use_2d_block or r.shuffle_scale or r.shuffle_out:
            return None
    return row, col


def install() -> bool:
    """Route frozen-weight MXFP4 GEMMs through ``_FrozenWeightFP4Gemm``.

    Wraps the ``gemm_fp4`` attribute Primus's Turbo linears call
    (``primus_turbo.pytorch.ops.gemm_fp4``). Anything that is not a frozen, 1x32,
    unshuffled weight pair goes to the stock op unchanged.
    """
    if _STATE["installed"]:
        return True
    import primus_turbo.pytorch.ops as turbo_ops

    stock = turbo_ops.gemm_fp4

    @torch._dynamo.disable
    def gemm_fp4(
        a,
        b,
        trans_a=False,
        trans_b=False,
        out_dtype=None,
        config=None,
        fuse_bgrad_accum_pattern=None,
    ):
        pair = None
        if (
            config is not None
            and not trans_a
            and trans_b
            and fuse_bgrad_accum_pattern is None
            and isinstance(a, torch.Tensor)
            and a.ndim == 2
            and a.dtype == torch.bfloat16
            and not getattr(config, "use_preshuffle", False)
        ):
            pair = _frozen_pair(b)
        if pair is None:
            _STATE["stock_calls"] += 1
            if _STATE["stock_calls"] == 1:
                _log(f"[FrozenFP4] stock gemm_fp4 for b={type(b).__name__}")
            return stock(a, b, trans_a, trans_b, out_dtype, config, fuse_bgrad_accum_pattern)
        row, col = pair
        _STATE["frozen_calls"] += 1
        if _STATE["frozen_calls"] == 1:
            _log(
                f"[FrozenFP4] frozen path: x={tuple(a.shape)} w={tuple(row.shape)} "
                f"gradient_sr={config.use_gradient_sr} srm={config.scale_rounding_mode}"
            )
        lora = LORA_SLOT[0]
        if lora is not None:
            LORA_SLOT[0] = None
            out = lora(a.contiguous(), row, col, config)
            if out is not None:
                return out
        return _FrozenWeightFP4Gemm.apply(
            a.contiguous(),
            row.qdata,
            row.scale_inv,
            col.qdata,
            col.scale_inv,
            out_dtype or a.dtype,
            config,
        )

    gemm_fp4.__wrapped__ = stock
    turbo_ops.gemm_fp4 = gemm_fp4
    _STATE["installed"] = True
    return True


def quantize_weight_pair(bf16_weight, quant_config):
    """1x32 (row, col) QuantizedTensor pair, built with one FlyDSL dual quantize.

    Bit-exact against ``QuantizedTensor.quantize(axis=-1)`` / ``(axis=-2)``; falls back
    to those when the shape or recipe is not FlyDSL-eligible.
    """
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, ScalingRecipe
    from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor

    assert quant_config.granularity == ScalingGranularity.MX_BLOCKWISE
    fp4 = torch.float4_e2m1fn_x2
    recipe = ScalingRecipe(use_2d_block=False)
    srm = quant_config.scale_rounding_mode
    meta = dict(
        shape=bf16_weight.size(),
        orig_dtype=bf16_weight.dtype,
        dest_dtype=fp4,
        granularity=quant_config.granularity,
        block_size=quant_config.block_size,
        scaling_recipe=recipe,
        scale_rounding_mode=srm,
        requires_grad=False,
    )
    w = bf16_weight.detach().contiguous()
    try:
        from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
            dual_eligible,
            flydsl_dual_quant,
        )

        if w.dtype == torch.bfloat16 and w.ndim == 2 and dual_eligible(*w.shape, recipe, recipe):
            rd, rs, cd, cs = flydsl_dual_quant(w, fp4, False, False, scale_rounding_mode=srm)
            row = QuantizedTensor(rd.contiguous(), rs.contiguous(), quantized_axis=-1, **meta)
            col = QuantizedTensor(cd.contiguous(), cs.contiguous(), quantized_axis=-2, **meta)
            return row, col
    except ImportError:
        pass
    kw = dict(
        dest_dtype=fp4,
        granularity=quant_config.granularity,
        block_size=quant_config.block_size,
        scaling_recipe=recipe,
        scale_rounding_mode=srm,
    )
    return (
        QuantizedTensor.quantize(w, axis=-1, **kw),
        QuantizedTensor.quantize(w, axis=-2, **kw),
    )
