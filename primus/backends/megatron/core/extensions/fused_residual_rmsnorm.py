###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime monkeypatch — fuse residual+add into RMSNorm at two sites.

This is the runtime install hook (callable from a trainer entry point such
as ``small_llm_moe_pretraining/primus/src/train.py``). The Triton RMSNorm
kernels live in Primus-Turbo (``primus_turbo.pytorch.ops.normalization``,
which exposes ``rmsnorm`` and the fused ``rmsnorm_residual`` variant).

V1 — IN-LAYER fuse  (``PRIMUS_FUSED_RESIDUAL_NORM=1``)
    self_attn_bda(residual + attn_out) → pre_mlp_layernorm
    Kills 1 of 2 ``vectorized_elementwise_kernel<CUDAFunctor_add<bf16>>``
    launches per layer (the in-layer ADD#1).

V2 — CROSS-LAYER fuse  (``PRIMUS_FUSED_RESIDUAL_NORM_V2=1``, implies V1)
    layer N: skip mlp_bda(mlp_out + residual_post_attn), instead stash
             ``(mlp_out, residual_post_attn)`` as a "carry" on layer N+1
    layer N+1: input_layernorm(carry) does the fused add+norm
    last layer: carry goes to final_layernorm via ``_v2_pending_carry``.
    Kills the remaining ADD#2 in the bf16 add tax (24 → 1 launches per step;
    the last layer keeps its add when final_layernorm isn't a
    ``PrimusTurboRMSNorm``, otherwise that one is fused too).

Activation
----------
* V1: ``PRIMUS_FUSED_RESIDUAL_NORM=1`` (default 0).
* V2: ``PRIMUS_FUSED_RESIDUAL_NORM_V2=1`` (default 0). Implies V1.
* Requires ``use_turbo_rms_norm=true`` so standalone norms are
  ``PrimusTurboRMSNorm``. On the Llama TE fused LN+Linear spec,
  ``pre_mlp_layernorm`` / ``input_layernorm`` are ``IdentityOp`` and the
  RMSNorm lives on ``mlp.linear_fc1.layer_norm_weight`` /
  ``self_attention.linear_qkv.layer_norm_weight``. V1/V2 then fuse into
  those weights via ``rmsnorm_residual`` and skip the inner ``rmsnorm``.

Falls back silently to the original ``TransformerLayer.forward`` whenever
any precondition fails (recompute paths, fp32 residual, inference fused TP,
non-zero hidden_dropout, cross-attention layers, missing fused-LN gamma).

Skip-``y`` (optional, ``patches/primus-te-ln-skip-y-ste.patch``)
---------------------------------------------------------------
This file is the legal skip-``y`` keep (QKV/MLP ``skip_y_store=True``,
final LN ``False``). ``APPLY_SKIP_Y_STE=0`` reverse-applies that patch
for a stored-``y`` A/B vs TTT 77.90. Do not ``mark_non_differentiable(y)``.

Safety
------
* Both gates default off; A/B is always reproducible.
* On any exception in the fused path the patch reverts permanently and the
  rest of the run uses the original Megatron forward.
* V2 keeps the original mlp_bda when the next consumer cannot fuse (e.g.
  final_layernorm is not a PrimusTurboRMSNorm, or the next layer doesn't
  pass ``_can_fuse``), so partial coverage degrades gracefully.
"""
from __future__ import annotations

import os
import sys
from typing import Any


def _enabled() -> bool:
    """V1 gate (or implied by V2 / R3)."""
    if _v2_enabled() or _r3_enabled():
        return True
    v = os.environ.get("PRIMUS_FUSED_RESIDUAL_NORM", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _v2_enabled() -> bool:
    v = os.environ.get("PRIMUS_FUSED_RESIDUAL_NORM_V2", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _r3_enabled() -> bool:
    v = os.environ.get("PRIMUS_FUSED_RMSNORM_MXFP4", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _run_rmsnorm_residual(x, residual, gamma, eps, skip_y_store=True):
    """V1/V2 residual RMSNorm; R3 also returns MXFP4 dual of y.

    ``skip_y_store`` is for QKV/MLP consumers that pass ``a_prequant`` /
    ``x_prequant``. Final LN feeds the vocab GEMM, which still quantizes
    BF16 ``y`` — that call site passes ``False``.
    """
    if _r3_enabled():
        from primus_turbo.pytorch.ops.normalization import rmsnorm_residual_mxfp4

        y, xpr, row, rs, col, cs = rmsnorm_residual_mxfp4(
            x, residual, gamma, eps, skip_y_store=skip_y_store
        )
        return y, xpr, (row.detach(), rs.detach(), col.detach(), cs.detach())
    from primus_turbo.pytorch.ops.normalization import (
        rmsnorm_residual as triton_rmsnorm_residual,
    )

    y, xpr = triton_rmsnorm_residual(x, residual, gamma, eps)
    return y, xpr, None


def _log(msg: str) -> None:
    rank = os.environ.get("RANK", "0")
    if rank == "0":
        print(f"[fused_residual_rmsnorm] {msg}", file=sys.stderr, flush=True)


_INSTALLED = False
_LOGGED_FC1_PATH = False


def _identity_op_type():
    try:
        from megatron.core.transformer.identity_op import IdentityOp

        return IdentityOp
    except ImportError:
        return type(None)


def _is_identity(mod: Any) -> bool:
    if mod is None:
        return True
    return isinstance(mod, _identity_op_type())


def _fc1_fused_ln_params(layer: Any):
    """TE dense spec: pre-MLP RMSNorm is fused into ``mlp.linear_fc1``."""
    mlp = getattr(layer, "mlp", None)
    fc1 = getattr(mlp, "linear_fc1", None) if mlp is not None else None
    ln_w = getattr(fc1, "layer_norm_weight", None) if fc1 is not None else None
    if ln_w is None:
        return None
    cfg = getattr(layer, "config", None)
    eps = getattr(fc1, "eps", None) or getattr(cfg, "layernorm_epsilon", 1e-5)
    return fc1, ln_w, eps


def _qkv_fused_ln_params(layer: Any):
    """TE dense spec: pre-attn RMSNorm is fused into QKV ``LayerNormLinear``."""
    attn = getattr(layer, "self_attention", None)
    qkv = getattr(attn, "linear_qkv", None) if attn is not None else None
    ln_w = getattr(qkv, "layer_norm_weight", None) if qkv is not None else None
    if ln_w is None:
        return None
    cfg = getattr(layer, "config", None)
    eps = getattr(qkv, "eps", None) or getattr(cfg, "layernorm_epsilon", 1e-5)
    return qkv, ln_w, eps


def install() -> bool:
    """Install the monkeypatch. Returns True on success."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not _enabled():
        _log("disabled (set PRIMUS_FUSED_RESIDUAL_NORM=1 or " "PRIMUS_FUSED_RESIDUAL_NORM_V2=1 to enable)")
        return False

    try:
        from primus_turbo.pytorch.ops.normalization import (
            rmsnorm_residual as triton_rmsnorm_residual,
        )
    except ImportError as exc:
        _log(f"could not import primus_turbo rmsnorm_residual: {exc}; abort install")
        return False

    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboRMSNorm,
        )
    except ImportError as exc:
        _log(f"PrimusTurboRMSNorm not importable: {exc}; abort install")
        return False

    try:
        from megatron.core.transformer.transformer_block import TransformerBlock
        from megatron.core.transformer.transformer_layer import TransformerLayer
    except ImportError as exc:
        _log(f"Megatron transformer modules not importable: {exc}; abort install")
        return False

    # ----- 1. Extend PrimusTurboRMSNorm to accept residual ------------------
    if not getattr(PrimusTurboRMSNorm, "_fused_residual_patched", False):
        _orig_norm_forward = PrimusTurboRMSNorm.forward

        def _new_norm_forward(self, x, residual=None):
            # V2 path: an upstream layer stashed a carry for *this* norm to
            # consume on its next forward call. The carry takes priority
            # over an explicit ``residual=`` arg (which is V1-style and is
            # only used when the layer's own _do_fused_forward already
            # picked the right pair).
            pending = getattr(self, "_v2_pending_carry", None)
            if pending is not None:
                self._v2_pending_carry = None
                x_pending, r_pending = pending
                gamma = self.weight
                if getattr(self, "zero_centered_gamma", False):
                    gamma = gamma + 1
                # Final LN / standalone RMSNorm: vocab GEMM (and any
                # non-skip-quant consumer) still reads BF16 y.
                norm_out, _xpr, preq = _run_rmsnorm_residual(
                    x_pending, r_pending, gamma, self.eps, skip_y_store=False
                )
                if preq is not None:
                    self._prequant_x = preq
                return norm_out
            if residual is None:
                return _orig_norm_forward(self, x)
            gamma = self.weight
            if getattr(self, "zero_centered_gamma", False):
                gamma = gamma + 1
            y, xpr, preq = _run_rmsnorm_residual(
                x, residual, gamma, self.eps, skip_y_store=False
            )
            if preq is not None:
                self._prequant_x = preq
            return y, xpr

        PrimusTurboRMSNorm.forward = _new_norm_forward
        PrimusTurboRMSNorm._fused_residual_patched = True
        _log("patched PrimusTurboRMSNorm.forward (residual= arg + V2 _pending_carry)")

    # ----- 2. Patch TransformerBlock.__init__ to wire V2 layer links --------
    if _v2_enabled() and not getattr(TransformerBlock, "_fused_residual_v2_init_patched", False):
        _orig_block_init = TransformerBlock.__init__

        def _v2_init(self, *args, **kwargs):
            _orig_block_init(self, *args, **kwargs)
            try:
                _wire_v2_layer_links(self)
            except Exception as exc:
                _log(f"V2 layer-link wiring raised {type(exc).__name__}: {exc}; " "block will run V1-only")

        TransformerBlock.__init__ = _v2_init
        TransformerBlock._fused_residual_v2_init_patched = True
        _log("patched TransformerBlock.__init__ (V2 layer-link wiring active)")

    # ----- 3. Replace TransformerLayer.forward with fused variant -----------
    if not getattr(TransformerLayer, "_fused_residual_patched", False):
        _orig_layer_forward = TransformerLayer.forward

        def _fused_layer_forward(self, *args, **kwargs):
            if not _can_fuse(self):
                # Per-layer fallback. If a carry was queued for *this* layer
                # but the layer itself can't fuse, drop it back to a real
                # add so the layer sees a correct hidden_states.
                _drain_carry_into_hidden_states(self, args, kwargs)
                return _orig_layer_forward(self, *args, **kwargs)
            try:
                return _do_fused_forward(self, *args, **kwargs)
            except Exception as exc:
                _log(
                    f"fused forward raised {type(exc).__name__}: {exc}; "
                    f"falling back to original forward for all layers"
                )
                TransformerLayer.forward = _orig_layer_forward
                return _orig_layer_forward(self, *args, **kwargs)

        TransformerLayer.forward = _fused_layer_forward
        TransformerLayer._fused_residual_patched = True
        if _v2_enabled():
            _log("patched TransformerLayer.forward (V1 ADD#1 + V2 cross-layer ADD#2 fusion active)")
        else:
            _log("patched TransformerLayer.forward (V1 in-layer ADD#1+norm fusion active)")
        if _r3_enabled():
            _log("R3 RMSNorm→MXFP4 enabled (PRIMUS_FUSED_RMSNORM_MXFP4)")

    _INSTALLED = True
    return True


# ---------------------------------------------------------------------------
# V2 layer-link wiring
# ---------------------------------------------------------------------------
def _wire_v2_layer_links(block: Any) -> None:
    """Annotate each layer in ``block`` with V2 navigation pointers.

    Sets per-layer attributes:
      * ``_v2_block``           -> back-reference to the owning block
      * ``_v2_next_layer``      -> the next TransformerLayer or None
      * ``_v2_is_last_layer``   -> True for the final layer in the block
      * ``_v2_final_layernorm`` -> reference to block.final_layernorm if any
                                   (only set on the last layer)
    """
    layers = getattr(block, "layers", None)
    if layers is None:
        return
    n = len(layers)
    if n == 0:
        return
    final_ln = getattr(block, "final_layernorm", None)
    # CRITICAL: nn.Module.__setattr__ auto-registers any nn.Module-valued
    # attribute as a child module. Setting layer._v2_next_layer = next_layer
    # would create cycles (each layer becomes a child of the previous one)
    # and blow up tree-walking ops like .cuda() / .children() with
    # RecursionError. Bypass with object.__setattr__ so these are plain
    # Python attributes invisible to nn.Module bookkeeping.
    for i, layer in enumerate(layers):
        nxt = layers[i + 1] if (i + 1) < n else None
        is_last = (i + 1) == n
        object.__setattr__(layer, "_v2_block", block)
        object.__setattr__(layer, "_v2_next_layer", nxt)
        object.__setattr__(layer, "_v2_is_last_layer", is_last)
        object.__setattr__(layer, "_v2_final_layernorm", final_ln if is_last else None)
        object.__setattr__(layer, "_v2_carry", None)


def _drain_carry_into_hidden_states(layer: Any, args: tuple, kwargs: dict) -> None:
    """If a V2 carry was left for ``layer`` but we are about to take the
    *original* (unpatched) forward, materialise the carry as a regular add
    into ``hidden_states`` so the original forward still gets the right
    activations.
    """
    carry = getattr(layer, "_v2_carry", None)
    if carry is None:
        return
    layer._v2_carry = None
    mlp_out, residual = carry
    new_hs = mlp_out + residual
    if args:
        # python tuples are immutable; the caller (TransformerBlock loop)
        # passes hidden_states via kwargs. We can't mutate the caller's
        # args tuple from here, so route via kwargs (the kwargs path is
        # always taken for hidden_states in current Megatron).
        kwargs["hidden_states"] = new_hs
        return
    kwargs["hidden_states"] = new_hs


# ---------------------------------------------------------------------------
# Per-layer "can we fuse?" guard
# ---------------------------------------------------------------------------
def _can_fuse(layer: Any) -> bool:
    """Return True when ``layer`` matches the assumptions of the fused path."""
    cfg = getattr(layer, "config", None)
    if cfg is None:
        return False
    if getattr(cfg, "fp32_residual_connection", False):
        return False
    if getattr(cfg, "inference_fuse_tp_communication", False):
        return False
    if getattr(layer, "recompute_input_layernorm", False):
        return False
    if getattr(layer, "recompute_pre_mlp_layernorm", False):
        return False
    if getattr(layer, "hidden_dropout", 0.0) != 0.0:
        return False
    if getattr(layer, "offload_attn_norm", False):
        return False
    if getattr(layer, "offload_mlp_norm", False):
        return False

    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboRMSNorm,
        )
    except ImportError:
        return False

    IdentityOp = _identity_op_type()
    cross_attn = getattr(layer, "cross_attention", None)
    if cross_attn is not None and not isinstance(cross_attn, IdentityOp):
        return False

    pre_mlp = getattr(layer, "pre_mlp_layernorm", None)
    if isinstance(pre_mlp, PrimusTurboRMSNorm):
        return True
    # Llama TE fused LN+Linear: IdentityOp pre_mlp, RMSNorm on fc1.
    if _is_identity(pre_mlp) and _fc1_fused_ln_params(layer) is not None:
        global _LOGGED_FC1_PATH
        if not _LOGGED_FC1_PATH:
            _log("V1 using mlp.linear_fc1.layer_norm_weight (IdentityOp pre_mlp)")
            _LOGGED_FC1_PATH = True
        return True
    return False


def _v2_next_can_consume_carry(layer: Any) -> bool:
    """Return True when layer N can stash an unfused carry instead of
    running mlp_bda. Requires that the next consumer (next layer's
    input_layernorm OR this block's final_layernorm) is a
    ``PrimusTurboRMSNorm`` so it can absorb the residual at no extra cost.
    """
    if not _v2_enabled():
        return False

    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboRMSNorm,
        )
    except ImportError:
        return False

    is_last = getattr(layer, "_v2_is_last_layer", False)
    if not is_last:
        nxt = getattr(layer, "_v2_next_layer", None)
        if nxt is None:
            return False
        nxt_in_ln = getattr(nxt, "input_layernorm", None)
        if not isinstance(nxt_in_ln, PrimusTurboRMSNorm) and _qkv_fused_ln_params(nxt) is None:
            return False
        # If the next layer can't fuse for its own reasons, we shouldn't
        # leave a carry it has to drain — drain it ourselves via the
        # _drain_carry_into_hidden_states fallback in the wrapper.
        return _can_fuse(nxt)
    # Last layer: route through final_layernorm if it's a PrimusTurboRMSNorm.
    final_ln = getattr(layer, "_v2_final_layernorm", None)
    if final_ln is None:
        return False
    return isinstance(final_ln, PrimusTurboRMSNorm)


# ---------------------------------------------------------------------------
# The fused forward
# ---------------------------------------------------------------------------
def _do_fused_forward(layer: Any, hidden_states=None, *args, **kwargs):
    """Mirror of ``TransformerLayer.forward`` with V1 + V2 fusion paths.

    V1: pre_mlp_layernorm receives ``(attn_out, residual)`` and emits both
        the normed activation AND ``x_plus_r`` for the next bda.
    V2: input_layernorm consumes a ``_v2_carry`` left by the previous
        layer, and at exit either stashes a new carry (next layer / final
        layernorm) or falls back to the explicit mlp_bda add.
    """
    from megatron.core.utils import (
        deprecate_inference_params,
        make_viewless_tensor,
        nvtx_range_pop,
        nvtx_range_push,
    )

    if hidden_states is None:
        hidden_states = kwargs.pop("hidden_states")
    else:
        kwargs.pop("hidden_states", None)

    inference_context = deprecate_inference_params(
        kwargs.get("inference_context"), kwargs.get("inference_params")
    )

    v2_active = _v2_enabled()

    # ---- input_layernorm (with optional V2 carry consume) ----------------
    nvtx_range_push(suffix="input_layernorm")
    carry = getattr(layer, "_v2_carry", None) if v2_active else None
    qkv_skip = None
    if carry is not None:
        # Fused path: absorb the previous layer's deferred mlp_bda add.
        layer._v2_carry = None
        prev_mlp_out, prev_residual = carry

        qkv_params = _qkv_fused_ln_params(layer)
        in_ln = getattr(layer, "input_layernorm", None)
        if qkv_params is not None and _is_identity(in_ln):
            qkv_skip, gamma, eps = qkv_params
            if getattr(qkv_skip, "zero_centered_gamma", False):
                gamma = gamma + 1
            input_layernorm_output, hidden_states, preq = _run_rmsnorm_residual(
                prev_mlp_out, prev_residual, gamma, eps
            )
            qkv_skip._skip_fused_norm = True
            if preq is not None:
                qkv_skip._prequant_x = preq
        else:
            # Standalone PrimusTurboRMSNorm input_layernorm.
            in_ln = layer.input_layernorm
            gamma = in_ln.weight
            if getattr(in_ln, "zero_centered_gamma", False):
                gamma = gamma + 1
            input_layernorm_output, hidden_states, preq = _run_rmsnorm_residual(
                prev_mlp_out, prev_residual, gamma, in_ln.eps
            )
            if preq is not None:
                in_ln._prequant_x = preq
    else:
        input_layernorm_output = layer.input_layernorm(hidden_states)
    nvtx_range_pop(suffix="input_layernorm")

    residual = hidden_states  # base for ADD#1 (post input_layernorm)

    # ---- self attention --------------------------------------------------
    nvtx_range_push(suffix="self_attention")
    attention_output_with_bias = layer.self_attention(
        input_layernorm_output,
        attention_mask=kwargs.get("attention_mask"),
        inference_context=inference_context,
        rotary_pos_emb=kwargs.get("rotary_pos_emb"),
        rotary_pos_cos=kwargs.get("rotary_pos_cos"),
        rotary_pos_sin=kwargs.get("rotary_pos_sin"),
        rotary_pos_cos_sin=kwargs.get("rotary_pos_cos_sin"),
        attention_bias=kwargs.get("attention_bias"),
        packed_seq_params=kwargs.get("packed_seq_params"),
        sequence_len_offset=kwargs.get("sequence_len_offset"),
    )
    nvtx_range_pop(suffix="self_attention")

    attn_out, attn_bias = attention_output_with_bias
    if attn_bias is not None:
        attn_out = attn_out + attn_bias.to(attn_out.dtype)

    # ---- V1 fuse: bda(attn_out, residual) + pre_mlp RMSNorm --------------
    nvtx_range_push(suffix="fused_residual_pre_mlp_layernorm")
    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboRMSNorm,
    )

    pre_mlp = getattr(layer, "pre_mlp_layernorm", None)
    fc1_skip = None
    if isinstance(pre_mlp, PrimusTurboRMSNorm):
        pre_mlp_layernorm_output, hidden_states = pre_mlp(attn_out, residual=residual)
    else:
        fc1_skip, gamma, eps = _fc1_fused_ln_params(layer)
        if getattr(fc1_skip, "zero_centered_gamma", False):
            gamma = gamma + 1
        pre_mlp_layernorm_output, hidden_states, preq = _run_rmsnorm_residual(
            attn_out, residual, gamma, eps
        )
        fc1_skip._skip_fused_norm = True
        if preq is not None:
            fc1_skip._prequant_x = preq
    nvtx_range_pop(suffix="fused_residual_pre_mlp_layernorm")
    # hidden_states now == attn_out + residual (= residual_post_attn).

    # ---- mlp -------------------------------------------------------------
    padding_mask = kwargs.get("padding_mask", None)
    try:
        try:
            mlp_output_with_bias = layer.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)
        except TypeError:
            mlp_output_with_bias = layer.mlp(pre_mlp_layernorm_output)
    finally:
        if fc1_skip is not None:
            fc1_skip._skip_fused_norm = False
            fc1_skip._prequant_x = None
        if qkv_skip is not None:
            qkv_skip._skip_fused_norm = False
            qkv_skip._prequant_x = None

    # ---- mlp_bda OR V2 carry stash ---------------------------------------
    if v2_active and _v2_next_can_consume_carry(layer):
        # Skip mlp_bda. Stash carry on the next consumer, return
        # residual_post_attn as the layer's "output" tensor (used only by
        # TransformerBlock for offload commit / make_viewless plumbing;
        # the next layer / final_layernorm reads the carry instead).
        mlp_out, mlp_bias = mlp_output_with_bias
        if mlp_bias is not None:
            mlp_out = mlp_out + mlp_bias.to(mlp_out.dtype)

        is_last = getattr(layer, "_v2_is_last_layer", False)
        if is_last:
            final_ln = layer._v2_final_layernorm
            final_ln._v2_pending_carry = (mlp_out, hidden_states)
        else:
            layer._v2_next_layer._v2_carry = (mlp_out, hidden_states)

        nvtx_range_push(suffix="v2_skip_mlp_bda")
        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )
        nvtx_range_pop(suffix="v2_skip_mlp_bda")
        return output, None

    # V1-only path (or V2 last-layer-without-norm-final): explicit mlp_bda.
    nvtx_range_push(suffix="mlp_bda")
    with layer.bias_dropout_add_exec_handler():
        hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
            mlp_output_with_bias, hidden_states, layer.hidden_dropout
        )
    nvtx_range_pop(suffix="mlp_bda")

    output = make_viewless_tensor(
        inp=hidden_states,
        requires_grad=hidden_states.requires_grad,
        keep_graph=True,
    )
    return output, None
