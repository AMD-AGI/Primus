###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route each residual add (``self_attn_bda`` / ``mlp_bda``) and the RMSNorm that the residual
stream meets next through one bit-exact fused add+norm.

See :mod:`primus.backends.megatron.core.extensions.residual_add_norm`. At block construction
every layer learns the Turbo RMSNorm its attention output meets next (``pre_mlp_layernorm``,
or the norm inside ``mlp.linear_fc1`` when the pre-MLP norm is fused there) and the one its
MLP output meets next (the next layer's norm inside ``self_attention.linear_qkv``, or the
block's ``final_layernorm``). The wrapped bda computes ``h`` and ``norm(h)`` in one pass and
leaves ``norm(h)`` in ``residual_add_norm.PENDING``; ``PrimusTurboRMSNorm.forward`` returns it
when it is asked to normalise that very ``h`` (anything else computes normally). A bias,
dropout, recompute, offloading or a non-Turbo norm keeps the original code.

``PRIMUS_FUSED_ADD_NORM=0`` disables it (it also needs ``PRIMUS_TURBO_RMSNORM_BWD_CONFIG``).
"""

from __future__ import annotations

import functools
import os

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _wrap_bda(layer, orig_bda, attr):
    from primus.backends.megatron.core.extensions import residual_add_norm as ran

    def factory(training, fused):
        inner = orig_bda(training, fused)

        def bda(x_with_bias, residual, prob):
            x, bias = x_with_bias
            norm = getattr(layer, attr, None)
            if (
                norm is not None
                and bias is None
                and (prob == 0.0 or not training)
                and torch.is_grad_enabled()
                and ran.eligible(norm, x, residual)
            ):
                h, y = ran.add_norm(norm, x, residual)
                ran.PENDING[id(norm)] = (h, y)
                ran.STATS["made"] += 1
                if not ran.STATS.get("logged") and ran.STATS["made"] >= 16:
                    ran.STATS["logged"] = True
                    log_rank_0(
                        f"[Patch:megatron.residual_add_norm_bitwise] fused add+norm made {ran.STATS['made']}, "
                        f"used {ran.STATS['used']}"
                    )
                return h
            return inner(x_with_bias, residual, prob)

        return bda

    return factory


def _turbo_norm(m):
    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

    return m if type(m) is PrimusTurboRMSNorm else None


def _attn_out_norm(layer):
    from megatron.core.transformer.identity_op import IdentityOp

    pre = getattr(layer, "pre_mlp_layernorm", None)
    if _turbo_norm(pre) is not None:
        return pre
    if isinstance(pre, IdentityOp):
        fc1 = getattr(getattr(layer, "mlp", None), "linear_fc1", None)
        return _turbo_norm(getattr(fc1, "layernorm", None))
    return None


def _layer_in_norm(layer):
    from megatron.core.transformer.identity_op import IdentityOp

    inp = getattr(layer, "input_layernorm", None)
    if _turbo_norm(inp) is not None:
        return inp
    if isinstance(inp, IdentityOp):
        qkv = getattr(getattr(layer, "self_attention", None), "linear_qkv", None)
        return _turbo_norm(getattr(qkv, "layernorm", None))
    return None


@register_patch(
    "megatron.residual_add_norm_bitwise",
    backend="megatron",
    phase="before_train",
    description="Fuse each residual add with the RMSNorm that follows it, bit-exact (PRIMUS_FUSED_ADD_NORM).",
    condition=lambda ctx: os.environ.get("PRIMUS_FUSED_ADD_NORM", "1") != "0"
    and bool(os.environ.get("PRIMUS_TURBO_RMSNORM_BWD_CONFIG")),
)
def patch_residual_add_norm(ctx: PatchContext) -> None:
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if getattr(TransformerLayer, "_fe_add_norm_patched", False):
        return
    orig_attn = TransformerLayer._forward_attention
    orig_mlp = TransformerLayer._forward_mlp
    orig_block_init = TransformerBlock.__init__

    @functools.wraps(orig_attn)
    def _forward_attention(self, *args, **kwargs):
        if not getattr(self, "_fe_attn_bda_wrapped", False):
            self.self_attn_bda = _wrap_bda(self, self.self_attn_bda, "_fe_attn_out_norm")
            self._fe_attn_bda_wrapped = True
        return orig_attn(self, *args, **kwargs)

    @functools.wraps(orig_mlp)
    def _forward_mlp(self, *args, **kwargs):
        if not getattr(self, "_fe_mlp_bda_wrapped", False):
            self.mlp_bda = _wrap_bda(self, self.mlp_bda, "_fe_mlp_out_norm")
            self._fe_mlp_bda_wrapped = True
        return orig_mlp(self, *args, **kwargs)

    @functools.wraps(orig_block_init)
    def _block_init(self, *args, **kwargs):
        orig_block_init(self, *args, **kwargs)
        try:
            layers = list(self.layers)
            for i, layer in enumerate(layers):
                if getattr(layer, "recompute_pre_mlp_layernorm", False) or getattr(layer, "offload_mlp_norm", False):
                    layer._fe_attn_out_norm = None
                else:
                    layer._fe_attn_out_norm = _attn_out_norm(layer)
                if i + 1 < len(layers):
                    nxt = layers[i + 1]
                    ok = not getattr(nxt, "recompute_input_layernorm", False) and not getattr(
                        nxt, "offload_attn_norm", False
                    )
                    layer._fe_mlp_out_norm = _layer_in_norm(nxt) if ok else None
                else:
                    layer._fe_mlp_out_norm = _turbo_norm(getattr(self, "final_layernorm", None))
            log_rank_0(
                "[Patch:megatron.residual_add_norm_bitwise] targets: "
                + ", ".join(
                    f"L{i}:{l._fe_attn_out_norm is not None}/{l._fe_mlp_out_norm is not None}"
                    for i, l in enumerate(layers)
                )
            )
        except Exception as e:  # never break model construction over an optimisation
            log_rank_0(f"[Patch:megatron.residual_add_norm_bitwise] off ({e!r})")

    TransformerBlock.__init__ = _block_init
    TransformerLayer._forward_attention = _forward_attention
    TransformerLayer._forward_mlp = _forward_mlp
    TransformerLayer._fe_add_norm_patched = True
    log_rank_0("[Patch:megatron.residual_add_norm_bitwise] installed")
