###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MLA decoupled-RoPE QK-Norm Patch
================================

In Multi-Latent Attention the query/key are split into a *content* part
(``qk_head_dim``) and a *decoupled RoPE* part (``qk_pos_emb_head_dim``). The
content part is normalized (``q_layernorm`` / ``kv_layernorm`` on the
compressed latents), but the **rotary part is not normalized anywhere**: it is
projected, split off, and fed straight into ``apply_rotary_pos_emb``.

For the 3B KDA-RoPE hybrid this is the divergence mechanism. Per-layer
instrumentation showed that around iter ~22k the deepest ~7 residual layers
(L49-L55) blow up *together* while the attention logits shoot to 16k-37k --
far past the ``qk_clip`` threshold (5500). ``qk_clip`` cannot hold them because
it only rescales the (normalized) content projections; the unnormalized rotary
component escapes it entirely, so its logit contribution grows without bound
and detonates the deep residual stream.

Fix: apply a **parameter-free RMSNorm** (unit RMS over ``qk_pos_emb_head_dim``)
to ``q_pos_emb`` and ``k_pos_emb`` before the rotary embedding. Since rotary is
norm-preserving, this caps the rotary logit contribution to
``<= qk_pos_emb_head_dim`` regardless of projection-weight growth -- the real,
architectural fix rather than another logit-threshold band-aid.

Why parameter-free (no learnable gain): the run resumes mid-training from an
existing checkpoint. A learnable RMSNorm gain would add new parameters and
change the ``state_dict`` / distributed-optimizer param layout, breaking a
clean resume. Unit-RMS with no parameters keeps the checkpoint and optimizer
shards byte-compatible while still capping the runaway.

Scope: this rewrites the non-fused training/eval path
(``apply_rope_fusion: false``, ``cache_mla_latents: false``), which is what the
hybrid uses. If RoPE fusion is enabled the patch logs a warning and no-ops,
since the fused kernel applies rotary internally.

This is a "source-string rewrite" patch (see ``_source_patch_utils``): the edit
sits in the middle of ``MLASelfAttention.get_query_key_value_tensors``'s nested
``qkv_up_proj_and_rope_apply``, where a wrapping monkey-patch cannot reach.
``PrimusMLASelfAttention`` inherits this method unchanged, so patching the base
``MLASelfAttention`` covers the hybrid too.

Enabled via the ``mla_rope_qk_norm`` config flag, or the
``PRIMUS_MLA_ROPE_QK_NORM=1`` env var (so it needs no config-schema change); a
no-op otherwise.
"""

import os

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._source_patch_utils import patch_method_source
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.transformer.mla_rope_qk_norm"

# Anchor: the non-fused q/k split immediately followed by the rotary apply on
# the query. We insert a parameter-free RMSNorm on the decoupled-RoPE q/k parts
# between the split and the rotary embedding. Written at the upstream file's
# absolute (16-space) indentation -- patch_method_source replaces on the raw,
# non-dedented source.
_ORI = """\
                # k_no_pe: [num_tokens, n, qk_head_dim]
                # value: [num_tokens, n, v_head_dim]
                k_no_pe, value = torch.split(
                    kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
                )

                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_pos_emb = apply_rotary_pos_emb("""

_NEW = """\
                # k_no_pe: [num_tokens, n, qk_head_dim]
                # value: [num_tokens, n, v_head_dim]
                k_no_pe, value = torch.split(
                    kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
                )

                # --- MLA decoupled-RoPE QK-Norm (parameter-free) ---
                # The rotary q/k parts bypass q_layernorm/kv_layernorm and also
                # escape qk_clip, so their magnitude can grow unbounded and drive
                # the attention-logit / deep-residual explosion. Normalize each
                # head's rotary vector to unit RMS (rotary is norm-preserving, so
                # this bounds the rotary logit contribution to qk_pos_emb_head_dim)
                # without adding parameters -- keeps the checkpoint/optimizer
                # layout identical for a clean mid-training resume.
                _qkn_eps = self.config.layernorm_epsilon
                q_pos_emb = q_pos_emb * torch.rsqrt(
                    q_pos_emb.float().pow(2).mean(-1, keepdim=True) + _qkn_eps
                ).to(q_pos_emb.dtype)
                k_pos_emb = k_pos_emb * torch.rsqrt(
                    k_pos_emb.float().pow(2).mean(-1, keepdim=True) + _qkn_eps
                ).to(k_pos_emb.dtype)

                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_pos_emb = apply_rotary_pos_emb("""


def _install_mla_rope_qk_norm_patch() -> None:
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention

    # Another patch may rebind this module attribute to a subclass
    # (PrimusMLASelfAttention). ``get_query_key_value_tensors`` is defined on the
    # upstream base and prebuilt hybrid layers may be base instances, so patch the
    # class that actually defines the method (walk the MRO) rather than whatever
    # the attribute currently points at.
    target = next(
        c for c in MLASelfAttention.__mro__ if "get_query_key_value_tensors" in c.__dict__
    )

    if is_patched(target, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}] already applied; skipping.")
        return

    patch_method_source(target, "get_query_key_value_tensors", _ORI, _NEW)

    mark_patched(target, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Applied parameter-free RMSNorm to the MLA "
        "decoupled-RoPE q/k components (non-fused path)."
    )


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    # Enable via the ``mla_rope_qk_norm`` config flag if present, or the
    # ``PRIMUS_MLA_ROPE_QK_NORM=1`` env var (works without a config-schema change).
    if not getattr(args, "mla_rope_qk_norm", False) and os.environ.get(
        "PRIMUS_MLA_ROPE_QK_NORM", "0"
    ) != "1":
        return False
    # The fused RoPE kernel applies rotary internally; this source patch only
    # covers the explicit non-fused split path. Warn + no-op if fusion is on.
    if getattr(args, "apply_rope_fusion", False):
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] apply_rope_fusion=True; QK-Norm patch only "
            "supports the non-fused path -- skipping (rotary q/k stay unnormalized)."
        )
        return False
    return True


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Parameter-free RMSNorm (QK-Norm) on the MLA decoupled-RoPE q/k "
        "components. Caps the rotary attention-logit contribution that bypasses "
        "q_layernorm/kv_layernorm and escapes qk_clip, fixing the deep-residual "
        "divergence in the KDA-RoPE hybrid. Enabled via the mla_rope_qk_norm "
        "config flag or PRIMUS_MLA_ROPE_QK_NORM=1."
    ),
    condition=_enabled,
)
def patch_mla_rope_qk_norm(ctx: PatchContext) -> None:
    _install_mla_rope_qk_norm_patch()
