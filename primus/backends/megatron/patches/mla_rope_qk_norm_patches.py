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
far past the ``qk_clip`` threshold (5500). ``qk_clip`` cannot hold them: it
rescales the per-head weights (content q/k and the rotary q rows of
``linear_q_(up_)proj``), but the rotary key ``k_pos_emb`` comes straight out of
``linear_kv_down_proj``, is shared by all heads, and is never rescaled or
normalized, so the rotary logit term grows through it without bound and
detonates the deep residual stream.

Fix: apply a **parameter-free RMSNorm** (unit RMS over ``qk_pos_emb_head_dim``)
to ``q_pos_emb`` and ``k_pos_emb`` before the rotary embedding. Rotary is a
rotation, so this caps the rotary logit contribution at
``qk_pos_emb_head_dim`` (``mscale**2 * qk_pos_emb_head_dim`` with YaRN, whose
cos/sin carry the ``mscale`` factor) regardless of projection-weight growth --
the real, architectural fix rather than another logit-threshold band-aid.

Why parameter-free (no learnable gain): the run resumes mid-training from an
existing checkpoint. A learnable RMSNorm gain would add new parameters and
change the ``state_dict`` / distributed-optimizer param layout, breaking a
clean resume. Unit-RMS with no parameters keeps the checkpoint and optimizer
shards byte-compatible while still capping the runaway.

Scope: this rewrites the non-fused training/eval path
(``apply_rope_fusion: false``, ``cache_mla_latents: false``), which is what the
hybrid uses. The patch logs a warning and no-ops when RoPE fusion is in effect
(the fused kernel applies rotary internally; the hybrids' inherited
``apply_rope_fusion: true`` does not count, since Megatron drops it for
``position_embedding_type: none``), when MLA latents are cached (a
different q/k path this anchor was not written for), and for NoPE MLA
(``mla_use_nope``, the Kimi K3 default), whose 64 positional dims are an
unrotated pass-through that must stay bit-exact.

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
                # head's rotary vector to unit RMS (rotary is a rotation, so this
                # bounds the rotary logit contribution at qk_pos_emb_head_dim,
                # times mscale**2 under YaRN) without adding parameters -- keeps
                # the checkpoint/optimizer layout identical for a clean resume.
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
    target = next(c for c in MLASelfAttention.__mro__ if "get_query_key_value_tensors" in c.__dict__)

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
    if not getattr(args, "mla_rope_qk_norm", False) and os.environ.get("PRIMUS_MLA_ROPE_QK_NORM", "0") != "1":
        return False
    # The fused RoPE kernel applies rotary internally; this source patch only
    # covers the explicit non-fused split path. Warn + no-op if fusion is on.
    # This runs before Megatron's validate_args (called inside pretrain), which
    # forces apply_rope_fusion off unless position_embedding_type == "rope". The
    # MLA hybrids inherit apply_rope_fusion: true with position_embedding_type:
    # none and still run the non-fused MLA rotary path, so judge the effective value.
    rope_fused = (
        getattr(args, "apply_rope_fusion", False)
        and getattr(args, "position_embedding_type", "rope") == "rope"
    )
    if rope_fused:
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] apply_rope_fusion=True; QK-Norm patch only "
            "supports the non-fused path -- skipping (rotary q/k stay unnormalized)."
        )
        return False
    if getattr(args, "cache_mla_latents", False):
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] cache_mla_latents=True; QK-Norm patch only "
            "supports the non-cached path -- skipping (rotary q/k stay unnormalized)."
        )
        return False
    # NoPE MLA (Kimi K3) still routes its positional dims through this block but
    # relies on a zero-width rotary table being a bit-exact pass-through, so
    # normalizing them would change the model. KimiK3TransformerConfig defaults
    # mla_use_nope to True, so a K3 run without the arg is NoPE too.
    use_nope = getattr(args, "mla_use_nope", None)
    if use_nope is None:
        use_nope = getattr(args, "model_type", None) == "kimi_k3"
    if use_nope:
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] mla_use_nope=True; there is no rotary component "
            "to normalize -- skipping."
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
    # ``run_patches`` swallows exceptions (stop_on_error=False) and continues with
    # the upstream method, which would silently train without a stabilizer the
    # user explicitly enabled (e.g. after upstream source drift breaks the
    # anchor). Escalate to a hard abort; ``SystemExit`` is a ``BaseException``
    # and is not caught by ``run_patches``.
    try:
        _install_mla_rope_qk_norm_patch()
    except Exception as exc:
        log_rank_0(f"[Patch:{_PATCH_KEY}] FATAL: {exc}")
        raise SystemExit(f"[{_PATCH_KEY}] QK-Norm patch installation failed: {exc}") from exc
