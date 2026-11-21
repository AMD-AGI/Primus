###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo MegaUlysses Patches

Ulysses context parallelism with the all-to-alls fused into the attention projections
(``use_turbo_mega_ulysses: true``; CP > 1 within a node, TP = 1). Self-attention then runs

    x [s/cp, b, h] --ulysses_qkv_gemm_a2a--> q, k, v [s, b, h/cp, d] (local heads, full sequence)
      -> q/k layernorm -> RoPE over the full sequence -> flash_attn_func (local, causal)
      --ulysses_a2a_o_gemm--> [s/cp, b, h]

so the QKV / o_proj GEMMs overlap their all-to-alls (GemmA2A / A2AGemm) and the pack / unpack
copies disappear. The fused ops shard the sequence contiguously (rank r holds tokens
[r*s/cp, (r+1)*s/cp)), so the CP batch split is switched from Megatron's zigzag to contiguous
chunks; with Ulysses every rank attends over the whole sequence, so the zigzag load balancing buys
nothing.

With MLA, ``MLASelfAttention`` runs the latent formulation instead: the q / kv down-projections
and norms stay on the local tokens, k_pe is rotated at its own positions, and the latents
[q_c | c_kv | k_pe] (2112 elements a token for DeepSeek-V3) are all-gathered into the local-head
up-projections (``ulysses_mla_latent_up_proj``, AgGemm / GemmRS), q_pe is rotated over the full
sequence, and the exit is ``ulysses_a2a_o_gemm``.

``turbo_mega_ulysses_impl`` picks the implementation on this same attention path, for A/B runs:
``fused`` (the fused kernels), ``serial`` (the same GEMMs with RCCL collectives in series) and, for
MLA only, ``per_head`` (Megatron's per-head Ulysses: every head up-projected on the local tokens,
then one all-to-all of q, k, v).

The split and the RoPE slicing are global, so no layer can fall back to the stock attention once
the patch is in: every unsupported setting known from the arguments keeps the whole run on the stock
path, and the ones only seen at call time (packed sequences, a non-causal mask, ...) raise.
"""

import importlib.util
import os

import torch
import torch.nn.functional as F

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH = "[Patch:megatron.turbo.mega_ulysses]"
_METADATA_KEYS = ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "local_cp_size", "hybrid_cp_group")
_GQA_OPS = "primus_turbo.pytorch.ops.attention.ulysses_fused_a2a"
_MLA_OPS = "primus_turbo.pytorch.ops.attention.ulysses_mla_latent"


def _impl(args) -> str:
    impl = getattr(args, "turbo_mega_ulysses_impl", None) or "fused"
    valid = (
        ("fused", "serial", "per_head")
        if getattr(args, "multi_latent_attention", False)
        else ("fused", "serial")
    )
    assert impl in valid, f"turbo_mega_ulysses_impl must be one of {valid}, got {impl!r}"
    return impl


_UNSUPPORTED_ARGS = (
    ("experimental_attention_variant", "experimental attention variants"),
    ("add_qkv_bias", "a QKV bias"),
    ("add_bias_linear", "linear biases"),
    ("attention_output_gate", "the attention output gate"),
    ("window_size", "sliding-window attention"),
    ("attention_dropout", "attention dropout"),
    (
        "use_turbo_fused_qk_rmsnorm_rope",
        "use_turbo_fused_qk_rmsnorm_rope (also replaces SelfAttention.forward)",
    ),
)


def _unsupported(args):
    """Why this run keeps the stock CP path, or None."""
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    # the fused kernels exchange over XGMI peer memory, so every CP group must sit inside one node;
    # with TP = 1 and Megatron's tp-cp-ep-dp-pp order a CP group is `cp` consecutive ranks
    if tp != 1 or cp == 1 or local_world % cp != 0:
        return f"needs TP=1 and CP>1 dividing {local_world} GPUs/node (got TP={tp}, CP={cp})"
    for name, what in _UNSUPPORTED_ARGS:
        if getattr(args, name, None):
            return f"{what} ({name}={getattr(args, name)}) is not supported"
    normalization = getattr(args, "normalization", "LayerNorm")
    if normalization not in ("RMSNorm", "LayerNorm"):
        return f"normalization={normalization} is not supported"
    if getattr(args, "multi_latent_attention", False):
        if getattr(args, "q_lora_rank", None) is None:
            return "MLA without a q LoRA (q_lora_rank=None) is not supported"
        if getattr(args, "cache_mla_latents", False):
            return "cache_mla_latents is not supported"
        if "mla_up_proj" in (getattr(args, "recompute_modules", None) or []):
            return "recomputing the MLA up-projection (recompute_modules: mla_up_proj) is not supported"
    return None


def _missing_ops(args):
    """The Primus-Turbo ops module this run needs and the installed Turbo lacks, or None."""
    needed = [_GQA_OPS] + ([_MLA_OPS] if getattr(args, "multi_latent_attention", False) else [])
    for name in needed:
        try:
            if importlib.util.find_spec(name) is None:
                return name
        except ModuleNotFoundError:
            return name
    return None


def _is_turbo_mega_ulysses_can_patch(ctx: PatchContext) -> bool:
    """
    Check if MegaUlysses is enabled and this run can take it.

    Requires:
      - primus_turbo package is installed
      - enable_primus_turbo == True
      - use_turbo_mega_ulysses == True
      - TP == 1, 1 < CP within a node, and none of the settings in ``_unsupported``
      - the installed Primus-Turbo provides the Ulysses ops (MLA needs ``ulysses_mla_latent``)
    """
    args = get_args(ctx)
    if not bool(getattr(args, "use_turbo_mega_ulysses", False)):
        return False
    _impl(args)
    reason = _unsupported(args)
    if reason is not None:
        warning_rank_0(f"{_PATCH} {reason}, keeping the stock CP path")
        return False
    if not is_primus_turbo_can_patch(ctx):
        return False
    missing = _missing_ops(args)
    if missing is not None:
        warning_rank_0(f"{_PATCH} the installed Primus-Turbo has no {missing}, keeping the stock CP path")
        return False
    return True


def _contiguous_cp_split(batch, cp_group):
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)
    if cp_size == 1:
        return batch
    for key, val in batch.items():
        if key in _METADATA_KEYS or val is None:
            continue
        seq_dim = 2 if key == "attention_mask" else 1
        chunk = val.size(seq_dim) // cp_size
        batch[key] = val.narrow(seq_dim, cp_rank * chunk, chunk)
    return batch


def _has_bias(linear) -> bool:
    # TE linears without a bias may still carry an empty `bias` tensor
    bias = getattr(linear, "bias", None)
    return isinstance(bias, torch.Tensor) and bias.numel() > 0


def _input_norm(linear, x):
    """The input norm TE fuses into ``linear_qkv`` (LayerNormLinear), applied as its forward would."""
    weight = getattr(linear, "layer_norm_weight", None)
    # fused_residual_rmsnorm hands over x already normalized
    if weight is None or getattr(linear, "_skip_fused_norm", False):
        return x
    if getattr(linear, "zero_centered_gamma", False):
        weight = weight + 1
    eps = linear.eps
    if linear.normalization == "RMSNorm":
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    if linear.normalization == "LayerNorm":
        return F.layer_norm(x, (x.shape[-1],), weight, linear.layer_norm_bias, eps)
    raise RuntimeError(f"{_PATCH} unsupported normalization {linear.normalization!r}")


def _run_forward_pre_hooks(module):
    # The fused path reads the module's weights without calling it, so run its forward pre-hooks the way
    # nn.Module.__call__ would; with overlap_param_gather, DDP waits there for the weights' all-gather.
    for hook_id, hook in module._forward_pre_hooks.items():
        if hook_id in module._forward_pre_hooks_with_kwargs:
            hook(module, (), {})
        else:
            hook(module, ())


def _raise_unsupported(unsupported, what):
    reasons = [name for name, hit in unsupported.items() if hit]
    if reasons:
        raise NotImplementedError(
            f"{_PATCH} {', '.join(reasons)} not supported{what}; the batch is already split contiguously "
            "for MegaUlysses, so the stock attention cannot run either. Set use_turbo_mega_ulysses: false"
        )


class _NoCPGroup:
    """RoPE's cp_group stand-in after the all-to-all: q and k already span the whole sequence."""

    def size(self):
        return 1

    def rank(self):
        return 0


def _a2a_helper(b, seq, hq, hkv, d, n):
    from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
        get_attention_cp_a2a_helper,
    )

    return get_attention_cp_a2a_helper(b, seq, hq, hkv, d, d, 0, n)


def _a2a(t, group):
    from torch.distributed.nn.functional import all_to_all_single

    return all_to_all_single(torch.empty_like(t), t.contiguous(), group=group)


def _serial_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout):
    """Unfused ``ulysses_qkv_gemm_a2a``: GEMM, pack, all-to-all, unpack in series."""
    s_local, b = x.shape[:2]
    helper = _a2a_helper(b, s_local * group.size(), hq, hkv, d, group.size())
    qkv = F.linear(x, w_qkv).view(s_local, b, hkv, hq // hkv + 2, d)
    q = qkv[..., : hq // hkv, :].reshape(s_local, b, hq, d)
    k, v = qkv[..., hq // hkv, :], qkv[..., hq // hkv + 1, :]
    return helper.splits_qkv_after_a2a(_a2a(helper.combine_qkv_before_a2a(q, k, v), group))


def _serial_a2a_o_gemm(o, w_o, residual, group, qkv_format):
    """Unfused ``ulysses_a2a_o_gemm``."""
    seq, b, heads, d = o.shape
    n = group.size()
    helper = _a2a_helper(b, seq, heads * n, heads * n, d, n)
    o_gathered = helper.reshape_o_after_a2a(_a2a(helper.reshape_o_before_a2a(o), group))
    return F.linear(o_gathered.reshape(*o_gathered.shape[:2], -1), w_o)


def _make_gqa_forward(args):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.transformer.enums import AttnMaskType
    from primus_turbo.pytorch.ops import flash_attn_func
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
        ulysses_a2a_o_gemm,
        ulysses_qkv_gemm_a2a,
    )

    no_cp = _NoCPGroup()
    if _impl(args) == "serial":
        qkv_op, o_op = _serial_qkv_gemm_a2a, _serial_a2a_o_gemm
    else:
        qkv_op, o_op = ulysses_qkv_gemm_a2a, ulysses_a2a_o_gemm

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        cfg = self.config
        cp_group = self.pg_collection.cp
        _raise_unsupported(
            {
                "cross attention": key_value_states is not None,
                "inference": inference_context is not None or inference_params is not None,
                "packed sequences": packed_seq_params is not None,
                "an attention bias": attention_bias is not None,
                f"attn_mask_type={self.attn_mask_type.name}": self.attn_mask_type != AttnMaskType.causal,
                "projection biases": _has_bias(self.linear_qkv) or _has_bias(self.linear_proj),
            },
            "",
        )

        hq, hkv, d = cfg.num_attention_heads, cfg.num_query_groups, cfg.kv_channels
        _run_forward_pre_hooks(self.linear_qkv)
        x = _input_norm(self.linear_qkv, hidden_states)
        q, k, v = qkv_op(x, self.linear_qkv.weight, hq, hkv, d, cp_group, "sbhd", "interleaved")
        if self.q_layernorm is not None:
            q = self.q_layernorm(q)
        if self.k_layernorm is not None:
            k = self.k_layernorm(k)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = (
                rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
            )
            q = apply_rotary_pos_emb(q, q_pos_emb, config=cfg, cp_group=no_cp)
            k = apply_rotary_pos_emb(k, k_pos_emb, config=cfg, cp_group=no_cp)
        # sbhd -> bshd views for flash-attn
        o = flash_attn_func(
            q.permute(1, 0, 2, 3),
            k.permute(1, 0, 2, 3),
            v.permute(1, 0, 2, 3),
            dropout_p=0.0,
            softmax_scale=cfg.softmax_scale,
            causal=True,
        )
        o = o.permute(1, 0, 2, 3).contiguous()
        _run_forward_pre_hooks(self.linear_proj)
        y = o_op(o, self.linear_proj.weight, None, cp_group, "sbhd")
        return y, None

    return forward


def _seq_to_heads(tensors, group):
    """``[s/n, b, h, d_i]`` tensors of this rank's tokens -> ``[s, b, h/n, d_i]`` of its heads, in one
    all-to-all (Megatron's per-head Ulysses)."""
    n = group.size()
    parts = [
        t.view(t.shape[0], t.shape[1], n, t.shape[2] // n, t.shape[3]).movedim(2, 0).reshape(n, -1)
        for t in tensors
    ]
    flat = _a2a(torch.cat(parts, dim=1), group)
    outs = flat.split([p.shape[1] for p in parts], dim=1)
    return [o.reshape(t.shape[0] * n, t.shape[1], t.shape[2] // n, t.shape[3]) for o, t in zip(outs, tensors)]


def _make_mla_forward(args):
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.transformer.enums import AttnMaskType
    from primus_turbo.pytorch.ops import flash_attn_func
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import ulysses_a2a_o_gemm
    from primus_turbo.pytorch.ops.attention.ulysses_mla_latent import (
        ulysses_mla_latent_up_proj,
    )

    impl = _impl(args)
    no_cp = _NoCPGroup()
    o_op = ulysses_a2a_o_gemm if impl == "fused" else _serial_a2a_o_gemm

    def rope(t, emb, cfg, mscale):
        return apply_rotary_pos_emb(
            t, emb, config=cfg, mscale=mscale, cp_group=no_cp, mla_rotary_interleaved=True
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        cfg = self.config
        cp_group = self.pg_collection.cp
        _raise_unsupported(
            {
                "cross attention": key_value_states is not None,
                "inference": inference_context is not None or inference_params is not None,
                "packed sequences": packed_seq_params is not None,
                "an attention bias": attention_bias is not None,
                f"attn_mask_type={self.attn_mask_type.name}": self.attn_mask_type != AttnMaskType.causal,
                "projection biases": any(
                    _has_bias(m) for m in (self.linear_q_up_proj, self.linear_kv_up_proj, self.linear_proj)
                ),
            },
            " with MLA",
        )

        n, rank = cp_group.size(), cp_group.rank()
        s_local, b = hidden_states.shape[:2]
        heads, dn, dr, dv = cfg.num_attention_heads, cfg.qk_head_dim, cfg.qk_pos_emb_head_dim, cfg.v_head_dim

        q_c, kv_combined = self._qkv_down_projection(hidden_states)
        c_kv, k_pe = torch.split(kv_combined, [cfg.kv_lora_rank, dr], dim=-1)
        _run_forward_pre_hooks(self.linear_q_up_proj)
        _run_forward_pre_hooks(self.linear_kv_up_proj)
        q_c = _input_norm(self.linear_q_up_proj, self.q_layernorm(q_c))
        c_kv = _input_norm(self.linear_kv_up_proj, self.kv_layernorm(c_kv))
        w_q, w_kv = self.linear_q_up_proj.weight, self.linear_kv_up_proj.weight

        # full-sequence frequencies: a size-1 group keeps YaRN from slicing them zigzag
        emb = self.rotary_pos_emb(s_local * n, cp_group=no_cp)
        emb, mscale = emb if isinstance(emb, tuple) else (emb, 1.0)
        k_pe = rope(k_pe.unsqueeze(-2), emb[rank * s_local : (rank + 1) * s_local], cfg, mscale).squeeze(-2)

        if impl == "per_head":
            q = F.linear(q_c, w_q).view(s_local, b, heads, dn + dr)
            k_nope, v = F.linear(c_kv, w_kv).view(s_local, b, heads, dn + dv).split([dn, dv], dim=-1)
            k = torch.cat((k_nope, k_pe.unsqueeze(-2).expand(s_local, b, heads, dr)), dim=-1)
            q, k, v = _seq_to_heads([q, k, v], cp_group)
        else:
            kv_latent = torch.cat((c_kv, k_pe), dim=-1)
            q, k, v = ulysses_mla_latent_up_proj(
                q_c, kv_latent, w_q, w_kv, heads, cp_group, dn, dr, dv, "sbhd", fused=impl == "fused"
            )
        q_nope, q_pe = q.split([dn, dr], dim=-1)
        q = torch.cat((q_nope, rope(q_pe, emb, cfg, mscale)), dim=-1)

        o = flash_attn_func(
            q.permute(1, 0, 2, 3),
            k.permute(1, 0, 2, 3),
            v.permute(1, 0, 2, 3),
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=True,
        )
        o = o.permute(1, 0, 2, 3).contiguous()
        _run_forward_pre_hooks(self.linear_proj)
        return o_op(o, self.linear_proj.weight, None, cp_group, "sbhd"), None

    return forward


_IMPL_DESCRIPTION = {
    (False, "fused"): "GemmA2A / A2AGemm",
    (False, "serial"): "GEMM + RCCL all-to-all in series",
    (True, "fused"): "latent AgGemm / GemmRS + A2AGemm / GemmA2A",
    (True, "serial"): "latent RCCL all-gather + GEMM, o all-to-all + GEMM in series",
    (True, "per_head"): "per-head q/k/v all-to-all (Megatron Ulysses), o all-to-all + GEMM in series",
}


@register_patch(
    "megatron.turbo.mega_ulysses",
    backend="megatron",
    phase="before_train",
    description="Ulysses CP with the all-to-alls fused into the attention projections (MegaUlysses)",
    condition=_is_turbo_mega_ulysses_can_patch,
)
def patch_mega_ulysses(ctx: PatchContext):
    import megatron.core.models.common.embeddings.rotary_pos_embedding as rope_mod
    import megatron.core.utils as mcore_utils

    args = get_args(ctx)
    mla = bool(getattr(args, "multi_latent_attention", False))
    # builds the forward (importing the Turbo ops) first: a failure must come before anything global is switched
    forward = _make_mla_forward(args) if mla else _make_gqa_forward(args)
    mcore_utils._get_batch_on_this_cp_rank_per_sequence_balancing = _contiguous_cp_split
    # RoPE runs after the all-to-all over the full sequence, so keep the full-length freqs
    rope_mod.get_pos_emb_on_this_cp_rank = lambda emb, seq_dim, cp_group: emb
    if mla:
        from megatron.core.transformer.multi_latent_attention import (
            MLASelfAttention as attention_cls,
        )
    else:
        from megatron.core.transformer.attention import SelfAttention as attention_cls
    attention_cls.forward = forward
    log_rank_0(
        f"{_PATCH} {attention_cls.__name__} -> {_IMPL_DESCRIPTION[(mla, _impl(args))]}, contiguous CP split"
    )
