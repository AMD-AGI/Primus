###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Portions of this file are adapted from NVIDIA Megatron-LM
# (https://github.com/NVIDIA/Megatron-LM),
# megatron/core/transformer/multi_latent_attention.py; and from Moonshot AI
# Kimi-Linear (https://huggingface.co/moonshotai/Kimi-K3),
# modeling_kimi_linear.py.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 full-attention layers — NoPE MLA with a sigmoid output gate.

24 of the 93 decoder layers use ``KimiMLAAttention``
(``modeling_kimi_linear.py:335-474``); the other 69 use KDA
(:mod:`.kimi_delta_attention`). Structurally it is DeepSeek-V3 MLA with
exactly two deltas, so this class *keeps* the upstream
:class:`~megatron.core.transformer.multi_latent_attention.MLASelfAttention`
``__init__`` — unlike :class:`DeepseekV4Attention`, which bypasses the
parent (``deepseek_v4_attention.py:505``) because V4 replaced MLA's
compressed KV with a single latent. Kimi K3's parameter layout is MLA's,
projection for projection::

    q_a_proj            -> linear_q_down_proj      7168 -> 1536
    q_a_layernorm       -> q_layernorm             RMSNorm(1536)
    q_b_proj            -> linear_q_up_proj        1536 -> 96 * 192
    kv_a_proj_with_mqa  -> linear_kv_down_proj     7168 -> 512 + 64
    kv_a_layernorm      -> kv_layernorm            RMSNorm(512)
    kv_b_proj           -> linear_kv_up_proj       512 -> 96 * (128 + 128)
    o_proj              -> linear_proj             96 * 128 -> 7168
    g_proj              -> linear_o_gate           7168 -> 96 * 128   (new)

Delta 1 — NoPE
--------------
``KimiMLAAttention`` asserts ``use_nope`` (``:396``), sets
``self.rotary_emb = None`` (``:403``), and splits ``q_rot`` / ``k_rot``
out (``:423-428``) only to concatenate them back **unrotated**
(``:439-440``). The 64 "rope" dims are just 64 extra learned
non-positional dims that keep the checkpoint layout DeepSeek-V3-shaped.

Megatron has no NoPE flag: ``rope_type`` is validated against exactly
``{"rope", "yarn"}`` (``multi_latent_attention.py:151-155``) and the
rotary application is unconditional (``:803``, ``:812``). Rather than
duplicate the parent's ~150-line ``get_query_key_value_tensors`` minus
two calls, this class hands the parent a **zero-width frequency table**:
``apply_rotary_pos_emb`` opens with

.. code-block:: python

    # rope_utils.py:110-113
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

so at ``rot_dim == 0`` the whole tensor lands in ``t_pass`` and the
closing ``torch.cat((t, t_pass), dim=-1)`` (``:126``) returns a
bit-exact copy. ``test_k3_mla_nope.py`` asserts that bit-identity on
both Q and K, so the implicit contract cannot rot silently.

**Do not** reach for ``qk_pos_emb_head_dim = 0`` instead. It also
disables rope — ``RotaryEmbedding(0)`` builds an empty ``inv_freq`` —
but it is a *different architecture*:

* it deletes ``k_rot`` outright, because ``linear_kv_down_proj``'s width
  is ``kv_lora_rank + qk_pos_emb_head_dim`` (``:469``). K3's 64 K dims
  are MQA-shared and come straight off the raw down-projection, *not*
  through the ``kv_lora_rank`` latent or ``kv_a_layernorm``
  (``modeling_kimi_linear.py:427-437``), so no setting of
  ``qk_head_dim`` can reproduce them;
* it narrows ``q_b_proj`` from ``96 * 192`` to ``96 * 128``;
* it changes ``softmax_scale`` from K3's ``192 ** -0.5`` to
  ``128 ** -0.5``, because the scale is derived from
  ``q_head_dim = qk_head_dim + qk_pos_emb_head_dim`` (``:115, 127-128``).

The released geometry is ``qk_head_dim: 128`` **and**
``qk_pos_emb_head_dim: 64``; :class:`KimiK3MLASelfAttention` logs a loud
warning if it is constructed with a zero-width positional head.

Delta 2 — sigmoid output gate
-----------------------------
``attn_out = attn_out * sigmoid(g_proj(hidden_states))`` applied
elementwise over ``num_heads * v_head_dim`` **before** ``o_proj``
(``modeling_kimi_linear.py:398-401, 470-473``).

Upstream already implements this feature for vanilla attention as
``config.attention_output_gate`` (``transformer_config.py:242``,
``attention.py:997, 1008, 1210-1236``) but **refuses it under MLA**::

    # transformer_config.py:2259-2260
    if self.attention_output_gate:
        raise NotImplementedError("Output gate is not supported for MLA yet.")

That guard lives in ``MLATransformerConfig.__post_init__``, i.e. it
fires while the *config* is being constructed, long before any attention
module exists — so no amount of ``__init__`` bypassing in an attention
subclass can dodge it. It is dodged at the config layer instead:
:class:`KimiK3TransformerConfig` carries its own ``mla_use_output_gate``
and leaves upstream's ``attention_output_gate`` at ``False``
(``kimi_k3_transformer_config.py:171-174``), and this class builds and
applies the gate itself. That is option (a) of ``DESIGN.md`` §10.2.1,
and it keeps the whole diff inside Primus rather than patching out an
upstream ``NotImplementedError`` that exists precisely because the MLA
gate path is untested.

Core attention must come from TransformerEngine
-----------------------------------------------
MLA's K and V head dims differ (192 vs 128), so
``MultiLatentAttention.__init__`` always passes ``k_channels`` /
``v_channels`` to the core-attention builder
(``multi_latent_attention.py:164-165``). Only
``TEDotProductAttention`` accepts them; the pure-PyTorch
``DotProductAttention`` does not, which makes
``LocalSpecProvider().core_attention()`` unusable here (and makes
upstream's own local MLA spec latently broken).
:func:`_check_core_attention_supports_mla` turns that into an
actionable error at spec-build time.

Tensor parallelism
------------------
``linear_o_gate`` is column-parallel over heads with
``gather_output=False``, so its local width
``num_heads_local * v_head_dim`` matches ``core_attn_out``'s and the
gate multiply stays entirely local. Only ``tp_size == 1`` is exercised
by this work package's unit tests.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import (
    deprecate_inference_params,
    nvtx_range_pop,
    nvtx_range_push,
)
from torch import Tensor

__all__ = [
    "apply_sigmoid_output_gate",
    "KimiK3MLASelfAttentionSubmodules",
    "KimiK3MLASelfAttention",
    "get_kimi_k3_mla_attention_submodules",
    "get_kimi_k3_mla_attention_spec",
]

logger = logging.getLogger(__name__)

# The zero-width-positional-head warning is per-process, not per-layer: 24
# identical lines per rank would bury it.
_warned_about_zero_width_pos_emb = False


def apply_sigmoid_output_gate(x: Tensor, gate: Tensor) -> Tensor:
    """``x * sigmoid(gate)``, with the sigmoid and the multiply in fp32.

    Mirrors upstream's ``Attention._apply_output_gate``
    (``attention.py:1230-1236``) exactly, minus its ``@jit_fuser``, which
    resolves to ``torch.compile`` (``jit.py:16-24``) and is not worth a
    compile of two elementwise ops.

    The HF reference computes ``g_proj(x).sigmoid()`` in the model dtype
    (``modeling_kimi_linear.py:471``); accumulating in fp32 instead is
    strictly more accurate and agrees to bf16 tolerance. In fp32 the two
    are bit-identical.
    """
    out_dtype = x.dtype
    gate = gate.contiguous().view(*x.shape)
    return (x * torch.sigmoid(gate.float())).to(out_dtype)


@dataclass
class KimiK3MLASelfAttentionSubmodules(MLASelfAttentionSubmodules):
    """Upstream's MLA submodules plus Kimi K3's sigmoid output gate.

    ``linear_o_gate`` is HF's ``g_proj`` (``modeling_kimi_linear.py:401``):
    ``hidden_size -> num_attention_heads * v_head_dim``, column-parallel,
    no bias. Required when ``config.mla_use_output_gate`` is set and
    ignored otherwise.
    """

    linear_o_gate: Optional[Union[ModuleSpec, type]] = None


class KimiK3MLASelfAttention(MLASelfAttention):
    """MLA self-attention with no positional encoding and a sigmoid output gate.

    Takes ``[s, b, h]`` and returns ``(output, bias)``, the standard
    Megatron self-attention contract. See the module docstring for the
    two deltas versus upstream MLA and for why neither the parent
    ``__init__`` nor ``get_query_key_value_tensors`` is re-implemented.

    Args:
        config: an :class:`MLATransformerConfig` (in production a
            :class:`KimiK3TransformerConfig`). ``mla_use_output_gate`` is
            read with a ``getattr`` default so the class also works
            against a plain ``MLATransformerConfig``, matching how
            :class:`KimiDeltaAttention` reads its own fields.
        submodules: :class:`KimiK3MLASelfAttentionSubmodules`.
        layer_number: 1-based index of this layer in the block.
        attn_mask_type: defaults to ``causal``; K3 has no sliding window
            and no attention sink.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: KimiK3MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        # Fail loudly rather than half-applying two gates. The config layer is
        # where upstream's flag is refused for MLA (transformer_config.py:2259-2260),
        # so seeing it True here would mean the guard moved.
        assert not config.attention_output_gate, (
            "Kimi K3 applies its own sigmoid output gate and requires upstream's "
            "attention_output_gate to stay False; MLATransformerConfig.__post_init__ "
            "raises NotImplementedError for it anyway (transformer_config.py:2259-2260)."
        )
        if config.rope_type != "rope":
            # The parent builds a YarnRotaryEmbedding for "yarn", whose forward
            # returns (emb, mscale) rather than a bare tensor
            # (multi_latent_attention.py:550-564). K3 rotates nothing, so the
            # cheap and unambiguous choice is to require the plain type.
            raise ValueError(
                f"KimiK3MLASelfAttention requires rope_type='rope', got {config.rope_type!r}. "
                "K3 never applies a rotation; 'rope' is the type whose zero-width "
                "frequency table degenerates cleanly."
            )
        if config.apply_rope_fusion:
            raise ValueError(
                "KimiK3MLASelfAttention requires apply_rope_fusion=False: the fused "
                "fused_apply_mla_rope_for_q/kv path (multi_latent_attention.py:747-771) "
                "takes cos/sin tables rather than a frequency table, so the zero-width "
                "NoPE trick does not reach it."
            )

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        # ---- NoPE -------------------------------------------------------
        # config.mla_use_nope selects the mechanism: replace the parent's
        # qk_pos_emb_head_dim-wide rotary table with a zero-width one.
        # apply_rotary_pos_emb then splits off nothing and returns
        # cat([<empty>, t]) == t, bit for bit (rope_utils.py:110-126), which
        # leaves the parent's get_query_key_value_tensors untouched — and, just
        # as importantly, leaves the *geometry* untouched: q_head_dim stays
        # qk_head_dim + qk_pos_emb_head_dim, so linear_kv_down_proj keeps
        # emitting K3's 64 MQA-shared K dims. Clearing the flag restores the
        # parent's real table, which is upstream MLA RoPE and not Kimi K3.
        self.mla_use_nope = bool(getattr(config, "mla_use_nope", True))
        if self.mla_use_nope:
            self.rotary_pos_emb = RotaryEmbedding(
                0,
                rotary_percent=1.0,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
            assert self.rotary_pos_emb.inv_freq.numel() == 0, (
                "NoPE requires a zero-width rotary table; got "
                f"{self.rotary_pos_emb.inv_freq.numel()} frequencies."
            )

        # softmax_scale is mscale**2 / sqrt(q_head_dim) (:127-128) and K3 wants
        # a plain q_head_dim ** -0.5 == 192 ** -0.5. mscale collapses to 1.0
        # only while rotary_scaling_factor <= 1 or mscale_all_dim == 0
        # (yarn_rotary_pos_embedding.py:252-255), and nothing else in the stack
        # would notice the difference.
        expected_scale = self.q_head_dim**-0.5
        if not math.isclose(self.softmax_scale, expected_scale, rel_tol=1e-9):
            raise ValueError(
                f"Kimi K3 uses softmax_scale = q_head_dim ** -0.5 = {expected_scale} "
                f"(modeling_kimi_linear.py:359), but this config yields "
                f"{self.softmax_scale}. The yarn mscale factor is non-unit; set "
                "mscale_all_dim=0.0 (or rotary_scaling_factor<=1) to restore it."
            )

        self._warn_if_positional_head_is_zero_width()
        self._apply_latent_layernorm_epsilon()

        # ---- the sigmoid output gate ------------------------------------
        self.use_output_gate = bool(getattr(config, "mla_use_output_gate", False))
        self.linear_o_gate = None
        if self.use_output_gate:
            gate_spec = getattr(submodules, "linear_o_gate", None)
            if gate_spec is None:
                raise ValueError(
                    "config.mla_use_output_gate is set but submodules.linear_o_gate is None. "
                    "Build the attention spec with get_kimi_k3_mla_attention_spec, or pass a "
                    "KimiK3MLASelfAttentionSubmodules with linear_o_gate filled in."
                )
            self.linear_o_gate = build_module(
                gate_spec,
                self.config.hidden_size,
                # v_head_dim * num_attention_heads, i.e. the same global width
                # as linear_proj's input (multi_latent_attention.py:113).
                self.query_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="o_gate",
                tp_group=self.pg_collection.tp,
            )

    def _apply_latent_layernorm_epsilon(self) -> None:
        """Retune ``q_layernorm`` / ``kv_layernorm`` to the released epsilon.

        The parent builds both with ``eps=config.layernorm_epsilon``
        (``multi_latent_attention.py:498-508``), i.e. Kimi K3's
        ``rms_norm_eps`` of 1e-5. The released model does not: ``KimiRMSNorm``
        defaults to ``eps=1e-6`` (``modeling_kimi_linear.py:226-227``) and MLA
        constructs exactly these two norms without passing one (``:368``,
        ``:383``), while every other ``KimiRMSNorm`` call site in the file is
        given ``eps=config.rms_norm_eps``. Kimi-Linear-48B's own
        ``modeling_kimi.py:365`` does the same, so this is not a typo in one
        file -- it is inherited from the DeepSeek-V3 MLA this was adapted from,
        where ``rms_norm_eps`` is itself 1e-6 and the two coincide.

        Found by comparing against the published checkpoint: with our 1e-5 the
        MLA output differs from the reference by ``rel_rms`` 1.95e-03 on real
        Kimi-Linear-48B layer-3 weights, and the disagreement is confined to
        the K/V path -- ``query`` (no norm, Kimi-Linear has ``q_lora_rank:
        null``) and the core attention on identical q/k/v are both
        bit-identical. At 1e-6 the whole module agrees to 4.69e-07. See
        ``hf_validate/FINDINGS.md`` §5.2.

        Set ``mla_latent_layernorm_epsilon=None`` to keep upstream's behaviour.
        """
        eps = getattr(self.config, "mla_latent_layernorm_epsilon", None)
        if eps is None:
            return
        eps = float(eps)
        for norm in (getattr(self, "q_layernorm", None), getattr(self, "kv_layernorm", None)):
            if norm is None or isinstance(norm, IdentityOp):
                continue
            # TE's RMSNorm keeps it in ``eps``; torch/apex flavours in
            # ``layer_norm_eps`` or ``variance_epsilon``. There is no common
            # accessor, so set whichever exists and fail if none did rather
            # than silently leaving the norm at the parent's value.
            applied = False
            for attr in ("eps", "layer_norm_eps", "variance_epsilon"):
                if hasattr(norm, attr):
                    setattr(norm, attr, eps)
                    applied = True
            if not applied:
                raise RuntimeError(
                    f"{type(norm).__name__} exposes no epsilon attribute, so "
                    "mla_latent_layernorm_epsilon cannot be honoured. Set it to None to "
                    "accept config.layernorm_epsilon on the MLA latent norms."
                )

    def _warn_if_positional_head_is_zero_width(self) -> None:
        """Warn once that ``qk_pos_emb_head_dim = 0`` is not the released geometry."""
        global _warned_about_zero_width_pos_emb
        if self.config.qk_pos_emb_head_dim != 0 or _warned_about_zero_width_pos_emb:
            return
        _warned_about_zero_width_pos_emb = True
        logger.warning(
            "[Primus:Kimi-K3] qk_pos_emb_head_dim == 0. NoPE does not need it: this class "
            "never applies a rotation. Zeroing it instead *removes* the 64 MQA-shared, "
            "unrotated K dims the release ships (qk_rope_head_dim=64, produced directly by "
            "kv_a_proj_with_mqa outside the kv_lora_rank latent), narrows q_b_proj from "
            "num_heads*192 to num_heads*128, and changes softmax_scale from 192**-0.5 to "
            "%s. Set qk_head_dim=128 and qk_pos_emb_head_dim=64 for the released geometry.",
            self.q_head_dim**-0.5,
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
        """Forward pass. ``hidden_states`` is ``[s, b, h]``.

        Without the output gate this delegates to
        :meth:`MultiLatentAttention.forward` verbatim, so the un-gated
        path can never drift from upstream. With it, the training path of
        ``multi_latent_attention.py:203-361`` is reproduced with the gate
        spliced in between ``core_attention`` and ``linear_proj``, which
        is where HF applies it (``modeling_kimi_linear.py:470-473``).
        """
        if self.linear_o_gate is None:
            return super().forward(
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                position_ids=position_ids,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        # The parent's preconditions, verbatim (multi_latent_attention.py:221-226).
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert rotary_pos_cos is None and rotary_pos_sin is None, "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context is not None or self.config.cache_mla_latents:
            raise NotImplementedError(
                "The Kimi K3 output gate is a training-path feature: KV caching, latent "
                "absorption and dynamic batching are not wired through it yet."
            )
        if self.config.experimental_attention_variant == "dsa":
            raise NotImplementedError(
                "experimental_attention_variant='dsa' needs the hidden states and the "
                "compressed query threaded into core_attention "
                "(multi_latent_attention.py:288-292); Kimi K3 does not use DSA."
            )
        for flag in ("offload_qkv_linear", "offload_core_attention", "offload_attn_proj"):
            if getattr(self, flag):
                raise NotImplementedError(
                    "fine_grained_activation_offloading is not supported by the Kimi K3 "
                    f"gated attention path ({flag} is set); the offload group boundaries "
                    "would have to account for the extra gate projection."
                )

        # =====================
        # Query, Key, and Value
        # =====================
        # NoPE happens inside here: self.rotary_pos_emb has zero width, so the
        # parent's two apply_rotary_pos_emb calls (:803, :812) are bit-exact
        # pass-throughs. Nothing else in the parent's implementation changes.
        nvtx_range_push(suffix="k3_mla_qkv")
        query, key, value, _q_compressed, _kv_compressed = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=None,
        )
        nvtx_range_pop(suffix="k3_mla_qkv")

        # TE only accepts contiguous tensors for MLA (:269-275). With no
        # inference context _adjust_key_value_for_inference is a pass-through
        # that only reports self.attn_mask_type back (attention.py:496-498),
        # so it is elided here.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        nvtx_range_push(suffix="k3_mla_core_attn")
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=self.attn_mask_type,
            )
        nvtx_range_pop(suffix="k3_mla_core_attn")

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            # (t, np, hn) -> (t, b=1, h=np*hn); batch is a dummy axis when packed.
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =====================================================
        # Sigmoid output gate, before o_proj (:470-473)
        # =====================================================
        nvtx_range_push(suffix="k3_mla_output_gate")
        gate, _ = self.linear_o_gate(hidden_states)
        core_attn_out = apply_sigmoid_output_gate(core_attn_out, gate)
        nvtx_range_pop(suffix="k3_mla_output_gate")

        # =================
        # Output. [sq, b, h]
        # =================
        nvtx_range_push(suffix="k3_mla_o_proj")
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix="k3_mla_o_proj")

        return output, bias

    def backward_dw(self) -> None:
        """Weight-gradient computation, extended with the gate projection."""
        super().backward_dw()
        if self.linear_o_gate is not None and hasattr(self.linear_o_gate, "backward_dw"):
            self.linear_o_gate.backward_dw()


def _check_core_attention_supports_mla(core_attention: type) -> None:
    """Reject a core attention that cannot take MLA's asymmetric head dims.

    ``MultiLatentAttention.__init__`` always passes
    ``k_channels=q_head_dim`` and ``v_channels=v_head_dim``
    (``multi_latent_attention.py:164-165``) because MLA's K and V head
    dims differ (192 vs 128). At this Megatron HEAD only
    ``TEDotProductAttention`` accepts those kwargs
    (``extensions/transformer_engine.py:1178``); the pure-PyTorch
    ``DotProductAttention`` does not, so
    ``LocalSpecProvider().core_attention()`` yields a spec that dies with
    an opaque ``TypeError`` several frames inside ``build_module``.
    Upstream's own ``get_gpt_layer_local_spec(multi_latent_attention=True)``
    (``gpt_layer_specs.py:366-381``) carries the same latent breakage.
    Fail here with something actionable instead.
    """
    try:
        params = inspect.signature(core_attention.__init__).parameters
    except (TypeError, ValueError):  # builtins / C extensions
        return
    if "k_channels" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    raise ValueError(
        f"{core_attention.__name__} cannot back a Kimi K3 full-attention layer: MLA passes "
        "k_channels / v_channels because its K and V head dims differ "
        "(multi_latent_attention.py:164-165), and only TEDotProductAttention accepts them at "
        "this Megatron HEAD. Use the TransformerEngine backend, or supply a core_attention that "
        "takes k_channels / v_channels."
    )


def get_kimi_k3_mla_attention_submodules(
    backend,
    *,
    rms_norm: bool = True,
    mla_use_output_gate: bool = True,
) -> KimiK3MLASelfAttentionSubmodules:
    """Submodule specs for :class:`KimiK3MLASelfAttention`.

    Follows the upstream MLA submodule tables verbatim — the TE flavour
    at ``gpt_layer_specs.py:246-262`` and the local flavour at
    ``gpt_layer_specs.py:366-381`` — and adds ``linear_o_gate``. K3
    always wants the ``q_layernorm`` / ``kv_layernorm`` slots filled
    (HF's ``q_a_layernorm`` / ``kv_a_layernorm``), so unlike upstream
    they are not gated on ``config.qk_layernorm``: ``kimi_k3_base.yaml``
    deliberately leaves that flag off because these norms come from the
    spec rather than from the generic MCore switch.

    Args:
        backend: an upstream ``BackendSpecProvider``. Note that
            ``LocalSpecProvider``'s ``DotProductAttention`` is not
            MLA-capable — see
            :func:`_check_core_attention_supports_mla`.
        rms_norm: K3 uses RMSNorm everywhere (``normalization: RMSNorm``).
        mla_use_output_gate: emit the ``linear_o_gate`` slot.
    """
    core_attention = backend.core_attention()
    _check_core_attention_supports_mla(core_attention)
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True)
    # The TE providers expose a replicated `linear()` for the low-rank
    # down-projections; LocalSpecProvider does not and upstream's local MLA
    # spec uses column-parallel there instead. MLASelfAttention.__init__
    # validates whichever class arrives (:410-419, :455-464).
    linear_factory = getattr(backend, "linear", None)
    down_proj = linear_factory() if linear_factory is not None else backend.column_parallel_linear()

    submodules = KimiK3MLASelfAttentionSubmodules(
        q_layernorm=qk_norm,
        kv_layernorm=qk_norm,
        linear_q_proj=backend.column_parallel_linear(),
        linear_q_down_proj=down_proj,
        linear_q_up_proj=backend.column_parallel_linear(),
        linear_kv_down_proj=down_proj,
        linear_kv_up_proj=backend.column_parallel_linear(),
        core_attention=core_attention,
        linear_proj=backend.row_parallel_linear(),
    )
    if mla_use_output_gate:
        submodules.linear_o_gate = backend.column_parallel_linear()
    return submodules


def get_kimi_k3_mla_attention_spec(
    config=None,
    *,
    backend=None,
    use_transformer_engine: bool = True,
    attn_mask_type: AttnMaskType = AttnMaskType.causal,
) -> ModuleSpec:
    """``ModuleSpec`` for one Kimi K3 full-attention layer.

    The layer / block assembly that consumes this — and the KDA spec it
    is interleaved with — belongs to the layer-spec work package; this
    only covers the attention module itself.

    Args:
        config: when given, ``normalization`` and ``mla_use_output_gate``
            are read from it. Defaults are K3's (RMSNorm, gate on).
        backend: an upstream ``BackendSpecProvider``. Defaults to
            ``TESpecProvider`` or ``LocalSpecProvider`` per
            ``use_transformer_engine``.
        use_transformer_engine: ignored when ``backend`` is passed.
        attn_mask_type: K3 is plain causal.
    """
    if backend is None:
        if use_transformer_engine:
            from megatron.core.extensions.transformer_engine_spec_provider import (
                TESpecProvider,
            )

            backend = TESpecProvider()
        else:
            from megatron.core.models.backends import LocalSpecProvider

            backend = LocalSpecProvider()

    rms_norm = True if config is None else getattr(config, "normalization", "RMSNorm") == "RMSNorm"
    mla_use_output_gate = True if config is None else bool(getattr(config, "mla_use_output_gate", False))

    return ModuleSpec(
        module=KimiK3MLASelfAttention,
        params={"attn_mask_type": attn_mask_type},
        submodules=get_kimi_k3_mla_attention_submodules(
            backend, rms_norm=rms_norm, mla_use_output_gate=mla_use_output_gate
        ),
    )
