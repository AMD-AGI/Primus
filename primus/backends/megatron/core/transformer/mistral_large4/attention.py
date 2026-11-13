###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 attention: Megatron MLA (no Q-LoRA) with the model's per-layer variants.

``MistralLarge4Attention`` subclasses :class:`MLASelfAttention`, so its projections are the
backend's TE column/row-parallel linears and its core attention is Megatron's (TE) core
attention (selective recompute, CP and TP come from there). On top of MLA it adds:

* NoPE layers: built from ``config.layer_config(i)`` (``qk_pos_emb_head_dim = 0``,
  ``qk_head_dim = 128``), so ``linear_kv_down_proj`` is ``[512, h]`` and no rope is applied;
* cross-layer KV sharing: a consumer layer has no ``linear_kv_down_proj`` / ``kv_layernorm``
  and reads the source layer's normalized latent from the block's per-forward state;
* SSMax on the query (after rope, before ``QK^T``), per head and TP-sharded with the heads;
* the DSA lightning indexer (``MistralLarge4Indexer``) and top-k sparse attention through
  ``sparse_topk_attention`` (the reference's eager top-k mask). With ``T <= index_topk`` every causal key is selected,
  so the sparse mask is the causal mask and the dense core attention is used (exactly equal on the
  TE core attention: both apply ``config.attention_dropout`` in training; ``EagerCoreAttention``,
  the local-backend stand-in, applies no attention dropout).

The rope table (YaRN + LeRoPE) is shared by every layer and owned by the decoder block.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    fused_qk_topk_naive,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from torch import nn

from primus.backends.megatron.core.transformer.mistral_large4.rotary import ssmax_scale

__all__ = ["EagerCoreAttention", "MistralLarge4Attention", "MistralLarge4Indexer"]


def _causal_mask(sq: int, sk: int, device) -> torch.Tensor:
    return torch.full((sq, sk), float("-inf"), device=device, dtype=torch.float32).triu_(1)


def topk_keep_mask(topk: torch.Tensor, sk: int, causal_keep: torch.Tensor) -> torch.Tensor:
    """``[b, sq, sk]`` bool: each query's top-k keys (``-1`` = none), restricted to the causal past."""
    b, sq, _ = topk.shape
    idx = topk.long()
    keep = torch.zeros(b, sq, sk + 1, dtype=torch.bool, device=topk.device)
    keep.scatter_(-1, torch.where(idx < 0, sk, idx), True)  # -1 goes to a dropped extra column
    return keep[..., :sk] & causal_keep


def sparse_topk_attention(query, key, value, keep, softmax_scale, dropout_p: float = 0.0):
    """Causal attention restricted to each query's top-k keys (eager reference math).

    As the transformers reference does in its eager / SDPA paths: keys outside the top-k are
    masked before the softmax (``index_mask``), then plain attention. ``query``/``key``
    ``[s, b, n, d]``, ``value`` ``[s, b, n, dv]``, ``keep`` ``[b, s, s]`` bool
    (``topk_keep_mask``); returns ``[s, b, n * dv]``. Memory O(s^2) per head. ``dropout_p`` is the
    attention-probability dropout (``config.attention_dropout`` in training, as the dense TE path).
    """
    sq, b, n, _ = query.shape
    assert keep.dtype == torch.bool, "sparse_topk_attention takes the bool keep mask (topk_keep_mask)"
    q, k, v = (t.permute(1, 2, 0, 3) for t in (query, key, value))  # [b, n, s, d]
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=keep[:, None], scale=softmax_scale, dropout_p=dropout_p
    )
    return out.permute(2, 0, 1, 3).reshape(sq, b, n * value.size(-1))


def _rope(x: torch.Tensor, freqs: torch.Tensor, config) -> torch.Tensor:
    """Interleaved (DeepSeek-layout) rope on the last dim of ``x`` [s, b, h, d]; output [evens|odds]."""
    return apply_rotary_pos_emb(x, freqs, config=config, mla_rotary_interleaved=True)


class MistralLarge4Indexer(MegatronModule):
    """DSA lightning indexer of Mistral Large 4 (eager reference math).

    ``q = wq_b(LayerNorm(x))`` (no Q-LoRA), SSMax on its ``index_n_heads`` heads,
    ``k = LayerNorm(wk(x))``, interleaved rope on the first ``qk_pos_emb_head_dim`` dims of q and k
    on rope layers, no Hadamard rotation,
    ``score = sum_h relu(q_h . k) * index_head_dim^-0.5 * w_h * index_n_heads^-0.5`` with
    ``w = weights_proj(x)`` in fp32, causal mask, top-k (Megatron's ``fused_qk_topk_naive``).
    The release has ``index_n_heads`` 32, ``qk_pos_emb_head_dim`` 64, ``index_head_dim`` 128.

    The reference runs the indexer under ``no_grad`` and trains no indexer loss, so its weights
    are frozen: they are persistent buffers (in the checkpoint, outside the optimizer, weight
    decay and FSDP sharding), replicated on every TP rank. ``weights_proj`` and SSMax stay fp32.
    """

    FP32 = ("weights_proj", "ssmax_s", "ssmax_b")

    def __init__(self, config, submodules=None, use_rope: bool = True):
        super().__init__(config=config)
        h = int(config.hidden_size)
        self.n_heads = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.topk = int(config.index_topk)
        self.rope_dim = int(config.qk_pos_emb_head_dim) if use_rope else 0
        self.use_rope = bool(use_rope)
        # n_heads**-0.5 (weights) x head_dim**-0.5 (score scale), applied once to the head weights
        self.weight_scale = (self.n_heads * self.head_dim) ** -0.5
        self.eps = float(config.index_norm_eps)
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        dt = config.params_dtype
        shapes = {
            "q_norm_weight": (h,),
            "q_norm_bias": (h,),
            "wq_b": (self.n_heads * self.head_dim, h),
            "wk": (self.head_dim, h),
            "k_norm_weight": (self.head_dim,),
            "k_norm_bias": (self.head_dim,),
            "weights_proj": (self.n_heads, h),
            "ssmax_s": (self.n_heads,),
            "ssmax_b": (self.n_heads,),
        }
        for name, shape in shapes.items():
            t = torch.empty(shape, device=device, dtype=torch.float32 if name in self.FP32 else dt)
            if name.endswith("weight") or name == "ssmax_b":
                t.fill_(1.0)
            elif name.endswith("bias") or name == "ssmax_s":
                t.zero_()
            else:
                (
                    config.init_method(t)
                    if config.perform_initialization
                    else t.normal_(0.0, config.init_method_std)
                )
            self.register_buffer(name, t)

    def _apply(
        self, fn, *args, **kwargs
    ):  # weights_proj and SSMax stay fp32, unrounded (reference keep_in_fp32)
        before = {name: getattr(self, name) for name in self.FP32}
        out = super()._apply(fn, *args, **kwargs)
        for name in self.FP32:
            t = getattr(self, name)
            if t.dtype != torch.float32:
                setattr(self, name, before[name].to(device=t.device, dtype=torch.float32))
        return out

    def at_init(self) -> bool:
        """Whether the weights are still this module's own initial values (norm weights 1, SSMax
        slope 0): the frozen indexer was never loaded from a checkpoint. A trained indexer (the
        release's) has learned norm weights and slopes."""
        return bool(
            (self.q_norm_weight == 1).all() and (self.k_norm_weight == 1).all() and (self.ssmax_s == 0).all()
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        freqs: Optional[torch.Tensor],
        causal: Optional[torch.Tensor] = None,
        log_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x [s, b, h] (full sequence), positions [s], ``causal`` the [s, s] additive causal mask
        (built here when not given); returns top-k indices [b, s, k] (-1 = none)."""
        s, b, h = x.shape
        q = F.layer_norm(x, (h,), self.q_norm_weight.to(x.dtype), self.q_norm_bias.to(x.dtype), self.eps)
        q = F.linear(q, self.wq_b.to(x.dtype)).reshape(s, b, self.n_heads, self.head_dim)
        q = ssmax_scale(q, positions, self.ssmax_s, self.ssmax_b, log_pos)
        k = F.linear(x, self.wk.to(x.dtype))
        k = F.layer_norm(
            k, (self.head_dim,), self.k_norm_weight.to(k.dtype), self.k_norm_bias.to(k.dtype), self.eps
        )
        k = k.reshape(s, b, 1, self.head_dim)
        if self.use_rope:
            r = self.rope_dim
            q = torch.cat([_rope(q[..., :r], freqs, self.config), q[..., r:]], dim=-1)
            k = torch.cat([_rope(k[..., :r], freqs, self.config), k[..., r:]], dim=-1)
        k = k.reshape(s, b, self.head_dim)
        weights = F.linear(x.float(), self.weights_proj) * self.weight_scale
        _, topk = fused_qk_topk_naive(
            q.float(),
            k.float(),
            weights,
            self.topk,
            mask=causal if causal is not None else _causal_mask(s, s, x.device),
            use_relu=True,
        )
        return topk


class MistralLarge4Attention(MLASelfAttention):
    """MLA self-attention of one Mistral Large 4 layer (see the module docstring).

    The decoder block hands each layer a per-forward context through
    ``set_forward_context(ctx, latent)``: ``ctx`` holds the shared rope angles, the
    query positions and the top-k indices of the most recent full-indexer layer (shared layers
    reuse only the nearest one); ``latent`` is the
    normalized KV latent of this layer's KV source when the layer consumes one. After the
    forward, ``latent_out`` holds this layer's latent when later layers reuse it.
    """

    def __init__(
        self,
        config,
        submodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        pg_collection=None,
        pp_layer_offset: Optional[int] = None,
        name: Optional[str] = None,
        kv_layernorm=None,
        indexer=None,
    ):
        idx = int(layer_number) - 1  # TransformerLayer passes the global 1-based number
        self.layer_idx = idx
        self.use_rope = config.uses_rope(idx)
        self.kv_source_idx = config.kv_source(idx)
        self.consumes_kv = self.kv_source_idx != idx
        self.stores_kv = config.stores_kv(idx)
        self.index_source_idx = config.index_source_layer(idx)
        self.has_indexer = self.index_source_idx == idx
        if self.consumes_kv and self.use_rope:
            raise NotImplementedError(
                f"Mistral Large 4 layer {idx}: KV sharing between rope layers (shared k_rot) is not implemented; "
                "the released checkpoint shares only NoPE latents"
            )
        layer_cfg = config.layer_config(idx)
        super().__init__(
            config=layer_cfg,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )
        # the rope table is the decoder block's (YaRN + LeRoPE, shared by every layer)
        del self.rotary_pos_emb
        self.qk_nope_dim = int(layer_cfg.qk_head_dim)
        self.qk_rope_dim = int(layer_cfg.qk_pos_emb_head_dim)
        self.kv_lora_rank = int(layer_cfg.kv_lora_rank)
        if self.consumes_kv:
            # the latent (and its norm) come from the source layer
            del self.linear_kv_down_proj
            self.kv_layernorm = None
        else:
            self.kv_layernorm = build_module(
                kv_layernorm, config=layer_cfg, hidden_size=self.kv_lora_rank, eps=layer_cfg.layernorm_epsilon
            )
        # SSMax per head, sharded with the heads over TP
        nh = self.num_attention_heads_per_partition
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        # trainable and in params_dtype like the other attention weights (unlike the indexer's
        # SSMax, which is frozen and kept fp32 as the reference keeps it); the converter rounds
        # the release values to params_dtype here
        self.ssmax_s = nn.Parameter(torch.zeros(nh, device=device, dtype=config.params_dtype))
        self.ssmax_b = nn.Parameter(torch.ones(nh, device=device, dtype=config.params_dtype))
        for p in (self.ssmax_s, self.ssmax_b):
            set_tensor_model_parallel_attributes(p, True, 0, 1)
        self.indexer = (
            build_module(indexer, config=config, use_rope=self.use_rope) if self.has_indexer else None
        )
        self._ctx = None
        self._latent_in = None
        self.latent_out = None

    @staticmethod
    def _local_ctx(ctx, device):
        """The block's per-forward state with the shared tensors (rope angles, positions) on this
        layer's device: a model split over GPUs by layer runs one block on several devices."""
        assert ctx is not None, "MistralLarge4Attention needs the decoder block's forward context"
        moved = {}
        for k in ("positions", "log_pos", "freqs"):
            t = ctx.get(k)
            if torch.is_tensor(t) and t.device != device:
                moved[k] = t.to(device)
        if not moved:
            return ctx
        local = dict(ctx)
        local.update(moved)
        return local

    def set_forward_context(self, ctx, latent=None):
        self._ctx = ctx
        self._latent_in = latent

    def _sp_gather(self, t):
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            return gather_from_sequence_parallel_region(t, group=self.pg_collection.tp)
        return t

    def _qkv(self, hidden_states, ctx):
        """query, key, value [s, b, n, d] of this layer (MLA up-projection, rope, SSMax)."""
        q, _ = self.linear_q_proj(hidden_states)
        if self.consumes_kv:
            if self._latent_in is None:
                raise RuntimeError(
                    f"Mistral Large 4 layer {self.layer_idx} reuses layer {self.kv_source_idx}'s latent KV, "
                    "but the decoder block passed none"
                )
            kv_c, k_pe = self._latent_in.to(hidden_states.device), None
        else:
            kv, _ = self.linear_kv_down_proj(hidden_states)
            kv_c, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_dim], dim=-1)
            kv_c = self.kv_layernorm(kv_c)
        self.latent_out = kv_c if self.stores_kv else None
        if k_pe is not None and self.qk_rope_dim:
            k_pe = self._sp_gather(k_pe)
        else:
            k_pe = None

        def up_proj_and_rope(q, kv_c, *k_pe_opt):
            s, b = q.shape[:2]
            nh = self.num_attention_heads_per_partition
            q = q.view(s, b, nh, self.qk_nope_dim + self.qk_rope_dim)
            kv, _ = self.linear_kv_up_proj(kv_c)
            kv = kv.view(s, b, nh, self.qk_nope_dim + self.config.v_head_dim)
            k_nope, value = torch.split(kv, [self.qk_nope_dim, self.config.v_head_dim], dim=-1)
            if self.qk_rope_dim:
                freqs = ctx["freqs"]
                k_pe = k_pe_opt[0].unsqueeze(2)  # [s, b, 1, r]
                q_nope, q_pe = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
                query = torch.cat([q_nope, _rope(q_pe, freqs, self.config)], dim=-1)
                k_pe = _rope(k_pe, freqs, self.config)
                key = torch.cat([k_nope, k_pe.expand(-1, -1, nh, -1)], dim=-1)
            else:
                query, key = q, k_nope
            query = ssmax_scale(query, ctx["positions"], self.ssmax_s, self.ssmax_b, ctx.get("log_pos"))
            return query.contiguous(), key.contiguous(), value.contiguous()

        args = (q, kv_c) + ((k_pe,) if k_pe is not None else ())
        if self.recompute_up_proj and self.training:
            # selective recompute of the up-projections (recompute_modules: mla_up_proj), as MLA does
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                fp8=bool(self.config.fp8 or self.config.fp4)
            )
            return self.qkv_up_checkpoint.checkpoint(up_proj_and_rope, *args)
        return up_proj_and_rope(*args)

    @staticmethod
    def _cached(ctx, key, build):
        """A per-forward tensor shared by every layer (built once by the first that needs it)."""
        if key not in ctx:
            ctx[key] = build()
        return ctx[key]

    def _build_keep(self, ctx, topk, source_idx):
        """The ``[b, s, s]`` bool keep mask of ``topk``, stored as the stage's one live mask."""
        s = topk.size(1)
        causal_keep = self._cached(
            ctx, "causal_keep", lambda: torch.ones(s, s, dtype=torch.bool, device=topk.device).tril_()
        ).to(topk.device)
        keep = topk_keep_mask(topk, s, causal_keep)
        ctx["keep"].clear()  # one [b, s, s] mask alive per stage, not one per full layer
        ctx["keep"][source_idx] = keep
        return keep

    def _check_indexer_loaded(self):
        """First sparse forward: the frozen indexer must hold loaded weights, not its init values
        (a run from scratch, also one resumed from its own checkpoint, never trains it)."""
        self._indexer_checked = True
        if self.indexer.at_init() and not getattr(self.config, "allow_untrained_indexer", False):
            raise RuntimeError(
                f"Mistral Large 4 layer {self.layer_idx}: sparse DSA attention (sequence > index_topk) with an "
                "indexer that still has its initial weights. The indexer is frozen (as in the reference), so "
                "it only selects keys well when loaded from the converted release. Load it, use "
                "seq_length <= index_topk, or set allow_untrained_indexer: true for code-path tests."
            )

    def _keep(self, hidden_states, ctx):
        """``[b, s, s]`` keep mask of this layer's top-k: a full-indexer layer runs its indexer;
        a ``shared`` layer reuses the mask its source layer built (same top-k, built once)."""
        if self.indexer is not None:
            x = self._sp_gather(hidden_states)
            s = x.size(0)
            causal = self._cached(ctx, "causal_add", lambda: _causal_mask(s, s, x.device)).to(x.device)
            if not getattr(self, "_indexer_checked", False):
                self._check_indexer_loaded()
            topk = self.indexer(x, ctx["positions"], ctx["freqs"], causal, ctx.get("log_pos"))
            ctx["topk"].clear()  # shared layers reuse the nearest full layer only: keep one entry
            ctx["topk"][self.layer_idx] = topk
            return self._build_keep(ctx, topk, self.layer_idx)
        if self.index_source_idx not in ctx["topk"]:
            raise RuntimeError(
                f"Mistral Large 4 layer {self.layer_idx} reuses the DSA top-k of layer {self.index_source_idx}, "
                "which did not run before it in this pipeline stage: start each pipeline stage on a full-indexer "
                "layer (indexer_types)"
            )
        keep = ctx["keep"].get(self.index_source_idx)
        if keep is None:  # recomputed chunk: the top-k was restored, its mask is rebuilt once
            topk = ctx["topk"][self.index_source_idx].to(hidden_states.device)
            keep = self._build_keep(ctx, topk, self.index_source_idx)
        return keep.to(hidden_states.device)

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
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("Mistral Large 4: KV-cache inference is not supported")
        if key_value_states is not None:
            raise NotImplementedError("Mistral Large 4 attention is self-attention only (key_value_states)")
        if any(t is not None for t in (rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin)):
            raise ValueError("Mistral Large 4 applies its own YaRN + LeRoPE rope: no rotary_pos_* inputs")
        if attention_bias is not None:
            raise NotImplementedError("Mistral Large 4 attention takes no attention_bias")
        ctx = self._local_ctx(self._ctx, hidden_states.device)
        query, key, value = self._qkv(hidden_states, ctx)
        if ctx["sparse"]:
            # top-k sparse attention (the indexer binds only past index_topk keys)
            if packed_seq_params is not None:
                raise NotImplementedError("Mistral Large 4 sparse attention with packed sequences")
            keep = self._keep(hidden_states, ctx)
            if self.checkpoint_core_attention and self.training:
                # selective recompute of core attention (recompute_modules core_attn), as the dense path
                # softmax_scale is bound, not passed: Megatron's CheckpointFunction saves its
                # positional args with save_for_backward, which takes tensors only
                core = tensor_parallel.checkpoint(
                    functools.partial(self._sparse_core, softmax_scale=self.softmax_scale),
                    False,
                    query,
                    key,
                    value,
                    keep,
                )
            else:
                core = self._sparse_core(query, key, value, keep, softmax_scale=self.softmax_scale)
        elif self.checkpoint_core_attention and self.training:
            core = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            core = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=None,
                packed_seq_params=packed_seq_params,
            )
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            core = core.reshape(core.size(0), 1, -1)
        if self.recompute_up_proj and self.training:
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core)
            self.qkv_up_checkpoint = None
        return self.linear_proj(core)

    def _sparse_core(self, query, key, value, keep, softmax_scale):
        """``sparse_topk_attention`` with the attention dropout of the dense path
        (``config.attention_dropout`` while training), its random state from the tensor-parallel
        RNG tracker as TE's core attention takes it (recomputed identically under checkpointing)."""
        p = float(self.config.attention_dropout) if self.training else 0.0
        if p == 0.0:
            return sparse_topk_attention(query, key, value, keep, softmax_scale)
        with tensor_parallel.get_cuda_rng_tracker().fork():
            return sparse_topk_attention(query, key, value, keep, softmax_scale, dropout_p=p)

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        """MLA's sharded state dict, with the SSMax ``ssmax_s`` / ``ssmax_b`` split over TP on axis 0.

        They hold this rank's heads only (``num_attention_heads_per_partition`` entries), so they
        must be TP-sharded in torch_dist checkpoints like the head dim of ``linear_q_proj``;
        Megatron's default treats an unknown parameter as TP-replicated."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        tp_group = getattr(self, "tp_group", None) or self.pg_collection.tp
        for name in ("ssmax_s", "ssmax_b"):
            key = f"{prefix}{name}"
            sharded[key] = make_tp_sharded_tensor_for_checkpoint(
                getattr(self, name),
                key,
                0,
                prepend_offsets=sharded_offsets,
                tp_group=tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        return sharded

    def _backward_kv_proj(self):
        """Delayed weight-gradient step: a KV-consumer layer has no linear_kv_down_proj (it reuses the source layer's latent)."""
        self.linear_kv_up_proj.backward_dw()
        if not self.consumes_kv:
            self.linear_kv_down_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        if not self.consumes_kv:
            super().set_for_recompute_input_layernorm()


class EagerCoreAttention(MegatronModule):
    """Causal softmax attention in fp32 (eager reference / local-backend core attention).

    New math: none; it exists because Megatron's local ``DotProductAttention`` (the only
    non-TE core attention) cannot take MLA's different key / value head sizes. Training on
    GPUs uses the TE core attention from the spec provider; this class is the CPU / local
    backend's stand-in, never the TE path's. ``[s, b, n, d]`` in, ``[s, b, n * dv]`` out.
    It applies no attention dropout (the sparse path and TE's core attention do).
    """

    def __init__(
        self,
        config,
        layer_number,
        attn_mask_type,
        attention_type,
        softmax_scale=None,
        k_channels=None,
        v_channels=None,
        cp_comm_type=None,
        pg_collection=None,
        **_unused,
    ):
        super().__init__(config=config)
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else (k_channels or config.kv_channels) ** -0.5
        )

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **_unused,
    ):
        if packed_seq_params is not None:
            raise NotImplementedError("EagerCoreAttention: packed sequences")
        sq, sk = query.size(0), key.size(0)
        scores = torch.einsum("sbnd,tbnd->bnst", query.float(), key.float()) * self.softmax_scale
        scores = scores + _causal_mask(sq, sk, query.device)
        probs = scores.softmax(-1)
        out = torch.einsum("bnst,tbnd->sbnd", probs, value.float()).to(value.dtype)
        return out.reshape(sq, out.size(1), -1)
