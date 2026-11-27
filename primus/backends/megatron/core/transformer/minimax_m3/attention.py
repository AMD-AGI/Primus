###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax Sparse Attention (MSA).

MSA is MiniMax-M3's attention: an index branch scores every
``sparse_block_size``-token KV block, the top ``sparse_topk_blocks`` blocks are
kept per GQA group (plus the init and local blocks), and the main branch then
runs exact attention over that subset only.

This class occupies the ``self_attention`` slot of upstream's
``TransformerLayer`` on the layers where
:attr:`MSATransformerConfig.sparse_layer_pattern` is 1, so M3 keeps running on
upstream ``GPTModel`` / ``TransformerLayer``. The swap is installed by
``primus/backends/megatron/patches/minimax_m3_patches.py``.

``forward`` is overridden rather than delegating to ``SelfAttention.forward``
because the attention core is MSA's own and must not reach Transformer Engine's
``core_attention``. The projections and the per-head q/k norms are still
upstream's -- only the attention core is replaced. (The unused
``core_attention`` module is still built by ``Attention.__init__``; it holds no
parameters, so leaving it costs nothing.)

``msa_backend`` picks the core: ``eager``, the plain-PyTorch reference that
materialises the full score matrix, or ``flydsl``, the gfx950 (MFMA) and
gfx1250 (wave32 WMMA) kernels in ``minimax_m3/flydsl`` that never do.

Mirrors ``MiniMaxM3VLAttention`` in transformers'
``models/minimax_m3_vl/modeling_minimax_m3_vl.py``.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
    MSATransformerConfig,
)
from primus.backends.megatron.core.transformer.minimax_m3.eager import (
    build_block_keep,
    eager_block_sparse_attention,
)
from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
    MinimaxM3Indexer,
    MinimaxM3IndexerSubmodules,
)
from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
    MSAIndexerLossAutoScaler,
    compute_indexer_loss,
    compute_sparse_indexer_loss,
    record_indexer_loss,
    slot_mass_from_probs,
    slot_mass_from_slot_lse,
)
from primus.backends.megatron.core.transformer.minimax_m3.rope import RopeTables
from primus.core.utils.module_utils import log_rank_0

# Backends that compute the block-sparse attention itself.
MSA_BACKENDS = ("eager", "flydsl")

_MSA_FUSIONS = (
    "msa_fused_qk_norm_rope",
    "msa_fused_indexer_loss",
    "msa_fused_indexer_loss_scatter",
    "msa_fused_select_plan",
    "msa_shared_rope_tables",
    "msa_fused_zero_centered_gamma",
    "msa_fused_indexer_input_norm",
)
# Kernel paths of the flydsl backend on gfx1250, taken wherever a shape is eligible (no switch;
# other GPUs never take them). Each is decided before any work, and its first call for a shape
# compares it with the path it replaces (bitwise_check on the call's tensors, or, for those that
# sum in another order and are checked at run time, stated tolerances and two identical runs on
# seeded probe inputs of the call's shape, so every rank decides alike) and keeps that one on a
# mismatch; only
# msa_bounded_plan_chunks (an integer rule for the plan's chunk sizes) and msa_ds_chunk_store
# (inside msa_dq_from_ds's comparison) rely on the unit tests alone. All are deterministic: no
# atomics, fixed summation orders.
#
# Bit-identical to the path they replace:
#   msa_lean_masks: the forward, dQ, dK/dV and indexer scoring kernels skip the per-key causal
#     select on steps whose keys are all visible (a wave-uniform branch), and the index dK
#     kernel takes the WMMA's unused K half from an LDS pad of zeros (msa_token_fwd_w32.py,
#     msa_token_bwd_w32.py, index_block_max.py).
#   msa_select_plan_wide: with msa_fused_select_plan, shapes past select_plan.supported() (more
#     than 64 blocks or 1024 rows of blocks, up to select_plan.wide_supported()) get the fused
#     selection bookkeeping and build_block_plan's FlyDSL passes (flydsl/select_plan.py).
#   msa_select_skip_fill: with msa_fused_select_plan and forced blocks, the scoring kernel's
#     output tables are left unfilled; the forced-block boost writes -inf after each query's
#     own block, and nothing reads the argmax there (index_select.py).
#   msa_owner_carry / msa_two_tile_slots: with msa_shared_tile_staging, the forward carries each
#     step's tile owner in its loop state and keeps one LDS slot for K and one for V (68 KB of
#     LDS instead of 140 KB) (msa_token_fwd_w32.py).
#   msa_walk_one_wave: the aligned walk's Triton kernel with one wave per program.
#   msa_dkdv_two_token_steps: the dK/dV kernel takes two tokens per step instead of four.
# The same terms summed in another order, so the last bits differ from the replaced path's:
#   msa_aligned_block_walk: with msa_shared_tile_staging, the forward and dQ kernels walk a
#     token's blocks in an order aligned with its neighbours', so a workgroup stages a common
#     block's tiles once (flydsl/block_walk.py).
#   msa_index_dk_scatter: the index dK chunk kernel adds each entry's fp32 g * q_row to its
#     winning key row (_build_index_dk_scatter_w32); sparse_index_dim 128, power-of-two top-k.
#   msa_indexer_loss_one_pass: with msa_fused_indexer_loss, the slot mass and indexer KL and
#     their backward in one Triton kernel each (indexer_loss_one_pass.py).
#   msa_bounded_plan_chunks: BlockPlan chunks of 256-384 entries (msa_token_bwd.plan_chunk_size),
#     so dK/dV and index dK partials are summed over other groups of entries.
_MSA_GFX1250_PATHS = (
    "msa_lean_masks",
    "msa_select_plan_wide",
    "msa_select_skip_fill",
    "msa_owner_carry",
    "msa_two_tile_slots",
    "msa_walk_one_wave",
    "msa_dkdv_two_token_steps",
    "msa_aligned_block_walk",
    "msa_index_dk_scatter",
    "msa_indexer_loss_one_pass",
    "msa_bounded_plan_chunks",
)
# Taken only with msa_bwd_store_ds (MSATransformerConfig, default true):
#   msa_dq_from_ds: the dK/dV kernel writes its bf16 dS to a workspace and a slim kernel forms
#     dQ = scale * dS K (build_dq_ds_w32); checked within 1e-3, workspaces above 16 GiB keep the
#     recomputing dQ kernel. Last bits differ (dS and delta are summed in another order).
#   msa_ds_chunk_store / msa_dq_early_issue: bit-identical on that path: one dS buffer descriptor
#     per workgroup (slices under 2 GiB) and the dQ kernel's next-step copies issued before it
#     waits for the current ones (msa_token_bwd_w32.py).
_MSA_STORE_DS_PATHS = ("msa_dq_from_ds", "msa_ds_chunk_store", "msa_dq_early_issue")
_FUSIONS_LOGGED = False


def _log_fusions_once(config, gfx1250: bool) -> None:
    global _FUSIONS_LOGGED
    if _FUSIONS_LOGGED:
        return
    _FUSIONS_LOGGED = True
    on = [name for name in _MSA_FUSIONS if getattr(config, name, False)]
    log_rank_0(f"MinimaxSparseAttention ({config.msa_backend}): Triton fusions {', '.join(on) or 'off'}")
    if config.msa_backend == "flydsl" and gfx1250:
        shared = "on" if getattr(config, "msa_shared_tile_staging", False) else "off"
        store_ds = "on" if getattr(config, "msa_bwd_store_ds", True) else "off"
        log_rank_0(
            f"MinimaxSparseAttention (flydsl, gfx1250): msa_shared_tile_staging {shared}; "
            f"msa_bwd_store_ds {store_ds} ({', '.join(_MSA_STORE_DS_PATHS)}); kernel paths where "
            f"eligible: {', '.join(_MSA_GFX1250_PATHS)}"
        )


def _require_gfx1250_flydsl() -> None:
    """Fail at build time, not at the first forward, when FlyDSL lacks what the gfx1250 MSA
    kernels use."""
    msg = (
        "msa_backend='flydsl' on gfx1250 needs FlyDSL's Tensor Data Mover ops "
        "(flydsl.expr.rocdl.tdm_ops), flydsl.expr.SharedAllocator and rocdl.ds_load_tr16_b128; "
        "FlyDSL 0.3.4.1 (the MI455X image's) has them."
    )
    try:
        import flydsl.expr as fx
        from flydsl._mlir.dialects import rocdl
        from flydsl.expr.rocdl import tdm_ops  # noqa: F401
    except ImportError as e:
        raise ImportError(msg) from e
    if not (hasattr(fx, "SharedAllocator") and hasattr(rocdl, "ds_load_tr16_b128")):
        raise ImportError(msg)


@dataclass
class MinimaxSparseAttentionSubmodules(SelfAttentionSubmodules):
    """``SelfAttentionSubmodules`` plus the indexer branch's four modules."""

    linear_index_q: Union[ModuleSpec, type] = None
    linear_index_k: Union[ModuleSpec, type] = None
    index_q_layernorm: Union[ModuleSpec, type] = None
    index_k_layernorm: Union[ModuleSpec, type] = None


class MinimaxSparseAttention(SelfAttention):
    """Self-attention layer for MiniMax-M3's MSA layers."""

    def __init__(
        self,
        config: MSATransformerConfig,
        submodules: MinimaxSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        name: Optional[str] = None,  # passed by Megatron-core 0.19+
    ):
        if not isinstance(config, MSATransformerConfig):
            raise TypeError(
                "MinimaxSparseAttention requires an MSATransformerConfig, got "
                f"{type(config).__name__}. Set `minimax_sparse_attention: true` in the model "
                "preset so the config-class patch selects it."
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            **({"name": name} if name is not None else {}),
        )

        self.block_size = config.sparse_block_size
        self.indexer_loss_coeff = config.sparse_indexer_loss_coeff
        # TEDotProductAttention keeps no `softmax_scale` attribute, so resolve it
        # the same way Attention does when it hands one to the kernel: the config
        # override if set (mu-P uses it), else 1/sqrt(head_dim) -- the
        # reference's `self.scaling = self.head_dim**-0.5`.
        self.softmax_scale = config.softmax_scale or self.hidden_size_per_attention_head**-0.5
        self.indexer = build_module(
            ModuleSpec(
                module=MinimaxM3Indexer,
                submodules=MinimaxM3IndexerSubmodules(
                    linear_index_q=submodules.linear_index_q,
                    linear_index_k=submodules.linear_index_k,
                    index_q_layernorm=submodules.index_q_layernorm,
                    index_k_layernorm=submodules.index_k_layernorm,
                ),
            ),
            config=config,
            pg_collection=self.pg_collection,
        )

        if self._fused_input_norm_weight() is not None and config.normalization != "RMSNorm":
            raise NotImplementedError(
                "MinimaxSparseAttention recomputes linear_qkv's fused input norm for the indexer "
                f"and supports RMSNorm only; got normalization={config.normalization!r}."
            )

        self.msa_attention = None
        self._gfx1250 = False
        if config.msa_backend == "flydsl":
            # Imported here, not at module level: flydsl is not in every build.
            from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_function import (
                msa_attention,
            )
            from primus.core.utils.gpu_arch import is_gfx1250

            if is_gfx1250():
                self._gfx1250 = True
                _require_gfx1250_flydsl()
                if self.block_size != 128:
                    raise NotImplementedError(
                        "msa_backend='flydsl' on gfx1250 supports sparse_block_size 128 only (the "
                        f"dK/dV backward stages 8 waves of 16 keys); got {self.block_size}."
                    )
            heads_per_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            if self.hidden_size_per_attention_head != 128 or heads_per_group != 16:
                raise NotImplementedError(
                    "msa_backend='flydsl' is built for head_dim 128 and 16 query heads per KV head "
                    f"(M3's shape); got head_dim={self.hidden_size_per_attention_head}, "
                    f"{heads_per_group} heads per KV head."
                )
            self.msa_attention = msa_attention
        _log_fusions_once(config, self._gfx1250)

    def _fused_input_norm_weight(self) -> Optional[torch.Tensor]:
        """The input-norm weight fused into ``linear_qkv``, or None when the norm is separate."""
        return getattr(self.linear_qkv, "layer_norm_weight", None)

    def _indexer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``input_layernorm(hidden_states)``, detached: what the reference indexer reads.

        The TE specs fuse the input norm into ``linear_qkv``, so ``hidden_states``
        arrives here as the raw residual stream; the norm is recomputed from the
        fused weight. Weight and input are both detached because the distillation
        loss must train the indexer alone, not the layers that feed it.
        """
        x = hidden_states.detach()
        norm_weight = self._fused_input_norm_weight()
        if norm_weight is None:
            return x
        if getattr(self.config, "msa_fused_indexer_input_norm", False):
            # the same ops in one bitwise Triton launch where its self-check passed (None
            # otherwise); imported only when switched on: the eager backend needs no Triton
            from primus.backends.megatron.core.transformer.minimax_m3 import (
                indexer_input_norm,
            )

            normed = indexer_input_norm.indexer_input_norm(
                x,
                norm_weight.detach(),
                self.config.layernorm_epsilon,
                bool(self.config.layernorm_zero_centered_gamma),
                self._indexer_input_eager,
            )
            if normed is not None:
                return normed
        return self._indexer_input_eager(x, norm_weight.detach())

    def _indexer_input_eager(self, x: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
        """The FP32 RMSNorm of ``x`` with the (detached) fused norm weight, cast back."""
        weight = norm_weight.float()
        if self.config.layernorm_zero_centered_gamma:
            weight = weight + 1.0
        normed = torch.nn.functional.rms_norm(
            x.float(), (x.shape[-1],), weight, self.config.layernorm_epsilon
        )
        return normed.to(x.dtype)

    def _aligned_walk(self, block_indices: torch.Tensor, block_plan: Optional[object]):
        """``msa_aligned_block_walk`` (with ``msa_walk_one_wave``): ``(walk_table, slots)`` of
        flydsl/block_walk.py, or None off gfx1250, when ``msa_shared_tile_staging`` is off
        (nothing would share a tile) or when the shape is not eligible. Decided before any work."""
        if not self._gfx1250 or not getattr(self.config, "msa_shared_tile_staging", False):
            return None
        # imported here: the eager backend needs no Triton
        from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
            block_walk,
        )

        table = getattr(block_plan, "table", None) if block_plan is not None else None
        if table is None or table.shape != block_indices.shape:
            table = block_indices.to(torch.int32)
        return block_walk.aligned_block_walk(table, self.block_size, one_wave=True)

    def _dq_from_ds(self, walk) -> dict:
        """``msa_bwd_store_ds`` (``msa_dq_from_ds`` with ``msa_ds_chunk_store`` and
        ``msa_dq_early_issue``): the keyword arguments that turn it on for the MSA backward, or
        none off gfx1250 or with ``msa_bwd_store_ds`` False. With an aligned walk the backward also
        needs the walk's slots as int32."""
        if not self._gfx1250 or not getattr(self.config, "msa_bwd_store_ds", True):
            return {}
        kwargs = {"dq_from_ds": True, "ds_chunk_store": True, "dq_early_issue": True}
        if walk is not None:
            kwargs["walk_slots"] = walk[1].to(torch.int32)
        return kwargs

    def _owner_carry(self) -> dict:
        """``msa_owner_carry`` (with ``msa_two_tile_slots``): the keyword arguments that turn it on
        for the MSA forward, or none off gfx1250 or when ``msa_shared_tile_staging`` is off (no
        tile owner to carry)."""
        if not self._gfx1250 or not getattr(self.config, "msa_shared_tile_staging", False):
            return {}
        return {"owner_carry": True, "two_tile_slots": True}

    def _dkdv_two_token_steps(self) -> dict:
        """``msa_dkdv_two_token_steps``: the keyword argument that turns it on for the MSA
        backward, or none off gfx1250."""
        return {"dkdv_two_token_steps": True} if self._gfx1250 else {}

    def _loss_one_pass(self) -> bool:
        """``msa_indexer_loss_one_pass``: whether ``msa_core`` asks the one-pass kernels for the
        indexer loss. It stands in for the Triton path of ``msa_fused_indexer_loss``, so it needs
        that switch, and gfx1250."""
        return bool(self._gfx1250 and getattr(self.config, "msa_fused_indexer_loss", False))

    def msa_core(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_indices: torch.Tensor,
        block_scores: torch.Tensor,
        block_plan: Optional[object],
        want_loss: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """flydsl backend: attention over the selected blocks and, with ``want_loss``, the sparse
        indexer loss.

        Takes ``[sq, b, heads, hn]`` query / key / value and the indexer's outputs; returns
        ``(core_attn_out [sq, b, np, hn], indexer_loss or None)``.
        """
        walk = self._aligned_walk(block_indices, block_plan)
        outs = self.msa_attention(
            query,
            key,
            value,
            block_indices,
            self.softmax_scale,
            self.block_size,
            return_slot_lse=want_loss,
            plan=block_plan,
            shared_tiles=getattr(self.config, "msa_shared_tile_staging", False),
            step_mask_skip=self._gfx1250,
            **({} if walk is None else {"walk_table": walk[0]}),
            **self._dq_from_ds(walk),
            **self._owner_carry(),
            **self._dkdv_two_token_steps(),
        )
        if not want_loss:
            return outs[0], None

        def exact_loss(slot_lse, lse, scores, block_indices, walk):
            """The loss on the bit-exact path (also the reference of the one-pass path's check)."""
            slot_mass = slot_mass_from_slot_lse(slot_lse, lse, block_indices.shape[1])
            if walk is not None:
                # the kernel's slot_lse is in walk order: position p of a row is the indexer's slot
                # walk[1][p]. One scatter on the [b, n_index, sq, topk] mass puts it back.
                slot_mass = torch.empty_like(slot_mass).scatter_(-1, walk[1], slot_mass)
            if self.config.msa_fused_indexer_loss:
                # bitwise, in a few Triton launches
                from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss_fused import (
                    sparse_indexer_loss as loss_fn,
                )
            else:
                loss_fn = compute_sparse_indexer_loss
            if self.config.msa_fused_indexer_loss and self.config.msa_fused_indexer_loss_scatter:
                return loss_fn(slot_mass, scores, block_indices, self.indexer_loss_coeff, fold_scatter=True)
            return loss_fn(slot_mass, scores, block_indices, self.indexer_loss_coeff)

        # msa_indexer_loss_one_pass (sums in another order): eligibility is decided before any work
        if self._loss_one_pass():
            from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss_one_pass import (
                one_pass_indexer_loss,
            )

            indexer_loss = one_pass_indexer_loss(
                outs[2],
                outs[1],
                block_scores,
                block_indices,
                walk,
                self.indexer_loss_coeff,
                exact_loss,
                self.block_size,
            )
            if indexer_loss is not None:
                return outs[0], indexer_loss
        return outs[0], exact_loss(outs[2], outs[1], block_scores, block_indices, walk)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[object] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Training-only forward: qkv -> rope -> indexer -> block-sparse attention -> proj."""
        unsupported = {
            "key_value_states": key_value_states,
            "inference_context": inference_context or inference_params,
            "attention_bias": attention_bias,
            "packed_seq_params": packed_seq_params,
        }
        for name, value in unsupported.items():
            if value is not None:
                raise NotImplementedError(f"MinimaxSparseAttention does not support {name} yet.")

        # attention_mask, when the dataloader builds one, is [b, 1, sq, sk] bool
        # with True meaning "do not attend". The eager kernel applies causality
        # itself and composes this on top, so padding is honoured. The block
        # selection is left alone: an extra allowed block is harmless, and the
        # composed mask is what decides the final weights.

        q_rope = k_rope = None
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = (
                rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
            )
            # RopeTables, not apply_rope_fusion; see the note in indexer.py. The indexer reuses
            # q_rope's tables.
            share = self.config.msa_shared_rope_tables
            q_rope = RopeTables(q_pos_emb, shared=share)
            k_rope = q_rope if k_pos_emb is q_pos_emb else RopeTables(k_pos_emb, shared=share)

        # [sq, b, np, hn] / [sq, b, ng, hn] -- upstream's projections, q/k norms and rope.
        # The Triton fusions are imported only when switched on: the eager backend needs no Triton.
        fused_qkv = False
        if self.config.msa_fused_qk_norm_rope and q_rope is not None and k_rope is q_rope:
            from primus.backends.megatron.core.transformer.minimax_m3 import (
                qk_norm_rope,
            )

            fused_qkv = qk_norm_rope.eligible(self, hidden_states, q_rope)
        if fused_qkv:
            # split + q/k norm + rope on the unsplit qkv output, bitwise (qk_norm_rope.py)
            mixed_qkv, _ = self.get_query_key_value_tensors(hidden_states, key_value_states, split_qkv=False)
            query, key, value = qk_norm_rope.qk_norm_rope(self, mixed_qkv, q_rope)
        else:
            query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
            if q_rope is not None:
                query = q_rope.apply(query, self.config.rotary_interleaved)
                key = k_rope.apply(key, self.config.rotary_interleaved)

        block_indices, block_scores, block_plan = self.indexer(self._indexer_input(hidden_states), q_rope)

        # Full recompute runs this forward twice, first under no_grad; gating on
        # grad mode computes and records the loss once, in the pass that backprops.
        want_loss = self.training and torch.is_grad_enabled() and self.indexer_loss_coeff > 0.0
        sparse_loss = self.config.sparse_indexer_loss_type == "sparse"
        sq, b = hidden_states.shape[0], hidden_states.shape[1]

        if self.msa_attention is not None:
            if attention_mask is not None:
                raise NotImplementedError(
                    "msa_backend='flydsl' applies causality itself and has no padding mask; set "
                    "`create_attention_mask_in_dataloader: false`."
                )
            if block_indices.shape[1] != key.shape[2]:
                raise NotImplementedError(
                    f"the indexer selected for {block_indices.shape[1]} KV heads but this rank holds "
                    f"{key.shape[2]}; msa_backend='flydsl' needs one selection per local KV head."
                )
            # [sq, b, heads, hn] in and out: o needs no permute before linear_proj.
            core_attn_out, indexer_loss = self.msa_core(
                query, key, value, block_indices, block_scores, block_plan, want_loss
            )
        else:
            # sbhd -> bhsd for the eager kernel, matching the reference's layout.
            block_keep = build_block_keep(block_indices, key.shape[0], self.block_size)
            need_probs = want_loss and sparse_loss
            outs = eager_block_sparse_attention(
                query.permute(1, 2, 0, 3),
                key.permute(1, 2, 0, 3),
                value.permute(1, 2, 0, 3),
                block_keep,
                self.softmax_scale,
                attention_mask=attention_mask,
                return_probs=need_probs,
            )
            core_attn_out = outs[0]
            if need_probs:
                slot_mass = slot_mass_from_probs(outs[2], block_indices, self.block_size)
                indexer_loss = compute_sparse_indexer_loss(
                    slot_mass, block_scores, block_indices, self.indexer_loss_coeff
                )
            elif want_loss:
                indexer_loss = compute_indexer_loss(
                    outs[1], block_scores, self.block_size, self.indexer_loss_coeff
                )
            # bhsd -> sbhd
            core_attn_out = core_attn_out.permute(2, 0, 1, 3)

        if want_loss:
            mtp_layers = getattr(self.config, "mtp_num_layers", None) or 0
            record_indexer_loss(indexer_loss, self.layer_number, self.config.num_layers + mtp_layers)
            core_attn_out = MSAIndexerLossAutoScaler.apply(core_attn_out, indexer_loss)

        return self.linear_proj(core_attn_out.reshape(sq, b, -1))
