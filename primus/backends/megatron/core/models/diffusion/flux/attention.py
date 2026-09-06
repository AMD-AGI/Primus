# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Flux attention mechanisms.

This module implements specialized attention for Flux's MMDiT architecture:
    - JointSelfAttention: Processes concatenated image + text tokens
    - FluxSingleAttention: Processes image tokens only
    - JointSelfAttentionSubmodules: Configuration for joint attention

These implementations follow Megatron-Core's attention patterns with
customizations for diffusion model conditioning.

Reference:
    - MMDiT: "Scaling Rectified Flow Transformers"
    - Megatron-Core: megatron.core.transformer.attention
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from megatron.core.transformer.attention import (
    Attention,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

# Fuse the QK RMS norm into the RoPE kernel, one kernel per direction instead of one
# forward and three backward. Unlike the ablation above this is numerically real: it is
# no less accurate than the path it replaces and better on dx. Off by default until the
# end-to-end number is confirmed. See scratch/mxfp6/rope/RESULTS_interleaved.md.
_MXFP6_FUSED_QK_ROPE = os.environ.get("MXFP6_FUSED_QK_ROPE", "0") == "1"

try:
    from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim
except ImportError:
    SplitAlongDim = None

try:
    from primus.backends.megatron.core.models.diffusion.common.fused_norm_rope import (
        fused_qk_norm_rope,
        fused_qkv_norm_rope,
    )
except ImportError:  # no triton, or an unsupported one
    fused_qk_norm_rope = None
    fused_qkv_norm_rope = None


def _rope_cos_sin(freqs: Tensor, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    """cos/sin as the fused kernel wants them: 2D [rows, D], contiguous, in t's dtype.

    Flux positions are per image, so ``freqs`` is [S, B, 1, D] and this flattens to
    [S*B, D] -- one row per position and batch element. A model with batch-shared
    positions gives [S, 1, 1, D] and flattens to [S, D]. The kernel takes either;
    see ``_cs_div``.

    Megatron's ``_apply_rotary_pos_emb_bshd`` derives them the same way and casts to
    the tensor's dtype before multiplying, so this keeps the arithmetic identical
    rather than quietly running the rotation at higher precision.
    """
    return (
        torch.cos(freqs).to(dtype).reshape(-1, freqs.shape[-1]).contiguous(),
        torch.sin(freqs).to(dtype).reshape(-1, freqs.shape[-1]).contiguous(),
    )


def _can_fuse_qk_norm_rope(attn, *names: str) -> bool:
    """Whether this module's QK norms are ones the fused kernel actually implements.

    Checked once at build time, not per step. ``WrappedTorchNorm`` dispatches on
    ``config.normalization`` rather than on the ``rms_norm=True`` the spec asks for, so
    a config change could hand us a LayerNorm here; the kernel does RMS only, with no
    mean subtraction and no bias. Joint blocks pass all four norms, since they carry a
    separate pair per stream and the fusion has to own every one of them or none.
    """
    if not _MXFP6_FUSED_QK_ROPE or fused_qk_norm_rope is None:
        return False
    for name in names or ("q_layernorm", "k_layernorm"):
        norm = getattr(attn, name, None)
        if not isinstance(norm, torch.nn.RMSNorm) or norm.weight is None:
            return False
    return True


def _slice_rope(rotary_pos_emb, lo: int, hi):
    """Take positions [lo, hi) of a (q_pos_emb, k_pos_emb) pair.

    RoPE is positionwise and the joint blocks concatenate along the sequence, so
    ``rope(cat(added, main))`` equals ``cat(rope(added, pos[:n]), rope(main, pos[n:]))``.
    Fusing per stream before the cat is what lets the joint blocks use the same kernel
    as the single ones, at seq 256 twice instead of 512 once.
    """
    if rotary_pos_emb is None:
        return None
    q, k = rotary_pos_emb
    return (q[lo:hi], k[lo:hi])


def _apply_qk_norm_rope(attn, query, key, q_norm, k_norm, out_dtype, rotary_pos_emb, packed_seq_params):
    """QK norm followed by RoPE, in one kernel per tensor when the shapes allow it.

    Callers that enable the fusion get Q and K back from the projection *before* the
    norm, so every path out of this function has to apply it. The fallbacks below are
    therefore not optional: dropping through without norming would silently train an
    unnormalized model.
    """
    q_pos_emb, k_pos_emb = rotary_pos_emb if rotary_pos_emb is not None else (None, None)
    cu_q = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
    cu_kv = packed_seq_params.cu_seqlens_kv if packed_seq_params is not None else None

    # The kernel rotates the whole head_dim and has no t_pass tail and no packed-sequence
    # indexing. Anything else takes the unfused path.
    fusable = (
        q_pos_emb is not None
        and cu_q is None
        and cu_kv is None
        and query.dim() == 4
        and key.dim() == 4
        and q_pos_emb.shape[-1] == query.shape[-1]
        and k_pos_emb.shape[-1] == key.shape[-1]
    )

    if fusable:
        interleaved = attn.config.rotary_interleaved
        q_cos, q_sin = _rope_cos_sin(q_pos_emb, out_dtype)
        # Self-attention hands the same freqs to both, which is worth checking for:
        # it halves the transcendental work and the two small tensors it produces.
        k_cos, k_sin = (q_cos, q_sin) if k_pos_emb is q_pos_emb else _rope_cos_sin(k_pos_emb, out_dtype)
        query = fused_qk_norm_rope(query, q_norm.weight, q_cos, q_sin, q_norm.eps, interleaved)
        key = fused_qk_norm_rope(key, k_norm.weight, k_cos, k_sin, k_norm.eps, interleaved)
        return query, key

    query = q_norm(query).to(out_dtype)
    key = k_norm(key).to(out_dtype)
    if q_pos_emb is not None:
        query = apply_rotary_pos_emb(query, q_pos_emb, config=attn.config, cu_seqlens=cu_q)
        key = apply_rotary_pos_emb(key, k_pos_emb, config=attn.config, cu_seqlens=cu_kv)
    return query, key


def _apply_qkv_norm_rope(attn, mixed_qkv, split_arg_list, q_norm, k_norm, rotary_pos_emb, packed_seq_params):
    """Split the projection output into Q, K, V with QK norm and RoPE already applied.

    The same fusion as ``_apply_qk_norm_rope``, one level up. Handing the op the whole
    projection output rather than two slices of it puts the split inside the op, so its
    backward writes d(mixed_qkv) directly at stride 3D. Splitting outside instead leaves
    autograd a concatenation to rebuild that tensor, pure data movement that the unfused
    path never paid, because Inductor folded it into its norm-backward.
    """
    D = attn.hidden_size_per_attention_head
    q_pos_emb, k_pos_emb = rotary_pos_emb if rotary_pos_emb is not None else (None, None)
    cu_q = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
    cu_kv = packed_seq_params.cu_seqlens_kv if packed_seq_params is not None else None

    fusable = (
        fused_qkv_norm_rope is not None
        and q_pos_emb is not None
        and cu_q is None
        and cu_kv is None
        and mixed_qkv.dim() == 4
        # MHA only. Under GQA the Q columns are wider than one head_dim, so the three
        # slices are not at 0, D, 2D and the kernel's column offsets would be wrong.
        and mixed_qkv.shape[-1] == 3 * D
        # The op carries one frequency table and one epsilon for both tensors.
        and k_pos_emb is q_pos_emb
        and q_pos_emb.shape[-1] == D
        and q_norm.eps == k_norm.eps
    )

    if fusable:
        cos, sin = _rope_cos_sin(q_pos_emb, mixed_qkv.dtype)
        return fused_qkv_norm_rope(
            mixed_qkv,
            q_norm.weight,
            k_norm.weight,
            cos,
            sin,
            q_norm.eps,
            attn.config.rotary_interleaved,
        )

    query, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)
    query = query.reshape(query.size(0), query.size(1), -1, D)
    query, key = _apply_qk_norm_rope(
        attn, query, key, q_norm, k_norm, value.dtype, rotary_pos_emb, packed_seq_params
    )
    return query, key, value


@dataclass
class JointSelfAttentionSubmodules:
    """
    Submodules configuration for Joint Self-Attention layer.

    Joint attention processes both image and text tokens together (MMDiT architecture).
    It requires separate QKV projections for image and text (context) streams.

    Attributes:
        linear_qkv: QKV projection for main stream (image tokens)
        added_linear_qkv: QKV projection for added stream (text/context tokens)
        core_attention: Core attention computation module
        linear_proj: Output projection for main stream
        q_layernorm: Optional layer norm for queries (main stream)
        k_layernorm: Optional layer norm for keys (main stream)
        added_q_layernorm: Optional layer norm for queries (added stream)
        added_k_layernorm: Optional layer norm for keys (added stream)

    Note:
        Flux uses RMSNorm for Q/K normalization to improve training stability.

    Reference:
        - Paper: "Scaling Rectified Flow Transformers"
    """

    linear_qkv: Union[ModuleSpec, type] = None
    added_linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    added_q_layernorm: Union[ModuleSpec, type] = None
    added_k_layernorm: Union[ModuleSpec, type] = None


class JointSelfAttention(Attention):
    """
    Joint Self-Attention for MMDiT (Multimodal Diffusion Transformer).

    Processes two token streams jointly -- main (image) and added (text) --
    by projecting each through separate QKV layers, concatenating, computing
    joint attention, then splitting back. This enables cross-modal interaction
    in Flux's "double blocks".

    Args:
        config: Transformer configuration
        submodules: JointSelfAttentionSubmodules with layer specifications
        layer_number: Layer index in the model
        attn_mask_type: Type of attention mask (default: padding)
        context_pre_only: If True, only compute Q/K/V for context (default: False)

    Input/Output:
        hidden_states [seq_main, B, H] + additional_hidden_states [seq_added, B, H]
        -> (main_output, added_output) with same shapes

    Reference:
        - "Scaling Rectified Flow Transformers"
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: JointSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        context_pre_only: bool = False,
        **kwargs,
    ):
        # Use RMSNorm for Q/K normalization (improves stability)
        config.normalization = "RMSNorm"

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            **kwargs,
        )

        # QKV projection for main stream (image tokens)
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        # QKV projection for added stream (text tokens)
        if submodules.added_linear_qkv is not None:
            self.added_linear_qkv = build_module(
                submodules.added_linear_qkv,
                self.config.hidden_size,
                self.query_projection_size + 2 * self.kv_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="qkv",
            )

        # Output projection for added stream (text tokens)
        if not context_pre_only:
            self.added_linear_proj = build_module(
                submodules.linear_proj,
                self.query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name="proj",
            )

        if (
            not context_pre_only
            and getattr(self.config, "use_dual_fp8_output_projection", False)
            and hasattr(self.linear_proj, "_fp8_config")
        ):
            from primus_turbo.pytorch.core.low_precision import ScalingGranularity

            if self.linear_proj._fp8_config.granularity == ScalingGranularity.TENSORWISE:
                from primus_turbo.pytorch.core.backend import BackendType

                from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
                    DualFP8LinearTensorwiseFunction,
                    _get_fp8_dtype,
                )

                self._dual_fp8_fn = DualFP8LinearTensorwiseFunction
                cfg = self.linear_proj._fp8_config
                self._dual_fp8_fwd_dtype = _get_fp8_dtype(cfg.format, is_fwd=True)
                self._dual_fp8_bwd_dtype = _get_fp8_dtype(cfg.format, is_fwd=False)
                self._dual_fp8_gran_value = ScalingGranularity.TENSORWISE.value
                self._dual_fp8_backend_value = BackendType.HIPBLASLT.value

        # Optional Q/K layer normalization for main stream
        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        # Optional Q/K layer normalization for added stream
        if submodules.added_q_layernorm is not None:
            self.added_q_layernorm = build_module(
                submodules.added_q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.added_q_layernorm = None

        if submodules.added_k_layernorm is not None:
            self.added_k_layernorm = build_module(
                submodules.added_k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.added_k_layernorm = None

        # Both streams have to qualify together: the two getters and the concatenation
        # in forward are one dataflow, and a half-fused version would have to normalize
        # one stream here and the other after the cat.
        self._fused_qk_rope = _can_fuse_qk_norm_rope(
            self, "q_layernorm", "k_layernorm", "added_q_layernorm", "added_k_layernorm"
        )

    def _split_qkv(self, mixed_qkv: Tensor, split: bool = True):
        """
        Split mixed QKV tensor into separate Q, K, V tensors.

        Args:
            mixed_qkv: Combined QKV tensor [seq, batch, hidden]
            split: When False, stop after the reshape and return the combined tensor
                with the split sizes. The fused QK norm+RoPE op does the split itself,
                so that the backward can write d(mixed_qkv) in place rather than leave
                a concatenation for the split to undo.

        Returns:
            Tuple of (query, key, value) tensors, or (mixed_qkv, split_arg_list)
        """
        # Reshape: [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Define split sizes for Q, K, V
        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if not split:
            return mixed_qkv, split_arg_list

        # Split tensor
        if SplitAlongDim is not None:
            # Use Transformer Engine's optimized split if available
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            # Fallback to PyTorch split
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # Reshape query: [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value

    def get_query_key_value_tensors(
        self, hidden_states: Tensor, key_value_states: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Derive Q, K, V tensors from main stream hidden states.

        Args:
            hidden_states: Main stream tokens [seq, batch, hidden]
            key_value_states: Not used for self-attention

        Returns:
            Tuple of (query, key, value) tensors
        """
        # Project to QKV: [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # Under the fusion the split belongs to the op, so hand the combined tensor to
        # forward untouched. See _apply_qkv_norm_rope.
        if self._fused_qk_rope:
            return self._split_qkv(mixed_qkv, split=False)

        # Split into Q, K, V
        query, key, value = self._split_qkv(mixed_qkv)

        # Apply optional Q/K normalization.
        #
        # The `.to(value.dtype)` is required, not defensive. Under torch.compile with
        # emulate_precision_casts off, inductor may leave the norm's output in its fp32
        # accumulation dtype, and only Q and K go through a norm. V then still carries
        # the intended dtype, so it is the reference. Without this, attention receives
        # fp32 Q/K against bf16 V and Turbo -- whose dense flash-attention backends all
        # require fp16/bf16 -- rejects the call as "No compatible backend found for
        # FlashAttnDenseDispatcher", naming shapes but never mentioning dtype. Casting
        # here rather than at the attention call keeps it inside the compiled region,
        # where it fuses into the norm's epilogue instead of costing an extra pass over
        # Q and K. This matches the reference implementation, whose QKNorm.forward in
        # backends/diffusion/models/flux/layers.py likewise returns `q.to(v), k.to(v)`.
        # When the fusion owns the norms, forward applies them together with RoPE, per
        # stream and before the concatenation. Returning Q and K un-normed here is what
        # gives the kernel the un-normed input it needs.
        if not self._fused_qk_rope:
            if self.q_layernorm is not None:
                query = self.q_layernorm(query).to(value.dtype)

            if self.k_layernorm is not None:
                key = self.k_layernorm(key).to(value.dtype)

        return query, key, value

    def get_added_query_key_value_tensors(
        self, added_hidden_states: Tensor, key_value_states: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Derive Q, K, V tensors from added stream (text) hidden states.

        Args:
            added_hidden_states: Added stream tokens [seq, batch, hidden]
            key_value_states: Not used for self-attention

        Returns:
            Tuple of (query, key, value) tensors
        """
        # Project to QKV
        mixed_qkv, _ = self.added_linear_qkv(added_hidden_states)

        # Deferred to forward under the fusion, as in the main-stream getter above.
        if self._fused_qk_rope:
            return self._split_qkv(mixed_qkv, split=False)

        # Split into Q, K, V
        query, key, value = self._split_qkv(mixed_qkv)

        # Apply optional Q/K normalization; deferred to forward under the fusion, as in
        # the main-stream getter above.
        if not self._fused_qk_rope:
            if self.added_q_layernorm is not None:
                query = self.added_q_layernorm(query).to(value.dtype)

            if self.added_k_layernorm is not None:
                key = self.added_k_layernorm(key).to(value.dtype)

        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        key_value_states: Optional[Tensor] = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        additional_hidden_states: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass: Joint attention over image and text tokens.

        Args:
            hidden_states: Main stream (image) tokens [seq_main, batch, hidden]
            attention_mask: Attention mask
            key_value_states: Not used for self-attention
            inference_params: Parameters for inference (e.g., KV cache)
            rotary_pos_emb: RoPE position embeddings
            packed_seq_params: Parameters for packed sequences
            additional_hidden_states: Added stream (text) tokens [seq_added, batch, hidden]

        Returns:
            Tuple of (main_output, added_output):
                - main_output: Processed main stream [seq_main, batch, hidden]
                - added_output: Processed added stream [seq_added, batch, hidden]
        """
        # Ensure rotary_pos_emb is a tuple for Q and K
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # Get Q, K, V for both streams
        main_qkv = self.get_query_key_value_tensors(hidden_states)
        added_qkv = self.get_added_query_key_value_tensors(additional_hidden_states)

        fused_rope_applied = False
        if self._fused_qk_rope:
            # The getters stopped at the projection, so this branch owns the split, the
            # norm and RoPE. It runs per stream and before the concatenation because
            # RoPE is positionwise: the added stream takes the leading positions of the
            # joint sequence and the main stream the rest, so each can be rotated
            # against its own slice of the table and the results concatenated as before.
            main_qkv, main_split = main_qkv
            added_qkv, added_split = added_qkv
            n_added = added_qkv.shape[0]
            # Splitting the table this way is only valid if it is laid out over the
            # joint sequence and nothing downstream is going to re-index it.
            per_stream_rope = (
                inference_params is None
                and packed_seq_params is None
                and rotary_pos_emb is not None
                and rotary_pos_emb[0].shape[0] == n_added + main_qkv.shape[0]
            )
            added_query, added_key, added_value = _apply_qkv_norm_rope(
                self,
                added_qkv,
                added_split,
                self.added_q_layernorm,
                self.added_k_layernorm,
                _slice_rope(rotary_pos_emb, 0, n_added) if per_stream_rope else None,
                packed_seq_params,
            )
            query, key, value = _apply_qkv_norm_rope(
                self,
                main_qkv,
                main_split,
                self.q_layernorm,
                self.k_layernorm,
                _slice_rope(rotary_pos_emb, n_added, None) if per_stream_rope else None,
                packed_seq_params,
            )
            # When the table could not be split, the two helpers applied the norm only
            # and the block below still owes both streams their rotation.
            fused_rope_applied = per_stream_rope
        else:
            query, key, value = main_qkv
            added_query, added_key, added_value = added_qkv

        # Concatenate streams: [added; main]
        query = torch.cat([added_query, query], dim=0)
        key = torch.cat([added_key, key], dim=0)
        value = torch.cat([added_value, value], dim=0)

        # Adjust for inference (KV caching, etc.)
        query, key, value, rotary_pos_emb, attn_mask_type, *_ = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb
        )

        # Handle packed sequences
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # Apply RoPE position embeddings
        if rotary_pos_emb is not None and not fused_rope_applied:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            cu_seqlens_q = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv if packed_seq_params is not None else None

            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        # Core attention computation
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        # Handle packed sequences output
        if packed_seq_params is not None:
            # Reshape: (t, np, hn) -> (t, b=1, h=np*hn)
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # Split output back into added and main streams
        encoder_attention_output = core_attn_out[: additional_hidden_states.shape[0], :, :]
        attention_output = core_attn_out[additional_hidden_states.shape[0] :, :, :]

        # Project outputs
        if hasattr(self, "_dual_fp8_fn"):
            from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
                _extract_fp8_weight,
            )

            w_fp8_a, w_scale_a = _extract_fp8_weight(
                self.linear_proj.weight,
                self._dual_fp8_fwd_dtype,
            )
            w_fp8_b, w_scale_b = _extract_fp8_weight(
                self.added_linear_proj.weight,
                self._dual_fp8_fwd_dtype,
            )
            result = self._dual_fp8_fn.apply(
                attention_output,
                self.linear_proj.weight,
                w_fp8_a,
                w_scale_a,
                encoder_attention_output,
                self.added_linear_proj.weight,
                w_fp8_b,
                w_scale_b,
                self._dual_fp8_fwd_dtype,
                self._dual_fp8_bwd_dtype,
                self._dual_fp8_gran_value,
                self._dual_fp8_backend_value,
            )
            output, encoder_output = result[0], result[1]
            if self.linear_proj.bias is not None:
                output = output + self.linear_proj.bias
            if self.added_linear_proj.bias is not None:
                encoder_output = encoder_output + self.added_linear_proj.bias
        else:
            output, bias = self.linear_proj(attention_output)
            encoder_output, encoder_bias = self.added_linear_proj(encoder_attention_output)
            output = output + bias
            encoder_output = encoder_output + encoder_bias

        return output, encoder_output


class FluxSingleAttention(SelfAttention):
    """
    Single-stream Self-Attention for Flux (image tokens only).

    Standard self-attention without cross-modal interaction. Used in Flux's
    "single blocks" after the joint MMDiT blocks.

    Args:
        config: Transformer configuration
        submodules: SelfAttentionSubmodules with layer specifications
        layer_number: Layer index in the model
        attn_mask_type: Type of attention mask (default: padding)
        cp_comm_type: Communication type for context parallelism
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        **kwargs,
    ):
        # Use RMSNorm for Q/K normalization
        config.normalization = "RMSNorm"

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            **kwargs,
        )

        # The original Flux proj_out (Diffusers) / linear2 (TorchTitan) is a single fused
        # projection with one bias. Megatron splits it into linear_proj + linear_fc2, so the
        # bias only needs to be on one path (linear_fc2) to preserve mathematical equivalence.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

        # Decided once, here, rather than per step: the getter and the forward have to
        # agree about who applies the norm, and a condition that could change between
        # them would drop it on the floor.
        self._fused_qk_rope = _can_fuse_qk_norm_rope(self)

    def get_query_key_value_tensors(self, *args, **kwargs):
        """
        Derive Q, K, V, realigning Q/K onto V's dtype.

        Megatron's implementation returns the QK-norm's output as-is, which under
        torch.compile can be its fp32 accumulation dtype. See the note in
        JointSelfAttention.get_query_key_value_tensors for why that breaks
        attention and why V is the reference. The joint blocks project QKV
        themselves and cast inline; the single blocks reuse Megatron's projection,
        so the cast goes here, still inside the compiled region.
        """
        if self._fused_qk_rope:
            # The fused op owns the split and the norm, so it needs the projection
            # output from before either. split_qkv=False is the base getter's own exit
            # above both; forward finishes the job. No dtype realignment is needed on
            # that path -- the op writes in Q's own dtype, so the fp32-accumulation
            # problem the cast below exists to fix cannot arise.
            return super().get_query_key_value_tensors(*args, split_qkv=False, **kwargs)

        out = super().get_query_key_value_tensors(*args, **kwargs)
        if len(out) < 3:
            # split_qkv=False: (mixed_qkv, split_arg_list), no norm applied yet.
            return out
        query, key, value, *rest = out
        return (query.to(value.dtype), key.to(value.dtype), value, *rest)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        key_value_states: Optional[Tensor] = None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass: Self-attention on image tokens.

        Args:
            hidden_states: Image tokens [seq, batch, hidden]
            attention_mask: Attention mask
            key_value_states: Not used for self-attention
            inference_params: Parameters for inference (e.g., KV cache)
            rotary_pos_emb: RoPE position embeddings
            packed_seq_params: Parameters for packed sequences

        Returns:
            Tuple of (output, bias):
                - output: Projected attention output [seq, batch, hidden]
                - bias: Projection bias (None when linear_proj has bias=False)
        """
        # Ensure rotary_pos_emb is a tuple for Q and K
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # Get Q, K, V
        qkv = self.get_query_key_value_tensors(hidden_states, key_value_states)

        fused_rope_applied = False
        if self._fused_qk_rope:
            # The getter stopped at the projection, so this owns the split, the norm and
            # the rotation, and applies the norm on its fallback paths too. KV caching
            # and packed sequences both re-index Q and K after this point, so they keep
            # the original ordering and get only the norm here.
            mixed_qkv, split_arg_list = qkv
            fused_rope_applied = inference_params is None and packed_seq_params is None
            query, key, value = _apply_qkv_norm_rope(
                self,
                mixed_qkv,
                split_arg_list,
                self.q_layernorm,
                self.k_layernorm,
                rotary_pos_emb if fused_rope_applied else None,
                packed_seq_params,
            )
        else:
            query, key, value = qkv

        # Adjust for inference
        query, key, value, rotary_pos_emb, attn_mask_type, *_ = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb
        )

        # Handle packed sequences
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # Apply RoPE position embeddings
        if rotary_pos_emb is not None and not fused_rope_applied:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            cu_seqlens_q = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv if packed_seq_params is not None else None

            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        # Core attention computation
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        # Handle packed sequences output
        if packed_seq_params is not None:
            # Reshape: (t, np, hn) -> (t, b=1, h=np*hn)
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # Project output (return both output and bias for skip_bias_add pattern)
        output, bias = self.linear_proj(core_attn_out)

        return output, bias
