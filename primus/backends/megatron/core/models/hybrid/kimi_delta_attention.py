# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Adapted from Kimi-Linear (Moonshot AI) delta attention architecture.
# Reference: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate

    HAVE_FLA_KDA = True
except ImportError:
    chunk_kda = None
    fused_kda_gate = None
    HAVE_FLA_KDA = False

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from torch.utils.checkpoint import checkpoint as _grad_checkpoint

logger = logging.getLogger(__name__)


# def torch_kda_gate(g, A_log, dt_bias=None):
#     """Pure-PyTorch replacement for ``fused_kda_gate``.

#     Computes  g_out = -exp(A_log) * softplus(g + dt_bias)  per head/dim.

#     Args:
#         g: Gate tensor of shape ``[..., H, K]``.
#         A_log: Per-head log-decay parameter of shape ``[H]``.
#         dt_bias: Optional bias of shape ``[H*K]``.

#     Returns:
#         Gated output in float32 with the same shape as *g*.
#     """
#     H = g.shape[-2]
#     g = g.float()
#     if dt_bias is not None:
#         g = g + dt_bias.view(H, -1)
#     return -A_log.view(H, 1).float().exp() * F.softplus(g)


# def torch_chunk_kda(
#     q,
#     k,
#     v,
#     g,
#     beta,
#     scale=None,
#     chunk_size=64,
#     initial_state=None,
#     output_final_state=False,
#     use_qk_l2norm_in_kernel=False,
# ):
#     """Pure-PyTorch chunked KDA (Kimi Delta Attention).

#     Drop-in replacement for ``chunk_kda`` from ``fla.ops.kda`` for platforms
#     where Triton backward kernels are unavailable (e.g. AMD MI300X / ROCm).

#     The algorithm follows the naive chunked delta-rule formulation from the FLA
#     library with per-key-dimension gating specific to KDA.

#     Args:
#         q: Queries  ``[B, T, H, K]``.
#         k: Keys     ``[B, T, H, K]``.
#         v: Values   ``[B, T, H, V]``.
#         g: Per-dim gate in log-decay space ``[B, T, H, K]`` (output of
#            ``torch_kda_gate`` / ``fused_kda_gate``).
#         beta: Per-head importance weights ``[B, T, H]`` (already sigmoided).
#         scale: Query scaling factor; defaults to ``K ** -0.5``.
#         chunk_size: Chunk size for the tiled processing.
#         initial_state: Optional initial recurrent state ``[B, H, K, V]``.
#         output_final_state: Whether to return the final state.
#         use_qk_l2norm_in_kernel: Apply L2-norm to Q/K before attention.

#     Returns:
#         ``(output, final_state)`` where *output* is ``[B, T, H, V]`` and
#         *final_state* is ``None`` unless *output_final_state* is ``True``.
#     """
#     initial_dtype = q.dtype
#     B, T, H, K = q.shape
#     V = v.shape[-1]
#     BT = chunk_size

#     if scale is None:
#         scale = K ** -0.5

#     if use_qk_l2norm_in_kernel:
#         q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
#         k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)

#     pad_size = (BT - T % BT) % BT
#     if pad_size > 0:
#         q = F.pad(q, (0, 0, 0, 0, 0, pad_size))
#         k = F.pad(k, (0, 0, 0, 0, 0, pad_size))
#         v = F.pad(v, (0, 0, 0, 0, 0, pad_size))
#         g = F.pad(g, (0, 0, 0, 0, 0, pad_size))
#         beta = F.pad(beta, (0, 0, 0, pad_size))

#     total_T = T + pad_size
#     NT = total_T // BT

#     q, k, v, g, beta = [
#         x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, beta)
#     ]
#     q = q * scale

#     q = q.reshape(B, H, NT, BT, K)
#     k = k.reshape(B, H, NT, BT, K)
#     v = v.reshape(B, H, NT, BT, V)
#     g = g.reshape(B, H, NT, BT, K)
#     beta = beta.reshape(B, H, NT, BT)

#     g = g.cumsum(dim=-2)

#     k_eg = k * g.exp()

#     # --- WY A-matrix (per-dim stable) ---
#     # A[c1,c2] = Σ_d k[c1,d]·exp(g[c1,d]-g[c2,d])·k[c2,d]
#     # We compute exp(g-g_j) per column j to stay numerically stable
#     # (g[c1]-g[j] <= 0 for c1 >= j, so exp never overflows).
#     A = torch.zeros(B, H, NT, BT, BT, dtype=torch.float, device=q.device)
#     for j in range(BT):
#         k_j = k[..., j, :]
#         g_j = g[..., j : j + 1, :]
#         decay = (g - g_j).clamp(max=0).exp()
#         A[..., j] = (k * decay * k_j.unsqueeze(-2)).sum(-1)

#     A = A * beta.unsqueeze(-1)

#     mask_upper = torch.triu(
#         torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0
#     )
#     A = -A.masked_fill(mask_upper, 0)

#     for i in range(1, BT):
#         A[..., i, :i] = A[..., i, :i].clone() + (
#             A[..., i, :, None].clone() * A[..., :, :i].clone()
#         ).sum(-2)

#     A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta.unsqueeze(-2)

#     w = A @ k_eg
#     u = A @ v

#     mask_causal = torch.triu(
#         torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1
#     )

#     S = q.new_zeros(B, H, K, V)
#     if initial_state is not None:
#         S = S + initial_state.float()

#     o = torch.zeros_like(v)

#     for i in range(NT):
#         q_i = q[:, :, i]
#         k_i = k[:, :, i]
#         u_i = u[:, :, i]
#         g_i = g[:, :, i]
#         w_i = w[:, :, i]

#         # Intra-chunk QK attention (per-column loop for stability)
#         A_qk = torch.zeros(B, H, BT, BT, dtype=torch.float, device=q.device)
#         for j in range(BT):
#             k_j = k_i[..., j, :]
#             g_j = g_i[..., j : j + 1, :]
#             decay = (g_i - g_j).clamp(max=0).exp()
#             A_qk[..., j] = (q_i * decay * k_j.unsqueeze(-2)).sum(-1)
#         A_qk = A_qk.masked_fill(mask_causal, 0)

#         v_i = u_i - w_i @ S
#         o[:, :, i] = (q_i * g_i.exp()) @ S + A_qk @ v_i

#         g_last = g_i[:, :, -1]
#         S = S * g_last.unsqueeze(-1).exp()
#         k_dec = (g_last.unsqueeze(-2) - g_i).exp() * k_i
#         S = S + k_dec.transpose(-1, -2) @ v_i

#     if not output_final_state:
#         S = None

#     o = o.reshape(B, H, -1, V)[:, :, :T]
#     o = o.transpose(1, 2).contiguous().to(initial_dtype)
#     return o, S


# def _torch_chunk_kda_ckpt(q, k, v, g, beta):
#     """Thin wrapper for ``torch_chunk_kda`` used with gradient checkpointing.

#     Returns only the output tensor so that ``torch.utils.checkpoint`` does not
#     need to handle ``None`` as the second tuple element.
#     """
#     o, _ = torch_chunk_kda(
#         q=q, k=k, v=v, g=g, beta=beta,
#         initial_state=None,
#         output_final_state=False,
#         use_qk_l2norm_in_kernel=True,
#     )
#     return o


@dataclass
class KimiDeltaAttentionSubmodules:
    """Module specs for the Kimi Delta Attention (KDA) layer.

    Uses a single fused in_proj (with TELayerNormColumnParallelLinear) for
    combined Q/K/V projection, mirroring GatedDeltaNet's approach.
    A separate gate_norm is provided for the side projections (gate, beta,
    output_gate) that cannot be included in the column-parallel in_proj.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp
    gate_norm: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class KimiDeltaAttention(MegatronModule):
    """Kimi Delta Attention (KDA) layer class.

    Based on the architecture from Kimi-Linear (Moonshot AI).
    Key differences from GatedDeltaNet:
    - Separate causal conv1d per Q, K, V (combined into one depthwise conv).
    - Low-rank gate factorization (f_a -> f_b) instead of direct alpha projection.
    - Low-rank output gate (g_a -> g_b) instead of direct gate projection.
    - Uses chunk_kda / fused_kda_gate from fla.ops.kda.

    KDA layer takes input with size [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_FLA_KDA and getattr(config, 'use_fla_triton_kda', False):
            raise ImportError(
                "use_fla_triton_kda is set but FLA KDA ops are not installed. "
                "Install fla-core or remove the flag to use the pure-PyTorch default."
            )

        super().__init__(config)

        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        assert pg_collection is not None, "pg_collection must be provided for KimiDeltaAttention"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.head_k_dim = config.linear_key_head_dim
        self.head_dim = config.linear_value_head_dim
        self.num_k_heads = config.linear_num_key_heads
        self.num_heads = config.linear_num_value_heads
        self.qk_dim = self.head_k_dim * self.num_k_heads
        self.v_dim = self.head_dim * self.num_heads
        self.num_heads_local_tp = self.num_heads // self.tp_size
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size

        # --- Fused Q+K+V projection (column-parallel, with fused LayerNorm) ---
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kda_in",
            tp_group=self.pg_collection.tp,
        )

        # --- Combined causal conv1d for Q, K, V (depthwise) ---
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)

        # --- Norm for gate/beta/output_gate side projections ---
        # The main norm is fused into in_proj (TELayerNormColumnParallelLinear).
        # Side projections (f_a, g_a, b_proj) also need normed input.
        self.gate_norm = build_module(
            submodules.gate_norm, config=self.config,
            hidden_size=self.hidden_size, eps=self.config.layernorm_epsilon,
        )

        # --- Low-rank gate: f_a (bottleneck) -> f_b (expand to heads) ---
        self.f_a_proj = nn.Linear(
            self.hidden_size, self.head_dim, bias=False,
            device=torch.cuda.current_device(), dtype=config.params_dtype,
        )
        self.f_b_proj = nn.Linear(
            self.head_dim, self.v_dim_local_tp, bias=False,
            device=torch.cuda.current_device(), dtype=config.params_dtype,
        )
        setattr(self.f_b_proj.weight, "tensor_model_parallel", True)

        # --- A_log and dt_bias (TP-split per head) ---
        self.A_log = nn.Parameter(torch.empty(
            1, 1, self.num_heads_local_tp, 1,
            dtype=torch.float32, device=torch.cuda.current_device(),
        ))
        setattr(self.A_log, "tensor_model_parallel", True)

        self.dt_bias = nn.Parameter(torch.empty(
            self.v_dim_local_tp,
            dtype=torch.float32, device=torch.cuda.current_device(),
        ))
        setattr(self.dt_bias, "tensor_model_parallel", True)

        # --- Beta projection (hidden -> num_heads, TP-split) ---
        self.b_proj = nn.Linear(
            self.hidden_size, self.num_heads_local_tp, bias=False,
            device=torch.cuda.current_device(), dtype=config.params_dtype,
        )
        setattr(self.b_proj.weight, "tensor_model_parallel", True)

        # --- Low-rank output gate: g_a (bottleneck) -> g_b (expand) ---
        self.g_a_proj = nn.Linear(
            self.hidden_size, self.head_dim, bias=False,
            device=torch.cuda.current_device(), dtype=config.params_dtype,
        )
        self.g_b_proj = nn.Linear(
            self.head_dim, self.v_dim_local_tp, bias=False,
            device=torch.cuda.current_device(), dtype=config.params_dtype,
        )
        setattr(self.g_b_proj.weight, "tensor_model_parallel", True)

        # --- Output norm and projection ---
        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.head_dim,
            eps=self.config.layernorm_epsilon,
        )
        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="kda_out",
            tp_group=self.pg_collection.tp,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                A = torch.empty(
                    self.num_heads_local_tp,
                    dtype=torch.float32, device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A).view(1, 1, -1, 1))
                nn.init.ones_(self.dt_bias)

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """Forward pass through the KDA module.

        Args:
            hidden_states (Tensor): [s, b, h] input tensor (un-normed).
            attention_mask (Tensor): Attention mask (reserved for future use).

        Returns:
            Tuple[Tensor, Tensor]: KDA output and bias.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        local_seq_len, batch, _ = hidden_states.shape
        seq_len = local_seq_len * self.sp_size
        use_fla_triton = True #(not self.config.deterministic_mode) and HAVE_FLA_KDA and getattr(self.config, 'use_fla_triton_kda', False)

        if inference_context is not None:
            raise NotImplementedError("KDA does not support inference yet.")
        if packed_seq_params is not None:
            raise NotImplementedError("KDA does not support packed sequences yet.")

        # --- Fused Q+K+V projection (LayerNorm is fused into in_proj) ---
        nvtx_range_push(suffix="kda_in_proj")
        qkv, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="kda_in_proj")

        # s b d -> b s d  (output is full seq_len from TE column-parallel)
        qkv = qkv.transpose(0, 1)

        # --- Causal conv1d on combined QKV ---
        nvtx_range_push(suffix="kda_conv")
        qkv = qkv.transpose(1, 2).contiguous()  # b s d -> b d s
        if (causal_conv1d_fn is None) or self.config.deterministic_mode:
            qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        else:
            assert self.activation in ["silu", "swish"]
            qkv = causal_conv1d_fn(
                x=qkv, weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias, activation=self.activation,
            )
        qkv = qkv.transpose(1, 2)  # b d s -> b s d
        nvtx_range_pop(suffix="kda_conv")

        # Split into Q, K, V
        q, k, v = torch.split(
            qkv,
            [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
            dim=-1,
        )

        # Reshape to (batch, seq, heads, head_dim)
        q = q.reshape(batch, seq_len, -1, self.head_k_dim)
        k = k.reshape(batch, seq_len, -1, self.head_k_dim)
        v = v.reshape(batch, seq_len, -1, self.head_dim)

        if self.num_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_heads // self.num_k_heads, dim=2)
            k = k.repeat_interleave(self.num_heads // self.num_k_heads, dim=2)

        # --- Gate / beta / output_gate side projections ---
        nvtx_range_push(suffix="kda_gate")
        h_normed = self.gate_norm(hidden_states)  # [s/tp or s, b, h]
        if self.sp_size > 1:
            h_normed = gather_from_sequence_parallel_region(
                h_normed, tensor_parallel_output_grad=True,
            )
        h_bsh = h_normed.transpose(0, 1)  # s b h -> b s h

        g = self.f_b_proj(self.f_a_proj(h_bsh))
        g = g.reshape(batch, seq_len, self.num_heads_local_tp, self.head_dim)

        if use_fla_triton:
            g = fused_kda_gate(g, self.A_log.view(-1), dt_bias=self.dt_bias)
        else:
            g = torch_kda_gate(g, self.A_log.view(-1), dt_bias=self.dt_bias)

        beta = self.b_proj(h_bsh).float().sigmoid()
        nvtx_range_pop(suffix="kda_gate")

        # --- Core KDA attention ---
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        nvtx_range_push(suffix="kda_attn")
        if use_fla_triton:
            core_attn_out, _ = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out = _grad_checkpoint(
                _torch_chunk_kda_ckpt, q, k, v, g, beta,
                use_reentrant=False,
            )
        nvtx_range_pop(suffix="kda_attn")

        # --- Output gate (g_a -> g_b) + gated norm ---
        nvtx_range_push(suffix="kda_out_gate")
        gate = self.g_b_proj(self.g_a_proj(h_bsh))
        gate = gate.reshape(batch, seq_len, -1, self.head_dim)
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="kda_out_gate")

        # b s (h*d) -> s b (h*d)
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # --- Output projection ---
        nvtx_range_push(suffix="kda_out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="kda_out_proj")

        return out, out_bias

    def _apply_gated_norm(self, x, gate):
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * gate.float().sigmoid()
        y = y.to(x_dtype)
        return y

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 2,
                "dt_bias": 0,
                "b_proj.weight": 0,
                "f_b_proj.weight": 0,
                "g_b_proj.weight": 0,
            },
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )

        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == "conv1d":
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {"weight": 0}
                if self.conv_bias:
                    tp_sharding_map["bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd, f"{prefix}{name}.", tp_sharding_map,
                    sharded_offsets, tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)

        # Split the combined in_proj into named chunks for checkpoint compatibility
        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )
        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
            ],
            ["query", "key", "value"],
            0,
        )

        # Split conv1d (same ordering as in_proj: Q, K, V)
        conv_layer_name_list = ["conv1d.weight"]
        assert (
            sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == self.conv_dim_local_tp
        ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.weight"])
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
            assert (
                sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == self.conv_dim_local_tp
            ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.bias"])
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [
                    self.qk_dim // self.tp_size,
                    self.qk_dim // self.tp_size,
                    self.v_dim // self.tp_size,
                ],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict

    def backward_dw(self):
        """Execute weight gradient computation for all linear layers."""
        self.in_proj.backward_dw()
        self.out_proj.backward_dw()


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key, data=t, dtype=t.dtype,
            replica_id=replica_id, flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start, orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens, t.shape,
        )
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key, orig_sh_ten.data, sh_ten_build_fn, sh_ten_merge_fn, orig_sh_ten.replica_id
    )
