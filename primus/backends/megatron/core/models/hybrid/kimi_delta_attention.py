# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Adapted from Kimi-Linear (Moonshot AI) delta attention architecture.
# Reference: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct

import logging
import math
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
    from fla.modules import FusedRMSNormGated

    HAVE_FUSED_RMS_NORM_GATED = True
except ImportError:
    FusedRMSNormGated = None
    HAVE_FUSED_RMS_NORM_GATED = False

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

# Optional FLA Triton causal_conv1d (accepts `[B, T, D]` directly, no
# transpose/contiguous needed). Matches the conv1d backend FLA's reference
# `ShortConvolution` uses in `fla/layers/kda.py`, so when this is enabled
# Primus runs the same kernel as FLA. Gated by `PRIMUS_FLA_CONV=1` to match
# GDN's parity recipe (`GDN_FLA_PARITY.md` "Env vars" table).
import os as _os
_USE_FLA_CONV = _os.environ.get('PRIMUS_FLA_CONV', '0') == '1'
try:
    from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d
except ImportError:
    _fla_causal_conv1d = None

from torch.utils.checkpoint import checkpoint as _grad_checkpoint

logger = logging.getLogger(__name__)


def torch_kda_gate(g, A_log, dt_bias=None):
    """Pure-PyTorch KDA gate (used for backward pass on ROCm)."""
    H = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    return -A_log.view(H, 1).float().exp() * F.softplus(g)


def _torch_chunk_kda_fwd(q, k, v, g, beta, scale=None, chunk_size=64,
                          use_qk_l2norm_in_kernel=False):
    """Pure-PyTorch chunked KDA forward -- used for backward recomputation
    on ROCm where FLA Triton backward kernels hang."""
    initial_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size

    if scale is None:
        scale = K ** -0.5

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)

    pad_size = (BT - T % BT) % BT
    if pad_size > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_size))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_size))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_size))
        g = F.pad(g, (0, 0, 0, 0, 0, pad_size))
        beta = F.pad(beta, (0, 0, 0, pad_size))

    total_T = T + pad_size
    NT = total_T // BT

    q, k, v, g, beta = [
        x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, beta)
    ]
    q = q * scale

    q = q.reshape(B, H, NT, BT, K)
    k = k.reshape(B, H, NT, BT, K)
    v = v.reshape(B, H, NT, BT, V)
    g = g.reshape(B, H, NT, BT, K)
    beta = beta.reshape(B, H, NT, BT)

    g = g.cumsum(dim=-2)
    k_eg = k * g.exp()

    A = torch.zeros(B, H, NT, BT, BT, dtype=torch.float, device=q.device)
    for j in range(BT):
        k_j = k[..., j, :]
        g_j = g[..., j : j + 1, :]
        decay = (g - g_j).clamp(max=0).exp()
        A[..., j] = (k * decay * k_j.unsqueeze(-2)).sum(-1)

    A = A * beta.unsqueeze(-1)
    mask_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = -A.masked_fill(mask_upper, 0)

    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (
            A[..., i, :, None].clone() * A[..., :, :i].clone()
        ).sum(-2)

    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta.unsqueeze(-2)
    w = A @ k_eg
    u = A @ v

    mask_causal = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    S = q.new_zeros(B, H, K, V)
    o = torch.zeros_like(v)

    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:,:,i], k[:,:,i], u[:,:,i], g[:,:,i], w[:,:,i]
        A_qk = torch.zeros(B, H, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k_i[..., j, :]
            g_j = g_i[..., j : j + 1, :]
            decay = (g_i - g_j).clamp(max=0).exp()
            A_qk[..., j] = (q_i * decay * k_j.unsqueeze(-2)).sum(-1)
        A_qk = A_qk.masked_fill(mask_causal, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + A_qk @ v_i
        g_last = g_i[:, :, -1]
        S = S * g_last.unsqueeze(-1).exp()
        k_dec = (g_last.unsqueeze(-2) - g_i).exp() * k_i
        S = S + k_dec.transpose(-1, -2) @ v_i

    o = o.reshape(B, H, -1, V)[:, :, :T]
    o = o.transpose(1, 2).contiguous().to(initial_dtype)
    return o


class _HybridChunkKDA(torch.autograd.Function):
    """Triton forward + PyTorch backward for chunk_kda on ROCm MI300X.

    FLA Triton forward kernels work on ROCm but the backward kernels hang.
    This uses Triton for the fast forward pass and recomputes with PyTorch
    for the backward pass.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        ctx.save_for_backward(q, k, v, g, beta)
        with torch.no_grad():
            o, _ = chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, g, beta = ctx.saved_tensors
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            g = g.detach().requires_grad_(True)
            beta = beta.detach().requires_grad_(True)
            o = _torch_chunk_kda_fwd(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
            o.backward(grad_output)
        return q.grad, k.grad, v.grad, g.grad, beta.grad


def hybrid_chunk_kda(q, k, v, g, beta):
    """chunk_kda: Triton forward, PyTorch backward (for ROCm MI300X)."""
    return _HybridChunkKDA.apply(q, k, v, g, beta)


def _torch_chunk_kda_ckpt(q, k, v, g, beta):
    """PyTorch-only fallback with gradient checkpointing."""
    return _torch_chunk_kda_fwd(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)


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

        # --- Fused projection (column-parallel) ---
        # Match GDN's parity recipe: pack EVERY `hidden_states → X` projection
        # into a single big matmul, so the per-step kernel-launch overhead drops
        # from ~6 launches/layer (FLA reference) or 4 (un-fused Primus) to 1.
        # The fused output is split downstream into:
        #
        #   [ qkv  (qk_dim*2 + v_dim) | f_a (head_v_dim) | g_a (head_v_dim) | beta (num_v_heads) ]
        #
        # f_a and g_a are the low-rank-bottleneck *inputs* to the (cheap)
        # f_b / g_b expansion projections — those stay separate because their
        # input is the 64-dim bottleneck output, not hidden_states.
        # b_proj's output is the raw beta gate (still sigmoid-applied later).
        #
        # FLA does these as six separate `nn.Linear` modules in
        # `fla/layers/kda.py:142-189`; numerically identical, but on ROCm each
        # separate matmul pays ~3-5 ms of HIP dispatch + autograd overhead, so
        # GDN's parity work measured this fusion alone at ~250 ms/iter saved.
        self.gate_dim_local_tp = self.num_heads_local_tp * self.head_k_dim   # used below
        # NOTE on TP: the fused in_proj is a ColumnParallelLinear, which splits
        # its output evenly across TP ranks. For the gate-bottleneck slices
        # (f_a, g_a) this is incorrect — the low-rank gate REQUIRES each rank
        # to see the FULL bottleneck output before f_b_proj / g_b_proj expand
        # it (otherwise per-rank f_b would map a non-contiguous slice of the
        # bottleneck to a chunk of the gate, producing the wrong output).
        # Hard-assert tp_size=1 here; if you ever need TP>1 for KDA, the right
        # recipe is to keep f_a_proj / g_a_proj as replicated nn.Linear modules
        # (so each rank computes the full bottleneck independently) and only
        # fuse q/k/v/beta into the column-parallel in_proj.
        assert self.tp_size == 1, (
            f"KDA fused in_proj currently requires tp_size=1 (got {self.tp_size}). "
            "See KimiDeltaAttention.__init__ for the TP>1 recipe."
        )
        self.in_proj_dim = (
            self.qk_dim * 2
            + self.v_dim
            + self.head_dim          # f_a output (bottleneck dim, = head_v_dim)
            + self.head_dim          # g_a output (bottleneck dim, = head_v_dim)
            + self.num_heads         # beta (per value-head scalar)
        )
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
        # Retained for backward compat with the TE-fused spec, which keeps
        # `in_proj` as TELayerNormColumnParallelLinear and computes the
        # side-projection inputs via this second norm.  In the no-TE spec
        # the wrapper layer (`KimiDeltaAttentionLayer.forward`) already
        # pre-norms hidden_states once, so submodules.gate_norm is
        # `IdentityOp` here and the call is a no-op.
        self.gate_norm = build_module(
            submodules.gate_norm, config=self.config,
            hidden_size=self.hidden_size, eps=self.config.layernorm_epsilon,
        )

        # --- Low-rank gate expansion: f_b (bottleneck → gate_dim) ---
        # f_a (hidden → head_v_dim) is now FUSED into the big in_proj above
        # so only the cheap 64→256 bottleneck-expand matmul remains here.
        # Gate g has shape [B, T, H, K] (per-key-dim gating); output dim
        # must match q/k after the optional repeat_interleave: num_heads * head_k_dim.
        self.f_b_proj = nn.Linear(
            self.head_dim, self.gate_dim_local_tp, bias=False,
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
            self.gate_dim_local_tp,
            dtype=torch.float32, device=torch.cuda.current_device(),
        ))
        setattr(self.dt_bias, "tensor_model_parallel", True)

        # b_proj (hidden → num_v_heads) is now FUSED into the big in_proj
        # above; beta is read out of the in_proj split and sigmoid-activated
        # in the forward pass.

        # --- Low-rank output gate expansion: g_b (bottleneck → value_dim) ---
        # g_a (hidden → head_v_dim) is FUSED into the big in_proj above; only
        # the cheap 64→512 bottleneck-expand matmul remains here.
        # Match FLA: g_proj's second linear has bias=True — the bias gives
        # the output gate a learnable pre-sigmoid offset that compounds
        # across all KDA layers (fla/layers/kda.py:189).
        self.g_b_proj = nn.Linear(
            self.head_dim, self.v_dim_local_tp, bias=True,
            device=torch.cuda.current_device(), dtype=config.params_dtype,
        )
        setattr(self.g_b_proj.weight, "tensor_model_parallel", True)
        setattr(self.g_b_proj.bias, "tensor_model_parallel", True)

        # --- Output norm and projection ---
        # Match FLA: use FusedRMSNormGated (RMSNorm + sigmoid-gate + multiply in
        # ONE Triton kernel). This avoids materializing the post-norm tensor and
        # the fp32-upcast gate for backward — saves ~6.4 GiB of activation memory
        # per rank at micro_batch=128 vs the unfused (out_norm + _apply_gated_norm)
        # path. Enabled when fla.modules.FusedRMSNormGated is importable AND the
        # config opts in via use_fla_fused_norm_gated (default: True when use_fla_triton_kda).
        self._use_fla_fused_norm_gated = (
            HAVE_FUSED_RMS_NORM_GATED
            and getattr(self.config, 'use_fla_fused_norm_gated',
                        getattr(self.config, 'use_fla_triton_kda', False))
        )
        if self._use_fla_fused_norm_gated:
            self.out_norm = FusedRMSNormGated(
                self.head_dim,
                activation="sigmoid",
                eps=self.config.layernorm_epsilon,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        else:
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
                # Match FLA dt_bias init exactly (fla/layers/kda.py:180-184):
                # log-uniform sample dt in [0.001, 0.1], then store its inverse-
                # softplus so that softplus(0 + dt_bias) ≈ dt at iter 0. Replaces
                # the naive ones_ init which gave dt ≈ 1.31 — a ~20x larger
                # initial decay step that compounds across all KDA layers and
                # produces a noticeable loss-curve drift from FLA by iter ~100.
                dt = torch.exp(
                    torch.rand(
                        self.gate_dim_local_tp,
                        dtype=torch.float32, device=torch.cuda.current_device(),
                    ) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                self.dt_bias.data.copy_(inv_dt)
                # g_b_proj.bias = 0 (PyTorch nn.Linear default for bias=True
                # initialises bias uniform in [-1/sqrt(in), +1/sqrt(in)]; FLA
                # uses default nn.Linear which gives the same thing, so we
                # leave it as nn.Linear default — no explicit init here).

    # NOTE: GDN's matched-parity forward (`gated_delta_net.py:285`) does NOT
    # carry `@torch.compiler.disable`. The decorator was added defensively
    # for KDA because the chunk_kda Triton kernel doesn't trace through
    # `torch.compile`, but Megatron does not currently wrap mixer forwards
    # in `torch.compile` at all, so the decorator was only adding ~20-30 ms
    # of per-call eager-dispatch overhead (12 layers × ~2 ms × 2 fwd+bwd).
    # Removing it matches GDN's recipe exactly.
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
        use_fla_triton = (not self.config.deterministic_mode) and HAVE_FLA_KDA and getattr(self.config, 'use_fla_triton_kda', False)

        if not hasattr(self, '_kda_kernel_logged'):
            self._kda_kernel_logged = True
            use_hybrid = getattr(self.config, 'use_fla_triton_kda_hybrid', False)
            in_kernel_gate = getattr(self.config, 'use_fla_kda_in_kernel_gate', True)
            if use_fla_triton and use_hybrid:
                mode = "Hybrid (Triton fwd, PyTorch bwd), gate materialized"
            elif use_fla_triton and in_kernel_gate:
                mode = "FLA Triton fwd+bwd, gate fused in kernel"
            elif use_fla_triton:
                mode = "FLA Triton fwd+bwd, explicit fused_kda_gate (tw006)"
            else:
                mode = "Pure PyTorch fallback (gradient checkpointed)"
            norm_mode = (
                "FusedRMSNormGated (FLA)" if self._use_fla_fused_norm_gated
                else "unfused (out_norm + sigmoid-multiply)"
            )
            logger.warning(
                f"[KDA layer {self.layer_number}] kernel={mode} | norm={norm_mode} | "
                f"HAVE_FLA_KDA={HAVE_FLA_KDA} use_fla_triton_kda={getattr(self.config, 'use_fla_triton_kda', False)} "
                f"deterministic={self.config.deterministic_mode}"
            )

        if inference_context is not None:
            raise NotImplementedError("KDA does not support inference yet.")
        if packed_seq_params is not None:
            raise NotImplementedError("KDA does not support packed sequences yet.")

        # --- Single fused projection ---
        # One big GEMM produces [q | k | v | f_a_out | g_a_out | beta_raw] for the
        # whole layer — matches GDN's parity recipe and removes 3 extra hidden→X
        # matmul launches per layer (~5 ms × 12 layers ≈ saved 60 ms/iter alone,
        # plus the corresponding backward GEMMs, plus removed activation copies
        # for h_normed).
        nvtx_range_push(suffix="kda_in_proj")
        fused, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="kda_in_proj")

        # s b d -> b s d  (output is full seq_len from column-parallel)
        fused = fused.transpose(0, 1)

        # Split into [qkv | f_a_out | g_a_out | beta_raw]. The slice sizes are
        # the per-TP local dims; sum equals self.in_proj_dim // tp_size.
        qkv_local = self.qk_dim_local_tp * 2 + self.v_dim_local_tp
        head_dim_local_tp = self.head_dim // self.tp_size
        qkv, f_a_out, g_a_out, beta_raw = torch.split(
            fused,
            [qkv_local, head_dim_local_tp, head_dim_local_tp, self.num_heads_local_tp],
            dim=-1,
        )

        # --- Causal conv1d on combined QKV ---
        # Three backends, chosen at runtime:
        #   (1) PRIMUS_FLA_CONV=1  → FLA's Triton causal_conv1d. Accepts
        #       [B, T, D] directly, matches FLA's reference run bit-for-bit,
        #       and removes the two transposes + contiguous() copy below.
        #       Saves ~3 ms/layer × 12 = ~35 ms/iter and ~1.6 GiB/iter peak
        #       (the contiguous() copy was a full-qkv allocation).
        #   (2) Tri-Dao's causal_conv1d_fn (CUDA package) — current default.
        #   (3) Pure-PyTorch fallback when neither is available or
        #       deterministic_mode is set.
        nvtx_range_push(suffix="kda_conv")
        if _USE_FLA_CONV and _fla_causal_conv1d is not None and not self.config.deterministic_mode:
            assert self.activation in ["silu", "swish"]
            qkv, _ = _fla_causal_conv1d(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation=self.activation,
                backend='triton',
            )
        else:
            qkv = qkv.transpose(1, 2).contiguous()  # b s d -> b d s
            if (causal_conv1d_fn is None) or self.config.deterministic_mode:
                qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
            else:
                assert self.activation in ["silu", "swish"]
                qkv = causal_conv1d_fn(
                    x=qkv, weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias, activation=self.activation,
                )
            # b d s -> b s d. Do NOT contiguous() here — the downstream
            # torch.split produces non-contiguous views along dim=-1 regardless,
            # so a copy is forced inside .reshape() anyway.
            qkv = qkv.transpose(1, 2)  # b d s -> b s d
        nvtx_range_pop(suffix="kda_conv")

        # Split conv output into Q, K, V
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

        # --- Gate (low-rank expansion only — f_a is already inside in_proj) ---
        nvtx_range_push(suffix="kda_gate")
        g = self.f_b_proj(f_a_out)
        g = g.reshape(batch, seq_len, self.num_heads_local_tp, self.head_k_dim)

        # Match FLA: when using the Triton fwd+bwd path, pass RAW g into the
        # kernel and let it fuse `-exp(A_log) * softplus(g + dt_bias) + cumsum`
        # internally (use_gate_in_kernel=True). The kernel recomputes the gate
        # in backward, so the fp32 [B,T,H,K] activated-gate tensor is never
        # materialized — saves ~3.2 GiB of activation memory at micro_batch=128.
        # For non-Triton paths (deterministic/CPU/fallback) we still compute
        # the gate up front in PyTorch.
        # Two opt-outs:
        #   - use_fla_triton_kda_hybrid: routes to hybrid_chunk_kda (Triton fwd,
        #     PyTorch bwd) for debugging; gate is materialized in PyTorch.
        #   - use_fla_kda_in_kernel_gate=False: keeps the standard chunk_kda
        #     path but materializes the gate up front (via fused_kda_gate),
        #     mirroring the pre-fusion tw006 numerics. The forward output is
        #     bit-identical between paths in fp32 but the bf16 in-kernel fused
        #     `-exp(A_log)*softplus(g+dt_bias)` accumulator has +/-1 ulp drift
        #     vs the explicit-gate path; with 12 layers of compounding this
        #     gives ~0.2 lm-loss above tw006 by iter 200. Set to False to
        #     recover tw006-tight loss curve at the cost of ~3 GiB extra
        #     activation memory.
        _use_in_kernel_gate = (
            use_fla_triton
            and getattr(self.config, 'use_fla_kda_in_kernel_gate', True)
            and not getattr(self.config, 'use_fla_triton_kda_hybrid', False)
        )
        if not _use_in_kernel_gate:
            if use_fla_triton:
                g = fused_kda_gate(g, self.A_log.view(-1), dt_bias=self.dt_bias)
            else:
                g = torch_kda_gate(g, self.A_log.view(-1), dt_bias=self.dt_bias)

        # beta = sigmoid of the raw beta slice from the fused in_proj. Match
        # FLA's `b_proj(h).sigmoid()` (fla/layers/kda.py:251) in bf16 exactly.
        # The earlier fp32 upcast was a defensive measure against bf16 drift
        # compounding across 12 layers, but with both in-kernel fusions enabled
        # the drift was empirically <0.001 vs fp32 at iter 200 — within batch
        # noise. Saves ~256 MiB/layer × 12 = ~6 GiB activation peak and ~5 ms.
        beta = beta_raw.sigmoid()
        nvtx_range_pop(suffix="kda_gate")

        # --- Core KDA attention ---
        # Q/K/V contiguity: only required when the in_proj is the TE-fused
        # variant (TELayerNormColumnParallelLinear), which leaves a non-contig
        # stride pattern that interacts pathologically with chunk_kda's bwd
        # (~29 GiB extra activation memory in the TE-fused path). With the
        # no-TE spec (plain ColumnParallelLinear → torch.split), strides are
        # already clean and match FLA's `rearrange()` output, so the explicit
        # .contiguous() calls just waste ~5-8 GiB on saved-for-backward
        # duplicates. Auto-detected via gate_norm == IdentityOp.
        if not isinstance(self.gate_norm, IdentityOp):
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        nvtx_range_push(suffix="kda_attn")
        use_hybrid = getattr(self.config, 'use_fla_triton_kda_hybrid', False)
        if _use_in_kernel_gate:
            # FLA's new (post-fusion) call site — fuses the gate compute and
            # qk-l2norm inside chunk_kda. Smallest activation footprint, but
            # the bf16 accumulator drifts ~+0.2 lm-loss vs the explicit-gate
            # path on ROCm at 12 layers depth.
            core_attn_out, _ = chunk_kda(
                q, k, v, g, beta,
                A_log=self.A_log.view(-1),
                dt_bias=self.dt_bias,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
            )
        elif use_fla_triton and use_hybrid:
            core_attn_out = hybrid_chunk_kda(q, k, v, g, beta)
        elif use_fla_triton:
            # tw006 call site — gate was pre-computed by fused_kda_gate above,
            # only qk-l2norm is fused. Bit-identical to FLA's pre-fusion code
            # and to the explicit-gate Triton path; this is the configuration
            # that hit loss=4.7281 @ iter 500 vs FLA/8=4.7350.
            core_attn_out, _ = chunk_kda(
                q, k, v, g, beta,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out = _grad_checkpoint(
                _torch_chunk_kda_ckpt, q, k, v, g, beta,
                use_reentrant=False,
            )
        nvtx_range_pop(suffix="kda_attn")

        # --- Output gate (g_b expansion only — g_a is already inside in_proj) + gated norm ---
        nvtx_range_push(suffix="kda_out_gate")
        gate = self.g_b_proj(g_a_out)
        gate = gate.reshape(batch, seq_len, -1, self.head_dim)
        if self._use_fla_fused_norm_gated:
            # FusedRMSNormGated: RMSNorm(core_attn_out) * sigmoid(gate) in ONE
            # Triton kernel — no intermediate post-norm tensor, no fp32 upcast.
            # Matches fla/layers/kda.py exactly.
            norm_out = self.out_norm(core_attn_out, gate)
        else:
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
                "f_b_proj.weight": 0,
                "g_b_proj.weight": 0,
                "g_b_proj.bias": 0,
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

        # Split the combined in_proj into named chunks for checkpoint compatibility.
        # in_proj output layout (after GDN-style fusion):
        #   [ query | key | value | f_a | g_a | beta ]
        # First three are TP-sharded along output dim (qk/v split per rank),
        # f_a/g_a have dim 64 (head_v_dim, NOT TP-sharded since head_v_dim < tp_size
        # in some configs — we still split per-rank as standard column-parallel),
        # beta has num_heads dim which is TP-sharded the same way.
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
                self.head_dim,                    # f_a (bottleneck dim, not sharded)
                self.head_dim,                    # g_a (bottleneck dim, not sharded)
                self.num_heads // self.tp_size,   # beta
            ],
            ["query", "key", "value", "f_a", "g_a", "beta"],
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
