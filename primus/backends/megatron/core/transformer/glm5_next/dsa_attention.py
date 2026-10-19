###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 DSA layer: NoPE absorbed MLA + kpool lightning indexer.

Transcribes SGLang ``Glm5NextForConditionalGeneration``'s DSA attention and the
Miles Megatron port (``miles_plugins/models/glm5_next/dsa.py``)::

    q_lora = q_a_layernorm(q_a_proj(x))                       # [t, q_lora_rank]
    q      = q_b_proj(q_lora).view(t, H, qk_head_dim)          # no rope part
    latent = kv_a_layernorm(kv_a_proj_with_mqa(x))             # [t, kv_lora_rank]
    w_kc, w_vc = kv_b_proj.weight.view(H, qk + v, kv_lora_rank).split
    query  = einsum(q, w_kc)                                   # [t, H, kv_lora_rank]
    o      = sparse_softmax(query . latent[idx] * qk_head_dim**-0.5) @ latent[idx]
    out    = o_proj(einsum(o, w_vc))                           # [t, hidden]

``idx`` comes from :mod:`.kpool_indexer`; for sequences no longer than
``index_topk`` it is plain causal attention. HF parameter names map 1:1 onto the
submodules below (``q_a_proj`` -> ``linear_q_down_proj`` ...); see
``glm5_next_hf_loader.py``.

Tensor parallelism shards the MLA heads (``q_b_proj`` / ``kv_b_proj`` column,
``o_proj`` row), the way upstream ``MultiLatentAttention`` does. The
down-projections and the indexer are replicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from torch import Tensor, nn

from primus.backends.megatron.core.transformer.glm5_next.fp32_params import (
    KeepFp32ParamsMixin,
)
from primus.backends.megatron.core.transformer.glm5_next.kpool_indexer import (
    build_pooled_keys,
    causal_all_indices,
    kpool_scores,
    kpool_scores_torch,
    kpool_select,
)
from primus.backends.megatron.core.transformer.glm5_next.sparse_mla import (
    sparse_mla,
    sparse_mla_torch,
)

__all__ = ["Glm5NextDSAAttention", "Glm5NextDSASubmodules", "Glm5NextKpoolIndexer"]


def _param_device() -> Optional[torch.device]:
    return torch.cuda.current_device() if torch.cuda.is_available() else None


def _dsa_cp_state():
    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        return 1, 0, None
    size = int(parallel_state.get_context_parallel_world_size() or 1)
    if size <= 1:
        return 1, 0, None
    return size, int(parallel_state.get_context_parallel_rank()), parallel_state.get_context_parallel_group()


class _AllGatherSequence(torch.autograd.Function):
    """Concatenate equal sequence shards. Backward reduce-scatters the token axis."""

    @staticmethod
    def forward(ctx, x: Tensor, group) -> Tensor:
        import torch.distributed as dist

        ctx.group = group
        world = dist.get_world_size(group)
        shards = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(shards, x.contiguous(), group=group)
        return torch.cat(shards, dim=0)

    @staticmethod
    def backward(ctx, grad: Tensor):
        import torch.distributed as dist

        world = dist.get_world_size(ctx.group)
        chunks = list(grad.contiguous().chunk(world, dim=0))
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None


def _gather_sequence(x: Tensor, group, *, grad: bool) -> Tensor:
    if group is None:
        return x
    if grad:
        return _AllGatherSequence.apply(x, group)
    import torch.distributed as dist

    shards = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, x.contiguous(), group=group)
    return torch.cat(shards, dim=0)


def _cu_list(cu: Tensor) -> list:
    return [int(v) for v in cu.detach().to("cpu").tolist()]


def _forward_dsa_segments(module, hidden_states: Tensor, packed_seq_params):
    if hidden_states.shape[1] != 1:
        raise NotImplementedError("Packed DSA expects batch size 1 (THD).")
    cu = _cu_list(packed_seq_params.cu_seqlens_q)
    if cu[-1] != hidden_states.shape[0]:
        raise RuntimeError(f"cu_seqlens ends at {cu[-1]} but the token axis is {hidden_states.shape[0]}.")
    pieces = []
    bias = None
    for start, end in zip(cu[:-1], cu[1:]):
        if end <= start:
            continue
        out, bias = module.forward(hidden_states[start:end], packed_seq_params=None)
        pieces.append(out)
    if not pieces:
        raise RuntimeError("Packed DSA received no tokens.")
    return torch.cat(pieces, dim=0), bias


def _packed_indices(indexer, x: Tensor, q_lora: Tensor, cu: Tensor) -> Tensor:
    """Per-sample kpool indices into the packed token axis. ``x`` is ``[S, B, H]``."""
    bounds = _cu_list(cu)
    chunks = []
    width = 0
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end <= start:
            continue
        idx = indexer(x[start:end], q_lora[start:end], 1, end - start)
        idx = torch.where(idx >= 0, idx + start, idx)
        chunks.append(idx)
        width = max(width, idx.shape[1])
    total = bounds[-1]
    out = torch.full((total, width), -1, dtype=torch.int32, device=x.device)
    cursor = 0
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end <= start:
            continue
        idx = chunks[cursor]
        cursor += 1
        out[start:end, : idx.shape[1]] = idx
    return out


@dataclass
class Glm5NextDSASubmodules:
    linear_q_down_proj: Union[ModuleSpec, type] = IdentityOp
    q_layernorm: Union[ModuleSpec, type] = IdentityOp
    linear_q_up_proj: Union[ModuleSpec, type] = IdentityOp
    linear_kv_down_proj: Union[ModuleSpec, type] = IdentityOp
    kv_layernorm: Union[ModuleSpec, type] = IdentityOp
    linear_kv_up_proj: Union[ModuleSpec, type] = IdentityOp
    linear_proj: Union[ModuleSpec, type] = IdentityOp
    indexer_linear: Union[ModuleSpec, type] = IdentityOp


class Glm5NextKpoolIndexer(KeepFp32ParamsMixin, MegatronModule):
    """Replicated kpool indexer. Every parameter is frozen (no indexer loss yet).

    HF names (``self_attn.indexer.*``): ``wq_b``, ``wk``, ``k_norm.{weight,bias}``,
    ``weights_proj``, ``index_kpool_compress_gate``, ``index_kpool_compress_ape``.
    """

    def __init__(self, config, linear_spec) -> None:
        super().__init__(config=config)
        self.n_heads = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.kpool = int(config.index_kpool)
        self.topk = int(config.index_topk)
        self.backend = str(config.dsa_backend)
        hidden = int(config.hidden_size)

        def _dup(inp, out, name):
            return build_module(
                linear_spec,
                inp,
                out,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=name,
                skip_weight_param_allocation=False,
                tp_group=None,
                parallel_mode="duplicated",
            )

        self.wq_b = _dup(int(config.q_lora_rank), self.n_heads * self.head_dim, "idx_wq_b")
        self.wk = _dup(hidden, self.head_dim, "idx_wk")
        self.k_norm = nn.LayerNorm(
            self.head_dim, eps=float(config.index_k_norm_eps), device=_param_device(), dtype=torch.float32
        )
        self.weights_proj = nn.Linear(
            hidden, self.n_heads, bias=False, device=_param_device(), dtype=torch.float32
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.head_dim, hidden, device=_param_device(), dtype=config.params_dtype)
        )
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.kpool, self.head_dim, device=_param_device(), dtype=torch.float32)
        )
        self._keep_fp32_enabled = bool(getattr(config, "glm5_keep_fp32_params", True))
        self._fp32_param_names = ("index_kpool_compress_ape",)
        for p in self.parameters():
            p.requires_grad_(False)

    def _apply(self, fn, *args, **kwargs):  # type: ignore[override]
        out = super()._apply(fn, *args, **kwargs)
        if self._keep_fp32_enabled:
            for m in (self.k_norm, self.weights_proj):
                for p in m.parameters():
                    if p.dtype != torch.float32:
                        p.data = p.data.float()
        return out

    @torch.no_grad()
    def forward(self, x: Tensor, q_lora: Tensor, B: int, S: int) -> Tensor:
        """Token selection ``[B * S, width]`` int32.

        Args:
            x: ``[S, B, hidden]`` full-sequence layer input.
            q_lora: ``[S, B, q_lora_rank]`` normalised q latent.
        """
        if S <= self.topk:
            return causal_all_indices(B, S, x.device)
        xb = x.transpose(0, 1)  # [B, S, hidden]
        q, _ = self.wq_b(q_lora.transpose(0, 1))
        q = q.view(B, S, self.n_heads, self.head_dim)
        k, _ = self.wk(xb)
        k = self.k_norm(k.float()).to(x.dtype)
        gate = F.linear(xb, self.index_kpool_compress_gate.to(xb.dtype))
        w = F.linear(xb.float(), self.weights_proj.weight.float())
        w = w * (self.n_heads**-0.5 * self.head_dim**-0.5)
        pooled = build_pooled_keys(k, gate, self.index_kpool_compress_ape, self.kpool)
        score_fn = kpool_scores if self.backend == "triton" else kpool_scores_torch
        scores = score_fn(q, pooled, w, self.kpool)
        return kpool_select(scores, B, S, self.topk, self.kpool)


class Glm5NextDSAAttention(MegatronModule):
    """GLM-5.3 DSA self-attention. Input / output ``[s, b, h]``."""

    def __init__(
        self,
        config,
        submodules: Glm5NextDSASubmodules,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        attn_mask_type=None,
        **_unused,
    ) -> None:
        super().__init__(config=config)
        assert pg_collection is not None, "pg_collection must be provided for Glm5NextDSAAttention"
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.pg_collection = pg_collection
        self.tp_size = pg_collection.tp.size()
        self.num_heads = int(config.num_attention_heads)
        assert self.num_heads % self.tp_size == 0
        self.num_heads_local = self.num_heads // self.tp_size
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_head_dim = int(config.qk_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.softmax_scale = self.qk_head_dim**-0.5
        self.backend = str(config.dsa_backend)
        hidden = int(config.hidden_size)
        eps = float(config.layernorm_epsilon)

        def _dup(spec, inp, out, name):
            return build_module(
                spec,
                inp,
                out,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=name,
                skip_weight_param_allocation=False,
                tp_group=None,
                parallel_mode="duplicated",
            )

        def _col(spec, inp, out, name):
            return build_module(
                spec,
                inp,
                out,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=name,
                tp_group=pg_collection.tp,
            )

        self.linear_q_down_proj = _dup(submodules.linear_q_down_proj, hidden, self.q_lora_rank, "q_down_proj")
        self.q_layernorm = build_module(
            submodules.q_layernorm, config=config, hidden_size=self.q_lora_rank, eps=eps
        )
        self.linear_q_up_proj = _col(
            submodules.linear_q_up_proj, self.q_lora_rank, self.num_heads * self.qk_head_dim, "q_up_proj"
        )
        self.linear_kv_down_proj = _dup(
            submodules.linear_kv_down_proj, hidden, self.kv_lora_rank, "kv_down_proj"
        )
        self.kv_layernorm = build_module(
            submodules.kv_layernorm, config=config, hidden_size=self.kv_lora_rank, eps=eps
        )
        # Only its weight is used (absorbed into q and o); it is never called.
        self.linear_kv_up_proj = _col(
            submodules.linear_kv_up_proj,
            self.kv_lora_rank,
            self.num_heads * (self.qk_head_dim + self.v_head_dim),
            "kv_up_proj",
        )
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.num_heads * self.v_head_dim,
            hidden,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=pg_collection.tp,
        )
        self.indexer = Glm5NextKpoolIndexer(config, submodules.indexer_linear)

    def _gather_seq(self, t: Tensor) -> Tensor:
        if self.config.sequence_parallel and self.tp_size > 1:
            return gather_from_sequence_parallel_region(t, group=self.pg_collection.tp)
        return t

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if inference_context is not None:
            raise NotImplementedError("Glm5NextDSAAttention has no inference cache.")

        cp_size, cp_rank, cp_group = _dsa_cp_state()
        if packed_seq_params is not None and cp_size == 1:
            return _forward_dsa_segments(self, hidden_states, packed_seq_params)
        if cp_size > 1 and packed_seq_params is None:
            raise NotImplementedError(
                "DSA context parallel requires packed THD (--qkv-format thd --allgather-cp)."
            )

        # ---- low-rank q / kv (on the sequence shard under SP) ----------
        q_c, _ = self.linear_q_down_proj(hidden_states)
        q_c = self.q_layernorm(q_c)
        q, _ = self.linear_q_up_proj(q_c)  # [S, B, Hl * qk] (column-parallel gathers under SP)
        S, B = q.shape[0], q.shape[1]
        packed_cp = packed_seq_params is not None and cp_size > 1
        if packed_cp:
            if B != 1:
                raise NotImplementedError("Packed DSA expects batch size 1 (THD).")
            global_len = int(packed_seq_params.cu_seqlens_q[-1].item())
            if S * cp_size != global_len:
                raise RuntimeError(
                    "DSA packed CP expects contiguous allgather-cp shards: "
                    f"local {S} * cp {cp_size} != global {global_len}."
                )

        kv_c, _ = self.linear_kv_down_proj(hidden_states)
        kv_c = self._gather_seq(kv_c)
        latent = self.kv_layernorm(kv_c)  # [S, B, kv_lora]
        if not self.config.sequence_parallel and self.tp_size > 1:
            # Each rank's heads contribute a partial gradient to the replicated latent.
            latent = copy_to_tensor_model_parallel_region(latent, group=self.pg_collection.tp)

        # ---- indexer (no grad; full sequence) --------------------------
        with torch.no_grad():
            x_full = hidden_states.detach()
            q_lora_full = q_c.detach()
            if self.config.sequence_parallel and self.tp_size > 1:
                x_full = gather_from_sequence_parallel_region(x_full, group=self.pg_collection.tp)
                q_lora_full = gather_from_sequence_parallel_region(q_lora_full, group=self.pg_collection.tp)
            if packed_cp:
                x_full = _gather_sequence(x_full, cp_group, grad=False)
                q_lora_full = _gather_sequence(q_lora_full, cp_group, grad=False)
                indices = _packed_indices(self.indexer, x_full, q_lora_full, packed_seq_params.cu_seqlens_q)
                indices = indices[cp_rank * S : (cp_rank + 1) * S]
            else:
                indices = self.indexer(x_full, q_lora_full, B, S)

        if packed_cp:
            latent = _gather_sequence(latent, cp_group, grad=True)

        # ---- absorbed sparse MLA ---------------------------------------
        w = self.linear_kv_up_proj.weight.view(
            self.num_heads_local, self.qk_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_kc, w_vc = w.split([self.qk_head_dim, self.v_head_dim], dim=1)
        q = (
            q.view(S, B, self.num_heads_local, self.qk_head_dim)
            .transpose(0, 1)
            .reshape(B * S, self.num_heads_local, self.qk_head_dim)
        )
        query = torch.einsum("thd,hdm->thm", q, w_kc)  # [T, Hl, kv_lora]
        key_len = latent.shape[0] * latent.shape[1]
        key = latent.transpose(0, 1).reshape(key_len, self.kv_lora_rank)
        attn_fn = sparse_mla if self.backend == "triton" else sparse_mla_torch
        o = attn_fn(query.contiguous(), key.contiguous(), indices, self.softmax_scale)
        o = torch.einsum("thm,hdm->thd", o, w_vc)  # [T, Hl, v]
        o = o.reshape(B, S, self.num_heads_local * self.v_head_dim).transpose(0, 1).contiguous()

        out, bias = self.linear_proj(o)
        return out, bias
