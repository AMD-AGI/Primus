#!/usr/bin/env python3
"""One GLM-5.3 attention layer, packed THD, CP=1 and CP=2.

CP=1 (``torchrun --nproc_per_node=1``) runs each sample on its own and the packed
batch, checks they match, and writes a reference. CP=2 (``--nproc_per_node=2``)
loads that reference and checks the contiguous allgather-cp shard.

    torchrun --nproc_per_node=1 run_one_layer_thd_cp.py --kind dsa --save /tmp/dsa_cp1.pt
    torchrun --nproc_per_node=2 run_one_layer_thd_cp.py --kind dsa --check /tmp/dsa_cp1.pt
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from primus.backends.megatron.core.models.glm5_next.glm5_next_transformer_config import (
    Glm5NextTransformerConfig,
)
from primus.backends.megatron.core.transformer.glm5_next.dsa_attention import (
    Glm5NextDSAAttention,
    Glm5NextDSASubmodules,
)
from primus.backends.megatron.core.transformer.glm5_next.kda import Glm5NextKimiDeltaAttention
from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
    KimiDeltaAttentionSubmodules,
)


class _DupLinear(nn.Module):
    """Replicated linear with the Megatron ``(output, bias)`` contract."""

    def __init__(self, input_size, output_size, *, config=None, init_method=None, bias=False, **_kwargs):
        super().__init__()
        self.inner = nn.Linear(int(input_size), int(output_size), bias=bias)
        if init_method is not None:
            init_method(self.inner.weight)

    def forward(self, x):
        return self.inner(x), None


class _RMS(nn.Module):
    def __init__(self, config=None, hidden_size=None, eps=1e-5, **_kwargs):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(hidden_size)))

    def forward(self, x):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class _Packed:
    def __init__(self, cu: torch.Tensor, max_seqlen: int) -> None:
        self.cu_seqlens_q = cu
        self.cu_seqlens_kv = cu
        self.max_seqlen_q = max_seqlen
        self.max_seqlen_kv = max_seqlen
        self.qkv_format = "thd"


def _init_dist(cp_size: int) -> int:
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world = dist.get_world_size()
    if world != cp_size:
        raise RuntimeError(f"world size {world} != cp size {cp_size}")
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    model_parallel_cuda_manual_seed(1234)
    return dist.get_rank()


def _config(kind: str, cp_size: int) -> Glm5NextTransformerConfig:
    is_dsa = kind == "dsa"
    return Glm5NextTransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=4,
        ffn_hidden_size=256,
        kv_channels=32,
        layernorm_epsilon=1e-5,
        normalization="RMSNorm",
        pipeline_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        bf16=True,
        sequence_parallel=False,
        context_parallel_size=cp_size,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        linear_attention_freq=[0 if is_dsa else 1],
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=32,
        v_head_dim=32,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        index_kpool=4,
        dsa_backend="torch",
        kda_backend="fla",
        kda_gate_lower_bound=-5.0,
        kda_use_full_rank_gate=False,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        perform_initialization=True,
    )


def _build(kind: str, cp_size: int) -> torch.nn.Module:
    config = _config(kind, cp_size)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    if kind == "dsa":
        layer = Glm5NextDSAAttention(
            config,
            Glm5NextDSASubmodules(
                linear_q_down_proj=_DupLinear,
                q_layernorm=_RMS,
                linear_q_up_proj=ColumnParallelLinear,
                linear_kv_down_proj=_DupLinear,
                kv_layernorm=_RMS,
                linear_kv_up_proj=ColumnParallelLinear,
                linear_proj=RowParallelLinear,
                indexer_linear=_DupLinear,
            ),
            layer_number=1,
            pg_collection=pg,
        )
    else:
        layer = Glm5NextKimiDeltaAttention(
            config,
            KimiDeltaAttentionSubmodules(
                q_proj=ColumnParallelLinear,
                k_proj=ColumnParallelLinear,
                v_proj=ColumnParallelLinear,
                f_a_proj=ColumnParallelLinear,
                f_b_proj=ColumnParallelLinear,
                b_proj=ColumnParallelLinear,
                g_a_proj=ColumnParallelLinear,
                g_b_proj=ColumnParallelLinear,
                o_proj=RowParallelLinear,
            ),
            layer_number=1,
            pg_collection=pg,
        )
    return layer.cuda().to(torch.bfloat16)


def _pack(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, _Packed]:
    cu = [0]
    for seq in seqs:
        cu.append(cu[-1] + seq.shape[0])
    tokens = torch.cat(seqs, dim=0)
    bounds = torch.tensor(cu, dtype=torch.int32, device=tokens.device)
    return tokens, _Packed(bounds, max(s.shape[0] for s in seqs))


def _forward(layer, hidden, packed):
    hidden = hidden.detach().requires_grad_(True)
    out, _ = layer(hidden, packed_seq_params=packed)
    loss = out.float().square().sum()
    loss.backward()
    return out.detach(), hidden.grad.detach()


def _run_cp1(layer, kind: str, path: str) -> None:
    torch.manual_seed(0)
    lengths = (24, 40)
    seqs = [torch.randn(n, 1, 128, device="cuda", dtype=torch.bfloat16) for n in lengths]
    packed_h, packed = _pack(seqs)
    separate = []
    separate_grad = []
    for seq in seqs:
        out, grad = _forward(layer, seq, None)
        separate.append(out)
        separate_grad.append(grad)
    layer.zero_grad(set_to_none=True)
    packed_out, packed_grad = _forward(layer, packed_h, packed)
    ref_out = torch.cat(separate, dim=0)
    ref_grad = torch.cat(separate_grad, dim=0)
    out_err = (packed_out.float() - ref_out.float()).abs().max().item()
    grad_err = (packed_grad.float() - ref_grad.float()).abs().max().item()
    print(f"[{kind} cp1] packed vs separate  out {out_err:.3e}  grad {grad_err:.3e}", flush=True)
    if out_err > 1e-2 or grad_err > 2e-2:
        raise SystemExit(f"{kind} CP=1 packed forward does not match per-sample forwards")
    if dist.get_rank() == 0:
        torch.save(
            {
                "state": {
                    k: v.detach().cpu()
                    for k, v in layer.state_dict().items()
                    if torch.is_tensor(v)
                },
                "hidden": packed_h.detach().cpu(),
                "out": packed_out.cpu(),
                "grad": packed_grad.cpu(),
                "cu": packed.cu_seqlens_q.cpu(),
            },
            path,
        )
        print(f"[{kind} cp1] wrote {path}", flush=True)


def _run_cp2(layer, kind: str, path: str) -> None:
    ref = torch.load(path, map_location="cpu", weights_only=False)
    layer.load_state_dict({k: v.cuda() for k, v in ref["state"].items()}, strict=False)
    hidden = ref["hidden"].cuda()
    cu = ref["cu"].cuda()
    rank = dist.get_rank()
    cp = dist.get_world_size()
    local_len = hidden.shape[0] // cp
    shard = hidden[rank * local_len : (rank + 1) * local_len]
    packed = _Packed(cu, int((cu[1:] - cu[:-1]).max().item()))
    layer.zero_grad(set_to_none=True)
    out, grad = _forward(layer, shard, packed)
    out_g = [torch.empty_like(out) for _ in range(cp)]
    grad_g = [torch.empty_like(grad) for _ in range(cp)]
    dist.all_gather(out_g, out.contiguous())
    dist.all_gather(grad_g, grad.contiguous())
    if rank == 0:
        got_out = torch.cat(out_g, dim=0).float()
        got_grad = torch.cat(grad_g, dim=0).float()
        out_err = (got_out - ref["out"].float().cuda()).abs().max().item()
        grad_err = (got_grad - ref["grad"].float().cuda()).abs().max().item()
        print(f"[{kind} cp2] vs cp1  out {out_err:.3e}  grad {grad_err:.3e}", flush=True)
        if out_err > 2e-2 or grad_err > 5e-2:
            raise SystemExit(f"{kind} CP=2 does not match CP=1")
        print(f"[{kind} cp2] ok", flush=True)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("dsa", "kda"), required=True)
    parser.add_argument("--save", default="")
    parser.add_argument("--check", default="")
    args = parser.parse_args()
    if bool(args.save) == bool(args.check):
        raise SystemExit("pass exactly one of --save or --check")
    cp_size = 1 if args.save else 2
    rank = _init_dist(cp_size)
    layer = _build(args.kind, cp_size)
    if args.save:
        if rank == 0:
            print(f"[{args.kind}] built {type(layer).__name__}", flush=True)
        _run_cp1(layer, args.kind, args.save)
    else:
        _run_cp2(layer, args.kind, args.check)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
