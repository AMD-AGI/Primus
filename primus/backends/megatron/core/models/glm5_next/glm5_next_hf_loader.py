###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Load a HuggingFace GLM-5.3 (``glm5_next_text``) checkpoint into Primus.

Online, per rank: every rank reads only the slices of the safetensors shards
its local parameters need, so any TP / EP / PP layout works without an offline
conversion step. Name mapping (HF prefix ``model.language_model.``)::

    embed_tokens / norm / lm_head      -> embedding.word_embeddings / decoder.final_layernorm / output_layer
    layers.i.input_layernorm           -> decoder.layers.*.input_layernorm
    layers.i.post_attention_layernorm  -> decoder.layers.*.pre_mlp_layernorm
    layers.i.hc_{attn,ffn}_{fn,base,scale}  (same name)
    KDA  self_attn.<name>              -> self_attention.<name>   (o_norm -> out_norm)
    DSA  self_attn.q_a_proj            -> self_attention.linear_q_down_proj
         q_a_layernorm / q_b_proj      -> q_layernorm / linear_q_up_proj
         kv_a_proj_with_mqa            -> linear_kv_down_proj
         kv_a_layernorm / kv_b_proj    -> kv_layernorm / linear_kv_up_proj
         o_proj                        -> linear_proj
         indexer.*                     -> indexer.*
    dense mlp gate/up/down             -> mlp.linear_fc1 (= [gate; up]) / mlp.linear_fc2
    mlp.gate.weight / e_score_correction_bias -> mlp.router.weight / mlp.router.expert_bias
    mlp.shared_experts.*               -> mlp.shared_experts.linear_fc{1,2}
    mlp.experts.e.*                    -> mlp.experts.linear_fc{1,2}.weight<local e>

TP slicing is inferred from the local vs full shape (the one axis that differs
is the partition axis); ``[gate; up]`` pairs are sliced per half.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from megatron.core import parallel_state

__all__ = ["load_glm5_next_hf_checkpoint"]

_HF_PREFIX = "model.language_model."

_DSA_MAP = {
    "linear_q_down_proj": "q_a_proj",
    "q_layernorm": "q_a_layernorm",
    "linear_q_up_proj": "q_b_proj",
    "linear_kv_down_proj": "kv_a_proj_with_mqa",
    "kv_layernorm": "kv_a_layernorm",
    "linear_kv_up_proj": "kv_b_proj",
    "linear_proj": "o_proj",
}


class _HFReader:
    def __init__(self, path: str) -> None:
        from safetensors import safe_open

        self._safe_open = safe_open
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}

    def get(self, key: str) -> torch.Tensor:
        fname = self.weight_map[key]
        h = self._handles.get(fname)
        if h is None:
            h = self._safe_open(os.path.join(self.path, fname), framework="pt", device="cpu")
            self._handles[fname] = h
        return h.get_tensor(key)


def _unwrap(module):
    while hasattr(module, "module"):
        module = module.module
    return module


def _slice_to(full: torch.Tensor, shape: torch.Size, rank: int, world: int, name: str) -> torch.Tensor:
    if tuple(full.shape) == tuple(shape):
        return full
    if full.numel() == int(torch.Size(shape).numel()):
        return full.reshape(shape)
    if full.dim() != len(shape):
        raise ValueError(f"{name}: cannot map HF shape {tuple(full.shape)} onto {tuple(shape)}")
    diff = [d for d in range(full.dim()) if full.shape[d] != shape[d]]
    if len(diff) != 1 or full.shape[diff[0]] != shape[diff[0]] * world:
        raise ValueError(f"{name}: HF shape {tuple(full.shape)} is not a {world}-way shard of {tuple(shape)}")
    d = diff[0]
    return full.narrow(d, rank * shape[d], shape[d])


def _layer_maps(
    layer, prefix: str, hf_layer: str, ep_rank: int, num_local_experts: int
) -> List[Tuple[str, object]]:
    """``[(local name suffix, HF spec)]``; spec is a key or a list of keys to concatenate on dim 0."""
    out: List[Tuple[str, object]] = []
    param_names = {pname for pname, _ in layer.named_parameters()}
    for pname, _ in layer.named_parameters():
        # Primus-Turbo grouped GEMM packs all local experts into `weights` and exposes
        # per-expert `weight{i}` views of it; the views are loaded below.
        m = re.fullmatch(r"(mlp\.experts\.linear_fc[12])\.weights", pname)
        if m is not None and f"{m.group(1)}.weight0" in param_names:
            continue
        hf: Optional[object] = None
        if pname.startswith("hc_"):
            hf = hf_layer + pname
        elif pname == "input_layernorm.weight":
            hf = hf_layer + "input_layernorm.weight"
        elif pname == "pre_mlp_layernorm.weight":
            hf = hf_layer + "post_attention_layernorm.weight"
        elif pname.startswith("self_attention."):
            sub = pname[len("self_attention.") :]
            attn = hf_layer + "self_attn."
            if layer.is_kda_layer:
                sub = sub.replace("out_norm.", "o_norm.")
                hf = attn + sub
            else:
                head, _, rest = sub.partition(".")
                if head == "indexer":
                    hf = attn + sub
                elif head in _DSA_MAP:
                    hf = attn + _DSA_MAP[head] + "." + rest
        elif pname.startswith("mlp."):
            sub = pname[len("mlp.") :]
            mlp = hf_layer + "mlp."
            if sub == "linear_fc1.weight":
                hf = [mlp + "gate_proj.weight", mlp + "up_proj.weight"]
            elif sub == "linear_fc2.weight":
                hf = mlp + "down_proj.weight"
            elif sub == "router.weight":
                hf = mlp + "gate.weight"
            elif sub == "shared_experts.linear_fc1.weight":
                hf = [mlp + "shared_experts.gate_proj.weight", mlp + "shared_experts.up_proj.weight"]
            elif sub == "shared_experts.linear_fc2.weight":
                hf = mlp + "shared_experts.down_proj.weight"
            else:
                m = re.fullmatch(r"experts\.linear_fc([12])\.weight(\d+)", sub)
                if m is not None:
                    e = ep_rank * num_local_experts + int(m.group(2))
                    ex = f"{mlp}experts.{e}."
                    hf = (
                        [ex + "gate_proj.weight", ex + "up_proj.weight"]
                        if m.group(1) == "1"
                        else ex + "down_proj.weight"
                    )
        if hf is None:
            raise KeyError(f"GLM5-Next HF loader: no mapping for parameter {prefix}{pname}")
        out.append((pname, hf))
    return out


@torch.no_grad()
def load_glm5_next_hf_checkpoint(model_chunks, hf_path: str) -> int:
    """Copy HF weights into every local parameter. Returns the number of tensors loaded."""
    reader = _HFReader(hf_path)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_world = parallel_state.get_tensor_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    etp_rank = parallel_state.get_expert_tensor_parallel_rank()
    etp_world = parallel_state.get_expert_tensor_parallel_world_size()

    loaded = 0
    for chunk in model_chunks:
        model = _unwrap(chunk)
        params = dict(model.named_parameters())

        def _copy(name: str, hf, *, expert: bool = False) -> None:
            nonlocal loaded
            p = params[name]
            rank, world = (etp_rank, etp_world) if expert else (tp_rank, tp_world)
            if isinstance(hf, list):
                half = list(p.shape)
                half[0] //= len(hf)
                pieces = [_slice_to(reader.get(k), torch.Size(half), rank, world, name) for k in hf]
                full = torch.cat(pieces, dim=0)
            else:
                full = _slice_to(reader.get(hf), p.shape, rank, world, name)
            p.data.copy_(full.to(device=p.device, dtype=p.dtype))
            loaded += 1

        if getattr(model, "pre_process", False):
            _copy_vocab(
                params["embedding.word_embeddings.weight"],
                reader.get(_HF_PREFIX + "embed_tokens.weight"),
                tp_rank,
            )
            loaded += 1
        if getattr(model, "post_process", False):
            if "output_layer.weight" in params:
                _copy_vocab(params["output_layer.weight"], reader.get("lm_head.weight"), tp_rank)
                loaded += 1
            _copy("decoder.final_layernorm.weight", _HF_PREFIX + "norm.weight")

        for local_idx, layer in enumerate(model.decoder.layers):
            prefix = f"decoder.layers.{local_idx}."
            hf_layer = f"{_HF_PREFIX}layers.{layer.layer_idx}."
            num_local_experts = 0
            if layer.is_moe_layer:
                num_local_experts = layer.mlp.num_local_experts
            for pname, hf in _layer_maps(layer, prefix, hf_layer, ep_rank, num_local_experts):
                _copy(prefix + pname, hf, expert=".experts.linear_fc" in pname)
            if layer.is_moe_layer and getattr(layer.mlp.router, "expert_bias", None) is not None:
                bias = reader.get(hf_layer + "mlp.gate.e_score_correction_bias").float()
                layer.mlp.router.expert_bias.data = bias.to(layer.mlp.router.expert_bias.device)
                loaded += 1
    return loaded


def _copy_vocab(param: torch.Tensor, full: torch.Tensor, tp_rank: int) -> None:
    rows = param.shape[0]
    start = tp_rank * rows
    shard = full[start : start + rows]
    param.data.zero_()
    param.data[: shard.shape[0]].copy_(shard.to(device=param.device, dtype=param.dtype))
