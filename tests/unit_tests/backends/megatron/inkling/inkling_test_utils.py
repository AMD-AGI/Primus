###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared helpers for the Inkling tests: presets, fixtures, synthetic checkpoints."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

REPO = Path(__file__).resolve().parents[5]
PRESETS = REPO / "primus" / "configs" / "models" / "megatron"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONVERTER = REPO / "tools" / "checkpoint_conversion" / "convert_inkling_hf_megatron.py"

RELEASE_VOCAB = 201024
_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


def preset(name: str) -> str:
    return str(PRESETS / f"{name}.yaml")


def release_config(model: str = "inkling_small") -> dict:
    return json.loads((FIXTURES / f"{model}_config.json").read_text())


def release_shapes() -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """Every tensor of the released Inkling-Small checkpoint: name -> (dtype, shape)."""
    raw = json.loads((FIXTURES / "inkling_small_checkpoint_shapes.json").read_text())["tensors"]
    return {k: (v[0], tuple(v[1])) for k, v in raw.items()}


def release_meta_tensors(prefix: str = "model.llm.") -> Dict[str, torch.Tensor]:
    return {
        k: torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
        for k, (dtype, shape) in release_shapes().items()
        if k.startswith(prefix)
    }


def numel(shapes) -> int:
    return sum(math.prod(s) for s in shapes)


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_inkling_hf_megatron", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release_schema(config, vocab: int = RELEASE_VOCAB) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
    """Released-checkpoint names and shapes implied by a Megatron config.

    Written from the checkpoint layout, independently of the importer; the
    full-size case is checked against the real headers.
    """
    h = config.hidden_size
    k = config.short_conv_kernel_size
    bf, f32 = torch.bfloat16, torch.float32
    out = {
        "model.llm.embed.weight": ((vocab, h), bf),
        "model.llm.embed_norm.weight": ((h,), bf),
        "model.llm.norm.weight": ((h,), bf),
        "model.llm.unembed.weight": ((vocab, h), bf),
    }
    e, s, inter, dense = (
        config.num_moe_experts,
        config.moe_num_shared_experts,
        config.moe_ffn_hidden_size,
        config.ffn_hidden_size,
    )
    for i in range(config.num_layers):
        geo = config.attention_geometry(i)
        heads, groups, hd = geo["num_attention_heads"], geo["num_query_groups"], geo["kv_channels"]
        p = f"model.llm.layers.{i}."
        out.update(
            {
                p + "attn_norm.weight": ((h,), bf),
                p + "attn.wq_du.weight": ((heads * hd, h), bf),
                p + "attn.wk_dv.weight": ((groups * hd, h), bf),
                p + "attn.wv_dv.weight": ((groups * hd, h), bf),
                p + "attn.wr_du.weight": ((heads * config.d_rel, h), bf),
                p + "attn.wo_ud.weight": ((h, heads * hd), bf),
                p + "attn.q_norm.weight": ((hd,), bf),
                p + "attn.k_norm.weight": ((hd,), bf),
                p + "attn.k_sconv.weight": ((groups * hd, 1, k), bf),
                p + "attn.v_sconv.weight": ((groups * hd, 1, k), bf),
                p + "attn.rel_logits_proj.proj": ((config.d_rel, geo["rel_extent"]), bf),
                p + "attn_sconv.weight": ((h, 1, k), bf),
                p + "mlp_norm.weight": ((h,), bf),
                p + "mlp_sconv.weight": ((h, 1, k), bf),
            }
        )
        if config.is_moe_layer(i):
            out.update(
                {
                    p + "mlp.gate.weight": ((e + s, h), bf),
                    p + "mlp.gate.bias": ((e,), f32),
                    p + "mlp.gate.global_scale": ((1,), f32),
                    p + "mlp.experts.w13_weight": ((e, 2 * inter, h), bf),
                    p + "mlp.experts.w2_weight": ((e, h, inter), bf),
                    p + "mlp.shared_experts.shared_w13_weight": ((s, 2 * inter, h), bf),
                    p + "mlp.shared_experts.shared_w2_weight": ((s, h, inter), bf),
                }
            )
        else:
            out.update(
                {
                    p + "mlp.w13_dn.weight": ((2 * dense, h), bf),
                    p + "mlp.w2_md.weight": ((h, dense), bf),
                    p + "mlp.global_scale": ((1,), bf),
                }
            )
    return out


def release_text_config(config, vocab: int) -> dict:
    """A released-style ``text_config`` for a Megatron config (inverse of ``megatron_config_from_release``)."""
    pattern = config.moe_layer_pattern()
    dense = pattern.index(1) if 1 in pattern else len(pattern)
    assert pattern == [0] * dense + [1] * (len(pattern) - dense)
    return dict(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        vocab_size=vocab,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_query_groups,
        head_dim=config.kv_channels,
        swa_num_attention_heads=config.swa_num_attention_heads,
        swa_num_key_value_heads=config.swa_num_query_groups,
        swa_head_dim=config.swa_kv_channels,
        sliding_window_size=config.sliding_window_size,
        local_layer_ids=list(config.local_layer_ids),
        d_rel=config.d_rel,
        rel_extent=config.rel_extent,
        log_scaling_n_floor=config.log_scaling_n_floor,
        log_scaling_alpha=config.log_scaling_alpha,
        rms_norm_eps=config.layernorm_epsilon,
        use_embed_norm=True,
        use_sconv=True,
        sconv_kernel_size=config.short_conv_kernel_size,
        dense_mlp_idx=dense,
        unpadded_vocab_size=config.unpadded_vocab_size,
        logits_mup_width_multiplier=config.logits_mup_width_multiplier,
        dense_intermediate_size=config.ffn_hidden_size,
        intermediate_size=config.moe_ffn_hidden_size,
        n_routed_experts=config.num_moe_experts,
        num_experts_per_tok=config.moe_router_topk,
        n_shared_experts=config.moe_num_shared_experts,
        shared_expert_sink=True,
        route_scale=config.moe_router_topk_scaling_factor,
    )


def random_release_state(config, vocab: int, seed: int = 0, dtype: Optional[torch.dtype] = None):
    """A synthetic released checkpoint with distinct random values in every tensor."""
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=g, dtype=torch.float32).to(dtype or dt)
        for name, (shape, dt) in release_schema(config, vocab).items()
    }
