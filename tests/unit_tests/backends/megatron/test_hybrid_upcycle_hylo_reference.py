###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus upcycling must match HyLo's from-teacher weight init.

The reference is ``HybridModelWrapper`` on AMD-Hybrid-Models ``feat/HyLo``:
``DeepseekV3Attention.re_init_q`` / ``re_init_kv`` and
``_copy_llama_attn_to_gdn``. The published 4MLA12GDN config supplies the
widths. When ``meta-llama/Llama-3.2-1B-Instruct`` is on disk, the same check
runs on those tensors.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from primus.backends.megatron.checkpoint.hybrid_upcycle import (
    UpcycleLayout,
    hylo_reinit_kv,
    hylo_reinit_q,
    upcycle_state_dict,
)

HYLO_ROOT = Path(os.environ.get("HYLO_ROOT", "/tmp/AMD-Hybrid-Models/HyLo"))
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", "/tmp/hf-models/llama"))
MLA_LAYERS = (1, 5, 10, 14)

# Llama-3.2-1B-Instruct and amd/HyLo-Llama-4MLA12GDN-8K-SFT.
HIDDEN = 2048
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 64
Q_RANK = 1344
KV_RANK = 128
NOPE = 32
ROPE = 32
V_HEAD = 64
GDN_HEADS = 6
GDN_HEAD_DIM = 256
GDN_VALUE_DIM = GDN_HEAD_DIM * 2


class _HyLoMLA:
    """Projection holder sized like HyLo ``DeepseekV3Attention`` for 4MLA12GDN."""

    def __init__(self):
        self.num_heads = HEADS
        self.num_kv_heads = KV_HEADS
        self.q_lora_rank = Q_RANK
        self.kv_lora_rank = KV_RANK
        self.qk_rope_head_dim = ROPE
        self.qk_nope_head_dim = NOPE
        self.v_head_dim = V_HEAD
        self.q_a_proj = nn.Linear(HIDDEN, Q_RANK, bias=False)
        self.q_b_proj = nn.Linear(Q_RANK, HEADS * (NOPE + ROPE), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN, KV_RANK + ROPE, bias=False)
        self.kv_b_proj = nn.Linear(KV_RANK, KV_HEADS * (NOPE + V_HEAD), bias=False)


def _import_hylo():
    wrapper = HYLO_ROOT / "hybrid" / "hybrid_wrapper.py"
    layer = HYLO_ROOT / "hybrid" / "mla" / "hybrid_mla_layer.py"
    if not wrapper.is_file() or not layer.is_file():
        pytest.skip(f"HyLo training code not found at {HYLO_ROOT}")
    from einops import rearrange

    layer_src = layer.read_text()
    start = layer_src.index("    def re_init_q")
    end = layer_src.index("    def forward", start)
    methods = textwrap.dedent(layer_src[start:end])
    method_ns = {"torch": torch, "rearrange": rearrange}
    exec(methods, method_ns)
    _HyLoMLA.re_init_q = method_ns["re_init_q"]
    _HyLoMLA.re_init_kv = method_ns["re_init_kv"]

    source = wrapper.read_text()
    start = source.index("def _copy_llama_attn_to_gdn")
    end = source.index("\nHYBRID_CONFIG_NAME")
    namespace = {"torch": torch}
    exec(source[start:end], namespace)
    return _HyLoMLA, namespace["_copy_llama_attn_to_gdn"]


def _layout() -> UpcycleLayout:
    return UpcycleLayout(
        num_attention_heads=HEADS,
        num_query_groups=KV_HEADS,
        head_dim=HEAD_DIM,
        q_lora_rank=Q_RANK,
        kv_lora_rank=KV_RANK,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_HEAD,
        linear_type="gdn",
        gdn_num_key_heads=GDN_HEADS,
        gdn_key_head_dim=GDN_HEAD_DIM,
        gdn_num_value_heads=GDN_HEADS,
        gdn_value_head_dim=GDN_VALUE_DIM,
    )


class _Attn:
    def __init__(self, q, k, v, o):
        self.q_proj = nn.Linear(HIDDEN, q.shape[0], bias=False)
        self.k_proj = nn.Linear(HIDDEN, k.shape[0], bias=False)
        self.v_proj = nn.Linear(HIDDEN, v.shape[0], bias=False)
        self.o_proj = nn.Linear(o.shape[1], o.shape[0], bias=False)
        self.q_proj.weight.data.copy_(q)
        self.k_proj.weight.data.copy_(k)
        self.v_proj.weight.data.copy_(v)
        self.o_proj.weight.data.copy_(o)


class _GDN:
    def __init__(self):
        self.q_proj = nn.Linear(HIDDEN, GDN_HEADS * GDN_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, GDN_HEADS * GDN_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, GDN_HEADS * GDN_VALUE_DIM, bias=False)
        self.o_proj = nn.Linear(GDN_HEADS * GDN_VALUE_DIM, HIDDEN, bias=False)


def _teacher(seed: int):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(HEADS * HEAD_DIM, HIDDEN, generator=gen)
    k = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN, generator=gen)
    v = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN, generator=gen)
    o = torch.randn(HIDDEN, HIDDEN, generator=gen)
    return q, k, v, o


def _assert_close(name, actual, expected):
    assert actual.shape == expected.shape, f"{name} {tuple(actual.shape)} != {tuple(expected.shape)}"
    if not torch.equal(actual, expected):
        delta = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{name} differs from HyLo training init, max_abs={delta}")


def _check_layer(q, k, v, o, attention_cls, copy_gdn):
    attn = attention_cls()
    attn.re_init_q(q, HEAD_DIM, q.dtype)
    attn.re_init_kv(k, v, KV_HEADS, HEAD_DIM, q.dtype)
    q_down, q_up = hylo_reinit_q(q, Q_RANK, HEADS, HEAD_DIM, NOPE, ROPE)
    kv_down, kv_up = hylo_reinit_kv(k, v, KV_HEADS, HEAD_DIM, KV_HEADS, KV_RANK, NOPE, ROPE, V_HEAD)
    _assert_close("q_a_proj", q_down, attn.q_a_proj.weight.data)
    _assert_close("q_b_proj", q_up, attn.q_b_proj.weight.data)
    _assert_close("kv_a_proj_with_mqa", kv_down, attn.kv_a_proj_with_mqa.weight.data)
    _assert_close("kv_b_proj", kv_up, attn.kv_b_proj.weight.data)

    gdn = _GDN()
    copy_gdn(_Attn(q, k, v, o), gdn, q.dtype, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS)
    key_dim = GDN_HEADS * GDN_HEAD_DIM
    value_dim = GDN_HEADS * GDN_VALUE_DIM
    in_rows = key_dim * 2 + value_dim * 2 + GDN_HEADS * 2
    dense = {
        "embedding.word_embeddings.weight": torch.zeros(1, 1),
        "decoder.final_layernorm.weight": torch.zeros(HIDDEN),
        "decoder.layers.0.input_layernorm.weight": torch.ones(HIDDEN),
        "decoder.layers.0.self_attention.linear_q_proj.weight": q,
        "decoder.layers.0.self_attention.linear_k_proj.weight": k,
        "decoder.layers.0.self_attention.linear_v_proj.weight": v,
        "decoder.layers.0.self_attention.linear_proj.weight": o,
        "decoder.layers.0.pre_mlp_layernorm.weight": torch.ones(HIDDEN),
        "decoder.layers.0.mlp.linear_fc1.weight": torch.zeros(2, HIDDEN),
        "decoder.layers.0.mlp.linear_fc2.weight": torch.zeros(HIDDEN, 2),
    }
    hybrid = {
        "embedding.word_embeddings.weight": torch.zeros(1, 1),
        "decoder.final_norm.weight": torch.zeros(HIDDEN),
        "decoder.layers.0.input_layernorm.weight": torch.zeros(HIDDEN),
        "decoder.layers.0.mixer.in_proj.weight": torch.full((in_rows, HIDDEN), -3.0),
        "decoder.layers.0.mixer.out_proj.weight": torch.full((HIDDEN, value_dim), -4.0),
        "decoder.layers.0.mixer.A_log": torch.full((GDN_HEADS,), -2.0),
        "decoder.layers.1.pre_mlp_layernorm.weight": torch.zeros(HIDDEN),
        "decoder.layers.1.mlp.linear_fc1.weight": torch.zeros(2, HIDDEN),
        "decoder.layers.1.mlp.linear_fc2.weight": torch.zeros(HIDDEN, 2),
    }
    upcycled, _report = upcycle_state_dict(dense, hybrid, "M-", _layout())
    copied = upcycled["decoder.layers.0.mixer.in_proj.weight"]
    init_in = hybrid["decoder.layers.0.mixer.in_proj.weight"]
    _assert_close("gdn.q_proj", copied[:key_dim], gdn.q_proj.weight.data)
    _assert_close("gdn.k_proj", copied[key_dim : key_dim * 2], gdn.k_proj.weight.data)
    v_rows = min(HEADS * HEAD_DIM, value_dim)
    v_start = key_dim * 2
    _assert_close("gdn.v_proj", copied[v_start : v_start + v_rows], gdn.v_proj.weight.data[:v_rows])
    assert torch.equal(
        copied[v_start + v_rows : v_start + value_dim], init_in[v_start + v_rows : v_start + value_dim]
    )
    tail = copied[v_start + value_dim :]
    assert torch.equal(tail, init_in[v_start + value_dim :])
    copied_cols = min(o.shape[1], value_dim)
    out = upcycled["decoder.layers.0.mixer.out_proj.weight"]
    _assert_close("gdn.o_proj", out[:, :copied_cols], gdn.o_proj.weight.data[:, :copied_cols])
    if out.shape[1] > copied_cols:
        assert torch.equal(
            out[:, copied_cols:], hybrid["decoder.layers.0.mixer.out_proj.weight"][:, copied_cols:]
        )


def test_primus_matches_hylo_training_init_on_4mla12gdn_shapes():
    attention_cls, copy_gdn = _import_hylo()
    _check_layer(*_teacher(0), attention_cls, copy_gdn)


def _llama_layer_weights(layer: int):
    weight = LLAMA_DIR / "model.safetensors"
    if not weight.is_file():
        pytest.skip(f"Llama-3.2-1B-Instruct weights not found at {weight}")
    from safetensors import safe_open

    prefix = f"model.layers.{layer}.self_attn"
    with safe_open(weight, framework="pt") as handle:
        return tuple(
            handle.get_tensor(f"{prefix}.{name}.weight") for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )


def test_primus_matches_hylo_training_init_on_llama32_1b_instruct():
    attention_cls, copy_gdn = _import_hylo()
    for layer in (0, *MLA_LAYERS):
        _check_layer(*_llama_layer_weights(layer), attention_cls, copy_gdn)
