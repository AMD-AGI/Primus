###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling weight mapping: released checkpoint <-> Megatron, plus transformers.

Three naming schemes meet here:

``release``
    The safetensors on the Hub (``thinkingmachines/Inkling-Small``):
    ``model.llm.layers.{i}.attn.wq_du.weight`` ... Gate/up pairs are stored
    **interleaved** row by row (``g0, u0, g1, u1, ...``), which is what
    transformers' ``Interleave`` conversion op undoes on load.
``megatron``
    :class:`InklingModel`'s single-rank (TP=PP=EP=1) state dict.
``transformers``
    ``InklingForCausalLM.state_dict()`` from the official implementation,
    used by the reference harness. :func:`transformers_to_release` replays
    transformers' conversion mapping in reverse, so the harness exercises the
    same importer path as a real checkpoint.

Only the text backbone is mapped. ``model.mtp.*`` (multi-token prediction,
P4), ``model.visual.*`` and ``model.audio.*`` (P5) are left out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Set,
    Tuple,
)

import torch

__all__ = [
    "InklingGeometry",
    "RELEASE_TEXT_PREFIX",
    "deinterleave_glu",
    "interleave_glu",
    "fuse_qkv",
    "split_qkv",
    "iter_release_to_megatron",
    "release_to_megatron",
    "megatron_to_release",
    "transformers_to_release",
    "release_text_keys",
    "megatron_config_from_release",
    "module_tensor_shapes",
]

RELEASE_TEXT_PREFIX = "model.llm."


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InklingGeometry:
    """The shape facts the mapping needs, from either side's config."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    swa_num_attention_heads: int
    swa_num_query_groups: int
    swa_kv_channels: int
    d_rel: int
    local_layer_ids: FrozenSet[int]
    moe_layer_ids: FrozenSet[int]
    num_experts: int
    num_shared_experts: int

    def attention(self, layer_idx: int) -> Tuple[int, int, int]:
        """``(heads, kv_groups, head_dim)`` of one layer."""
        if layer_idx in self.local_layer_ids:
            return self.swa_num_attention_heads, self.swa_num_query_groups, self.swa_kv_channels
        return self.num_attention_heads, self.num_query_groups, self.kv_channels

    def is_moe(self, layer_idx: int) -> bool:
        return layer_idx in self.moe_layer_ids

    @classmethod
    def from_megatron_config(cls, config) -> "InklingGeometry":
        moe = config.moe_layer_pattern()
        return cls(
            num_layers=int(config.num_layers),
            hidden_size=int(config.hidden_size),
            num_attention_heads=int(config.num_attention_heads),
            num_query_groups=int(config.num_query_groups),
            kv_channels=int(config.kv_channels),
            swa_num_attention_heads=int(config.swa_num_attention_heads),
            swa_num_query_groups=int(config.swa_num_query_groups),
            swa_kv_channels=int(config.swa_kv_channels),
            d_rel=int(config.d_rel),
            local_layer_ids=frozenset(int(i) for i in config.local_layer_ids),
            moe_layer_ids=frozenset(i for i, m in enumerate(moe) if m),
            num_experts=int(config.num_moe_experts or 0),
            num_shared_experts=int(config.moe_num_shared_experts or 0),
        )

    @classmethod
    def from_release_config(cls, text_config: Mapping) -> "InklingGeometry":
        kw = megatron_config_from_release(text_config)
        n = kw["num_layers"]
        dense = int(text_config.get("dense_mlp_idx", 0))
        return cls(
            num_layers=n,
            hidden_size=kw["hidden_size"],
            num_attention_heads=kw["num_attention_heads"],
            num_query_groups=kw["num_query_groups"],
            kv_channels=kw["kv_channels"],
            swa_num_attention_heads=kw["swa_num_attention_heads"],
            swa_num_query_groups=kw["swa_num_query_groups"],
            swa_kv_channels=kw["swa_kv_channels"],
            d_rel=kw["d_rel"],
            local_layer_ids=frozenset(kw["local_layer_ids"]),
            moe_layer_ids=frozenset(range(dense, n)),
            num_experts=kw["num_experts"],
            num_shared_experts=kw["moe_num_shared_experts"],
        )


def megatron_config_from_release(text_config: Mapping) -> Dict:
    """Primus/Megatron model-config values implied by a released ``text_config``.

    Keys are the YAML / :class:`InklingTransformerConfig` names. Released
    names: ``intermediate_size`` is one routed expert and
    ``dense_intermediate_size`` the dense MLP; ``dense_mlp_idx`` dense layers
    come first.
    """
    c = dict(text_config)
    n = int(c["num_hidden_layers"])
    local = c.get("local_layer_ids")
    if local is None:
        local = [i for i in range(n) if (i + 1) % 6]
    dense = int(c.get("dense_mlp_idx", 0))
    for key, want in (
        ("use_sconv", True),
        ("shared_expert_sink", True),
        ("gate_activation", "sigmoid"),
        ("norm_after_topk", True),
        ("use_global_scale", True),
        ("use_gate_bias", True),
        ("q_bias", False),
        ("o_bias", False),
        ("final_logit_softcapping", None),
    ):
        if key in c and c[key] != want:
            raise ValueError(f"Unsupported Inkling variant: {key}={c[key]!r} (expected {want!r})")
    return dict(
        num_layers=n,
        hidden_size=int(c["hidden_size"]),
        num_attention_heads=int(c["num_attention_heads"]),
        num_query_groups=int(c["num_key_value_heads"]),
        kv_channels=int(c["head_dim"]),
        swa_num_attention_heads=int(c.get("swa_num_attention_heads", c["num_attention_heads"])),
        swa_num_query_groups=int(c.get("swa_num_key_value_heads", c["num_key_value_heads"])),
        swa_kv_channels=int(c.get("swa_head_dim", c["head_dim"])),
        sliding_window_size=int(c["sliding_window_size"]),
        local_layer_ids=sorted(int(i) for i in local),
        d_rel=int(c["d_rel"]),
        rel_extent=int(c["rel_extent"]),
        log_scaling_n_floor=c.get("log_scaling_n_floor"),
        log_scaling_alpha=float(c.get("log_scaling_alpha", 0.1)),
        short_conv_kernel_size=int(c.get("sconv_kernel_size", c.get("conv_kernel_size", 4))),
        use_embed_norm=bool(c.get("use_embed_norm", True)),
        logits_mup_width_multiplier=float(c["logits_mup_width_multiplier"]),
        unpadded_vocab_size=c.get("unpadded_vocab_size"),
        layernorm_epsilon=float(c["rms_norm_eps"]),
        ffn_hidden_size=int(c.get("dense_intermediate_size", c["intermediate_size"])),
        moe_ffn_hidden_size=int(c["intermediate_size"]),
        num_experts=int(c["n_routed_experts"]),
        moe_router_topk=int(c["num_experts_per_tok"]),
        moe_num_shared_experts=int(c["n_shared_experts"]),
        moe_shared_expert_intermediate_size=int(c["n_shared_experts"]) * int(c["intermediate_size"]),
        moe_router_topk_scaling_factor=float(c["route_scale"]),
        moe_layer_freq=f"([0]*{dense}+[1]*{n - dense})",
        vocab_size=int(c["vocab_size"]),
    )


# ---------------------------------------------------------------------------
# Tensor layouts
# ---------------------------------------------------------------------------


def deinterleave_glu(w: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a row-interleaved ``[g0, u0, g1, u1, ...]`` tensor into ``(gate, up)``."""
    dim = dim % w.dim()
    shape = list(w.shape)
    if shape[dim] % 2:
        raise ValueError(f"odd GLU dim {shape[dim]}")
    v = w.reshape(shape[:dim] + [shape[dim] // 2, 2] + shape[dim + 1 :])
    return v.select(dim + 1, 0), v.select(dim + 1, 1)


def interleave_glu(gate: torch.Tensor, up: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of :func:`deinterleave_glu`."""
    dim = dim % gate.dim()
    shape = list(gate.shape)
    shape[dim] *= 2
    return torch.stack([gate, up], dim=dim + 1).reshape(shape)


def fuse_qkv(q, k, v, *, heads: int, groups: int, head_dim: int) -> torch.Tensor:
    """Group-interleave q/k/v rows into Megatron's ``linear_qkv`` layout (see ``attention.py``)."""
    hidden = q.shape[-1]
    per = heads // groups
    parts = (
        q.reshape(groups, per * head_dim, hidden),
        k.reshape(groups, head_dim, hidden),
        v.reshape(groups, head_dim, hidden),
    )
    return torch.cat(parts, dim=1).reshape(-1, hidden)


def split_qkv(fused, *, heads: int, groups: int, head_dim: int):
    """Inverse of :func:`fuse_qkv`: ``(q, k, v)`` in checkpoint layout."""
    hidden = fused.shape[-1]
    per = heads // groups
    sizes = [per * head_dim, head_dim, head_dim]
    q, k, v = fused.reshape(groups, sum(sizes), hidden).split(sizes, dim=1)
    return (
        q.reshape(heads * head_dim, hidden),
        k.reshape(groups * head_dim, hidden),
        v.reshape(groups * head_dim, hidden),
    )


def module_tensor_shapes(module: torch.nn.Module) -> Dict[str, Tuple[int, ...]]:
    """Parameter and persistent-buffer shapes by state-dict name (works on meta)."""
    out = {name: tuple(p.shape) for name, p in module.named_parameters()}
    for mod_name, mod in module.named_modules():
        for buf_name, buf in mod.named_buffers(recurse=False):
            if buf is None or buf_name in mod._non_persistent_buffers_set:
                continue
            out[f"{mod_name}.{buf_name}" if mod_name else buf_name] = tuple(buf.shape)
    return out


def _resize_rows(w: torch.Tensor, rows: Optional[int]) -> torch.Tensor:
    if rows is None or rows == w.shape[0]:
        return w
    if rows < w.shape[0]:
        return w[:rows]
    pad = w.new_zeros((rows - w.shape[0],) + tuple(w.shape[1:]))
    return torch.cat([w, pad], dim=0)


# ---------------------------------------------------------------------------
# release -> megatron
# ---------------------------------------------------------------------------


def release_text_keys(keys: Iterable[str]) -> Set[str]:
    """The text-backbone keys of a released checkpoint (MTP, vision and audio excluded)."""
    return {k for k in keys if k.startswith(RELEASE_TEXT_PREFIX)}


def _layer_release_to_megatron(
    get: Callable[[str], torch.Tensor], geo: InklingGeometry, i: int
) -> Iterator[Tuple[str, torch.Tensor]]:
    src = f"model.llm.layers.{i}."
    dst = f"decoder.layers.{i}."
    heads, groups, head_dim = geo.attention(i)

    yield dst + "input_layernorm.weight", get(src + "attn_norm.weight")
    yield dst + "self_attention.linear_qkv.weight", fuse_qkv(
        get(src + "attn.wq_du.weight"),
        get(src + "attn.wk_dv.weight"),
        get(src + "attn.wv_dv.weight"),
        heads=heads,
        groups=groups,
        head_dim=head_dim,
    )
    yield dst + "self_attention.linear_rel.weight", get(src + "attn.wr_du.weight")
    yield dst + "self_attention.linear_proj.weight", get(src + "attn.wo_ud.weight")
    yield dst + "self_attention.q_layernorm.weight", get(src + "attn.q_norm.weight")
    yield dst + "self_attention.k_layernorm.weight", get(src + "attn.k_norm.weight")
    yield dst + "self_attention.k_sconv.weight", get(src + "attn.k_sconv.weight")
    yield dst + "self_attention.v_sconv.weight", get(src + "attn.v_sconv.weight")
    yield dst + "self_attention.rel_proj", get(src + "attn.rel_logits_proj.proj")
    yield dst + "attn_sconv.weight", get(src + "attn_sconv.weight")
    yield dst + "pre_mlp_layernorm.weight", get(src + "mlp_norm.weight")
    yield dst + "mlp_sconv.weight", get(src + "mlp_sconv.weight")

    if not geo.is_moe(i):
        gate, up = deinterleave_glu(get(src + "mlp.w13_dn.weight"), dim=0)
        yield dst + "mlp.linear_fc1.weight", torch.cat([gate, up], dim=0)
        yield dst + "mlp.linear_fc2.weight", get(src + "mlp.w2_md.weight")
        yield dst + "mlp.global_scale", get(src + "mlp.global_scale")
        return

    e, s = geo.num_experts, geo.num_shared_experts
    router = get(src + "mlp.gate.weight")
    if router.shape[0] != e + s:
        raise ValueError(f"layer {i}: router has {router.shape[0]} rows, expected {e} routed + {s} shared")
    yield dst + "mlp.router.weight", router[:e]
    yield dst + "mlp.router.shared_weight", router[e:]
    yield dst + "mlp.router.global_scale", get(src + "mlp.gate.global_scale")
    yield dst + "mlp.router.expert_bias", get(src + "mlp.gate.bias")

    w13 = get(src + "mlp.experts.w13_weight")
    w2 = get(src + "mlp.experts.w2_weight")
    gate, up = deinterleave_glu(w13, dim=1)
    for x in range(e):
        yield dst + f"mlp.experts.linear_fc1.weight{x}", torch.cat([gate[x], up[x]], dim=0)
        yield dst + f"mlp.experts.linear_fc2.weight{x}", w2[x]

    sgate, sup = deinterleave_glu(get(src + "mlp.shared_experts.shared_w13_weight"), dim=1)
    hidden = sgate.shape[-1]
    yield dst + "mlp.shared_experts.linear_fc1.weight", torch.cat(
        [sgate.reshape(-1, hidden), sup.reshape(-1, hidden)], dim=0
    )
    sw2 = get(src + "mlp.shared_experts.shared_w2_weight")  # [s, hidden, inter]
    yield dst + "mlp.shared_experts.linear_fc2.weight", sw2.permute(1, 0, 2).reshape(sw2.shape[1], -1)


def iter_release_to_megatron(
    get: Callable[[str], torch.Tensor],
    geo: InklingGeometry,
    *,
    target_vocab_size: Optional[int] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(megatron_key, tensor)`` one layer at a time.

    ``get`` reads one released tensor by name, so a caller can stream from
    safetensors. ``target_vocab_size`` trims or zero-pads the embedding and
    LM-head rows to Megatron's padded vocabulary.
    """
    yield "embedding.word_embeddings.weight", _resize_rows(get("model.llm.embed.weight"), target_vocab_size)
    yield "embed_norm.weight", get("model.llm.embed_norm.weight")
    for i in range(geo.num_layers):
        yield from _layer_release_to_megatron(get, geo, i)
    yield "decoder.final_layernorm.weight", get("model.llm.norm.weight")
    yield "output_layer.weight", _resize_rows(get("model.llm.unembed.weight"), target_vocab_size)


def release_to_megatron(
    release: Mapping[str, torch.Tensor],
    geo: InklingGeometry,
    *,
    target_vocab_size: Optional[int] = None,
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert an in-memory released state dict.

    With ``strict`` every text-backbone key must be consumed exactly once.
    """
    used: Set[str] = set()

    def get(name: str) -> torch.Tensor:
        if name in used:
            raise KeyError(f"{name} read twice")
        used.add(name)
        return release[name]

    out = dict(iter_release_to_megatron(get, geo, target_vocab_size=target_vocab_size))
    if strict:
        left = release_text_keys(release.keys()) - used
        if left:
            raise KeyError(f"{len(left)} released text keys not mapped, e.g. {sorted(left)[:5]}")
    return out


# ---------------------------------------------------------------------------
# megatron -> release (round-trip checks; full HF export is deferred)
# ---------------------------------------------------------------------------


def megatron_to_release(
    sd: Mapping[str, torch.Tensor],
    geo: InklingGeometry,
    *,
    release_vocab_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "model.llm.embed.weight": _resize_rows(sd["embedding.word_embeddings.weight"], release_vocab_size),
        "model.llm.embed_norm.weight": sd["embed_norm.weight"],
        "model.llm.norm.weight": sd["decoder.final_layernorm.weight"],
        "model.llm.unembed.weight": _resize_rows(sd["output_layer.weight"], release_vocab_size),
    }
    for i in range(geo.num_layers):
        src = f"decoder.layers.{i}."
        dst = f"model.llm.layers.{i}."
        heads, groups, head_dim = geo.attention(i)
        q, k, v = split_qkv(
            sd[src + "self_attention.linear_qkv.weight"],
            heads=heads,
            groups=groups,
            head_dim=head_dim,
        )
        out.update(
            {
                dst + "attn_norm.weight": sd[src + "input_layernorm.weight"],
                dst + "attn.wq_du.weight": q,
                dst + "attn.wk_dv.weight": k,
                dst + "attn.wv_dv.weight": v,
                dst + "attn.wr_du.weight": sd[src + "self_attention.linear_rel.weight"],
                dst + "attn.wo_ud.weight": sd[src + "self_attention.linear_proj.weight"],
                dst + "attn.q_norm.weight": sd[src + "self_attention.q_layernorm.weight"],
                dst + "attn.k_norm.weight": sd[src + "self_attention.k_layernorm.weight"],
                dst + "attn.k_sconv.weight": sd[src + "self_attention.k_sconv.weight"],
                dst + "attn.v_sconv.weight": sd[src + "self_attention.v_sconv.weight"],
                dst + "attn.rel_logits_proj.proj": sd[src + "self_attention.rel_proj"],
                dst + "attn_sconv.weight": sd[src + "attn_sconv.weight"],
                dst + "mlp_norm.weight": sd[src + "pre_mlp_layernorm.weight"],
                dst + "mlp_sconv.weight": sd[src + "mlp_sconv.weight"],
            }
        )
        if not geo.is_moe(i):
            gate, up = sd[src + "mlp.linear_fc1.weight"].chunk(2, dim=0)
            out[dst + "mlp.w13_dn.weight"] = interleave_glu(gate, up, dim=0)
            out[dst + "mlp.w2_md.weight"] = sd[src + "mlp.linear_fc2.weight"]
            out[dst + "mlp.global_scale"] = sd[src + "mlp.global_scale"]
            continue
        e, s = geo.num_experts, geo.num_shared_experts
        out[dst + "mlp.gate.weight"] = torch.cat(
            [sd[src + "mlp.router.weight"], sd[src + "mlp.router.shared_weight"]], dim=0
        )
        out[dst + "mlp.gate.global_scale"] = sd[src + "mlp.router.global_scale"]
        out[dst + "mlp.gate.bias"] = sd[src + "mlp.router.expert_bias"]
        fc1 = torch.stack([sd[src + f"mlp.experts.linear_fc1.weight{x}"] for x in range(e)])
        gate, up = fc1.chunk(2, dim=1)
        out[dst + "mlp.experts.w13_weight"] = interleave_glu(gate, up, dim=1)
        out[dst + "mlp.experts.w2_weight"] = torch.stack(
            [sd[src + f"mlp.experts.linear_fc2.weight{x}"] for x in range(e)]
        )
        sfc1 = sd[src + "mlp.shared_experts.linear_fc1.weight"]
        hidden = sfc1.shape[-1]
        sgate, sup = (t.reshape(s, -1, hidden) for t in sfc1.chunk(2, dim=0))
        out[dst + "mlp.shared_experts.shared_w13_weight"] = interleave_glu(sgate, sup, dim=1)
        sfc2 = sd[src + "mlp.shared_experts.linear_fc2.weight"]  # [hidden, s * inter]
        out[dst + "mlp.shared_experts.shared_w2_weight"] = sfc2.reshape(sfc2.shape[0], s, -1).permute(1, 0, 2)
    return out


# ---------------------------------------------------------------------------
# transformers -> release (reference harness)
# ---------------------------------------------------------------------------

_TRANSFORMERS_RENAMES = (
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "mlp_norm.weight"),
    ("self_attn.q_proj.weight", "attn.wq_du.weight"),
    ("self_attn.k_proj.weight", "attn.wk_dv.weight"),
    ("self_attn.v_proj.weight", "attn.wv_dv.weight"),
    ("self_attn.r_proj.weight", "attn.wr_du.weight"),
    ("self_attn.o_proj.weight", "attn.wo_ud.weight"),
    ("self_attn.q_norm.weight", "attn.q_norm.weight"),
    ("self_attn.k_norm.weight", "attn.k_norm.weight"),
    ("self_attn.k_sconv.conv1d.weight", "attn.k_sconv.weight"),
    ("self_attn.v_sconv.conv1d.weight", "attn.v_sconv.weight"),
    ("self_attn.rel_logits_proj.proj", "attn.rel_logits_proj.proj"),
    ("attn_sconv.conv1d.weight", "attn_sconv.weight"),
    ("mlp_sconv.conv1d.weight", "mlp_sconv.weight"),
    ("mlp.down_proj.weight", "mlp.w2_md.weight"),
    ("mlp.global_scale", "mlp.global_scale"),
    ("mlp.gate.weight", "mlp.gate.weight"),
    ("mlp.gate.global_scale", "mlp.gate.global_scale"),
    ("mlp.gate.e_score_correction_bias", "mlp.gate.bias"),
    ("mlp.experts.down_proj", "mlp.experts.w2_weight"),
    ("mlp.shared_experts.down_proj", "mlp.shared_experts.shared_w2_weight"),
)


def transformers_to_release(sd: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """``InklingForCausalLM.state_dict()`` -> released checkpoint names and layouts."""
    out: Dict[str, torch.Tensor] = {
        "model.llm.embed.weight": sd["model.embed_tokens.weight"],
        "model.llm.embed_norm.weight": sd["model.embed_tokens.embed_norm.weight"],
        "model.llm.norm.weight": sd["model.norm.weight"],
        "model.llm.unembed.weight": sd["lm_head.weight"],
    }
    layers = sorted({int(m.group(1)) for k in sd if (m := re.match(r"^model\.layers\.(\d+)\.", k))})
    for i in layers:
        src = f"model.layers.{i}."
        dst = f"model.llm.layers.{i}."
        for a, b in _TRANSFORMERS_RENAMES:
            if src + a in sd:
                out[dst + b] = sd[src + a]
        if src + "mlp.gate_proj.weight" in sd:
            out[dst + "mlp.w13_dn.weight"] = interleave_glu(
                sd[src + "mlp.gate_proj.weight"], sd[src + "mlp.up_proj.weight"], dim=0
            )
        if src + "mlp.experts.gate_up_proj" in sd:
            gate, up = sd[src + "mlp.experts.gate_up_proj"].chunk(2, dim=1)
            out[dst + "mlp.experts.w13_weight"] = interleave_glu(gate, up, dim=1)
        if src + "mlp.shared_experts.gate_proj" in sd:
            out[dst + "mlp.shared_experts.shared_w13_weight"] = interleave_glu(
                sd[src + "mlp.shared_experts.gate_proj"], sd[src + "mlp.shared_experts.up_proj"], dim=1
            )
    return out
