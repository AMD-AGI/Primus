###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Released-checkpoint <-> Megatron weight mapping (no model build)."""

from __future__ import annotations

import pytest
import torch

from primus.backends.megatron.core.models.inkling.inkling_builders import (
    inkling_config_from_model_yaml,
)
from primus.backends.megatron.core.models.inkling.inkling_hf_mapping import (
    InklingGeometry,
    deinterleave_glu,
    fuse_qkv,
    interleave_glu,
    megatron_to_release,
    release_text_keys,
    release_to_megatron,
    split_qkv,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import (
    numel,
    preset,
    random_release_state,
    release_config,
    release_meta_tensors,
    release_schema,
    release_shapes,
)


def _transformers_interleave(t: torch.Tensor, dim: int) -> torch.Tensor:
    """``transformers.core_model_loading.Interleave(dim).convert`` (load direction)."""
    shape = list(t.shape)
    shape[dim : dim + 1] = [shape[dim] // 2, 2]
    return t.reshape(shape).transpose(dim, dim + 1).reshape(t.shape)


@pytest.mark.parametrize("dim", [0, 1])
def test_deinterleave_matches_transformers(dim):
    w = torch.randn(3, 8, 5) if dim == 1 else torch.randn(8, 5)
    gate, up = deinterleave_glu(w, dim=dim)
    # transformers de-interleaves, then takes the first half as gate (Chunk / gate_up_proj.chunk(2))
    ref_gate, ref_up = _transformers_interleave(w, dim).chunk(2, dim=dim)
    torch.testing.assert_close(gate, ref_gate, rtol=0, atol=0)
    torch.testing.assert_close(up, ref_up, rtol=0, atol=0)
    torch.testing.assert_close(interleave_glu(gate, up, dim=dim), w, rtol=0, atol=0)


def test_fused_qkv_layout_groups_heads():
    heads, groups, hd, hidden = 4, 2, 3, 5
    tag = lambda kind, n: torch.arange(n, dtype=torch.float32)[:, None] + kind * 1000 + torch.zeros(1, hidden)
    q, k, v = tag(1, heads * hd), tag(2, groups * hd), tag(3, groups * hd)
    fused = fuse_qkv(q, k, v, heads=heads, groups=groups, head_dim=hd)
    per_group = (heads // groups + 2) * hd
    assert fused.shape == (groups * per_group, hidden)
    # second KV group: q heads 2-3, then k/v head 1
    g1 = fused[per_group:, 0].tolist()
    assert g1[: 2 * hd] == [1000 + x for x in range(2 * hd, 4 * hd)]
    assert g1[2 * hd : 3 * hd] == [2000 + x for x in range(hd, 2 * hd)]
    assert g1[3 * hd :] == [3000 + x for x in range(hd, 2 * hd)]
    for a, b in zip(split_qkv(fused, heads=heads, groups=groups, head_dim=hd), (q, k, v)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_linear_qkv_matches_megatron_muon_split():
    """``linear_qkv`` is Megatron's 3-part grouped layout, which Muon's ``qkv_split_shapes`` assumes."""
    heads, groups, hd, hidden = 4, 2, 3, 5
    q, k, v = torch.randn(heads * hd, hidden), torch.randn(groups * hd, hidden), torch.randn(groups * hd, hidden)
    fused = fuse_qkv(q, k, v, heads=heads, groups=groups, head_dim=hd)
    split_shapes = [heads // groups * hd, hd, hd]
    mq, mk, mv = fused.view(groups, sum(split_shapes), hidden).split(split_shapes, dim=1)
    torch.testing.assert_close(mq.reshape(-1, hidden), q, rtol=0, atol=0)
    torch.testing.assert_close(mk.reshape(-1, hidden), k, rtol=0, atol=0)
    torch.testing.assert_close(mv.reshape(-1, hidden), v, rtol=0, atol=0)


def test_release_schema_matches_checkpoint_headers():
    config = inkling_config_from_model_yaml(preset("inkling_small"))
    schema = {k: shape for k, (shape, _) in release_schema(config).items()}
    real = {k: shape for k, (_, shape) in release_shapes().items() if k.startswith("model.llm.")}
    assert schema == real


def test_full_checkpoint_maps_one_to_one():
    """Every text tensor of Inkling-Small is consumed exactly once and no element is lost."""
    geo = InklingGeometry.from_release_config(release_config()["text_config"])
    release = release_meta_tensors()
    out = release_to_megatron(release, geo, strict=True)
    assert numel(t.shape for t in out.values()) == numel(t.shape for t in release.values())
    # 4 model-level + 12 per layer + 3 per dense MLP + per MoE: 4 router, 2 x 256 experts, 2 shared
    assert len(out) == 4 + 42 * 12 + 2 * 3 + 40 * (4 + 2 * 256 + 2)
    skipped = set(release_shapes()) - release_text_keys(release_shapes())
    assert {k.split(".")[1] for k in skipped} == {"mtp", "visual", "audio"}


def test_geometry_from_release_matches_preset():
    config = inkling_config_from_model_yaml(preset("inkling_small"))
    assert InklingGeometry.from_release_config(release_config()["text_config"]) == (
        InklingGeometry.from_megatron_config(config)
    )


def test_release_megatron_round_trip_values():
    config = inkling_config_from_model_yaml(preset("inkling_small_tiny"))
    geo = InklingGeometry.from_megatron_config(config)
    release = random_release_state(config, vocab=1000)
    back = megatron_to_release(release_to_megatron(release, geo), geo)
    assert back.keys() == release.keys()
    for k, v in release.items():
        torch.testing.assert_close(back[k], v, rtol=0, atol=0, msg=k)


def test_vocab_resize():
    config = inkling_config_from_model_yaml(preset("inkling_small_tiny"))
    geo = InklingGeometry.from_megatron_config(config)
    release = random_release_state(config, vocab=1000)
    trimmed = release_to_megatron(release, geo, target_vocab_size=900)
    padded = release_to_megatron(release, geo, target_vocab_size=1024)
    torch.testing.assert_close(trimmed["output_layer.weight"], release["model.llm.unembed.weight"][:900])
    assert padded["embedding.word_embeddings.weight"].shape[0] == 1024
    assert padded["embedding.word_embeddings.weight"][1000:].abs().sum() == 0


def test_strict_rejects_unmapped_text_tensor():
    config = inkling_config_from_model_yaml(preset("inkling_small_tiny"))
    release = random_release_state(config, vocab=64)
    release["model.llm.layers.0.attn.extra.weight"] = torch.zeros(1)
    release["model.mtp.layers.0.embed_norm.weight"] = torch.zeros(1)  # skipped by design
    with pytest.raises(KeyError, match="not mapped"):
        release_to_megatron(release, InklingGeometry.from_megatron_config(config))
