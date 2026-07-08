###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from pathlib import Path

import pytest
import torch

from primus.backends.diffusion.models.registrations.wan import (
    _convert_diffusers_wan_dit_state_dict,
    _resolve_dit_checkpoint_path,
)


def test_resolve_diffusers_transformer_subfolder(tmp_path: Path):
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer_2").mkdir()

    assert _resolve_dit_checkpoint_path(str(tmp_path), "transformer") == str(tmp_path / "transformer")
    assert _resolve_dit_checkpoint_path(str(tmp_path), "transformer_2") == str(tmp_path / "transformer_2")


def test_resolve_diffusers_transformer_rejects_unknown_stage(tmp_path: Path):
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="active_transformer"):
        _resolve_dit_checkpoint_path(str(tmp_path), "middle_noise")


def test_convert_diffusers_wan_dit_keys_to_primus_keys():
    state = {
        "condition_embedder.text_embedder.linear_1.weight": torch.empty(1),
        "condition_embedder.text_embedder.linear_2.bias": torch.empty(1),
        "condition_embedder.time_embedder.linear_1.weight": torch.empty(1),
        "condition_embedder.time_proj.bias": torch.empty(1),
        "patch_embedding.weight": torch.empty(1),
        "scale_shift_table": torch.empty(1),
        "proj_out.weight": torch.empty(1),
        "blocks.0.scale_shift_table": torch.empty(1),
        "blocks.0.attn1.to_q.weight": torch.empty(1),
        "blocks.0.attn1.to_out.0.bias": torch.empty(1),
        "blocks.0.attn2.to_k.weight": torch.empty(1),
        "blocks.0.attn2.norm_q.weight": torch.empty(1),
        "blocks.0.ffn.net.0.proj.weight": torch.empty(1),
        "blocks.0.ffn.net.2.bias": torch.empty(1),
        "blocks.0.norm2.weight": torch.empty(1),
    }

    converted = _convert_diffusers_wan_dit_state_dict(state)

    assert "text_embedding.0.weight" in converted
    assert "text_embedding.2.bias" in converted
    assert "time_embedding.0.weight" in converted
    assert "time_projection.1.bias" in converted
    assert "patch_embedding.weight" in converted
    assert "head.modulation" in converted
    assert "head.head.weight" in converted
    assert "blocks.0.modulation" in converted
    assert "blocks.0.self_attn.q.weight" in converted
    assert "blocks.0.self_attn.o.bias" in converted
    assert "blocks.0.cross_attn.k.weight" in converted
    assert "blocks.0.cross_attn.norm_q.weight" in converted
    assert "blocks.0.ffn.0.weight" in converted
    assert "blocks.0.ffn.2.bias" in converted
    assert "blocks.0.norm3.weight" in converted
