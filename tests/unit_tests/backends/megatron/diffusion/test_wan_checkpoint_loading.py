# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
``backbone_pretrained`` has to accept what ``convert_wan_hf_to_primus.py`` writes.

The converter emits a single ``.safetensors`` file, while ``backbone_subfolder``
defaults to ``transformer``. Appending the subfolder unconditionally made the
loader look for ``<file>.safetensors/transformer``, so the documented
convert-then-train flow could not load.
"""

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from primus.backends.megatron.core.models.diffusion.wan.model import (
    _load_backbone_checkpoint,
    resolve_backbone_checkpoint,
    resolve_expert_checkpoints,
)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"weight": torch.full((2, 2), value)}, str(path))


def test_a_converted_file_ignores_the_subfolder(tmp_path):
    ckpt = tmp_path / "primus_wan.safetensors"
    _save(ckpt, 1.0)

    assert resolve_backbone_checkpoint(str(ckpt), "transformer") == ckpt

    module = nn.Linear(2, 2, bias=False)
    _load_backbone_checkpoint(module, str(ckpt), subfolder="transformer")
    assert torch.equal(module.weight, torch.ones(2, 2))


def test_a_repo_layout_directory_selects_the_subfolder(tmp_path):
    _save(tmp_path / "transformer" / "model.safetensors", 1.0)
    _save(tmp_path / "transformer_2" / "model.safetensors", 2.0)

    assert resolve_backbone_checkpoint(str(tmp_path), "transformer_2") == tmp_path / "transformer_2"

    module = nn.Linear(2, 2, bias=False)
    _load_backbone_checkpoint(module, str(tmp_path), subfolder="transformer_2")
    assert torch.equal(module.weight, torch.full((2, 2), 2.0))


def test_a_single_transformer_directory_loads_as_is(tmp_path):
    _save(tmp_path / "model.safetensors", 3.0)

    assert resolve_backbone_checkpoint(str(tmp_path), "transformer") == tmp_path


def test_a_missing_checkpoint_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.safetensors"):
        _load_backbone_checkpoint(nn.Linear(2, 2), str(tmp_path / "nope.safetensors"), "transformer")


def test_dual_expert_rejects_one_checkpoint_for_both(tmp_path):
    ckpt = tmp_path / "primus_wan.safetensors"
    _save(ckpt, 1.0)

    with pytest.raises(ValueError, match="same weights"):
        resolve_expert_checkpoints(str(ckpt), "transformer", "transformer_2")


def test_dual_expert_resolves_each_subfolder(tmp_path):
    _save(tmp_path / "transformer" / "model.safetensors", 1.0)
    _save(tmp_path / "transformer_2" / "model.safetensors", 2.0)

    assert resolve_expert_checkpoints(str(tmp_path), "transformer", "transformer_2") == (
        tmp_path / "transformer",
        tmp_path / "transformer_2",
    )
