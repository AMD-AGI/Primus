###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (glm5_next) HF loader: checkpoint name-prefix detection."""

import json

import pytest
import torch
from safetensors.torch import save_file

from primus.backends.megatron.core.models.glm5_next.glm5_next_hf_loader import _HFReader


def _write_checkpoint(path, prefix):
    tensors = {
        f"{prefix}embed_tokens.weight": torch.arange(6.0).view(3, 2),
        f"{prefix}norm.weight": torch.ones(2),
        "lm_head.weight": torch.zeros(3, 2),
    }
    save_file(tensors, str(path / "model.safetensors"))
    index = {"weight_map": {k: "model.safetensors" for k in tensors}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


@pytest.mark.parametrize("prefix", ["model.language_model.", "model."])
def test_reader_detects_prefix(tmp_path, prefix):
    _write_checkpoint(tmp_path, prefix)
    reader = _HFReader(str(tmp_path))
    assert reader.prefix == prefix
    embed = reader.get(reader.prefix + "embed_tokens.weight")
    torch.testing.assert_close(embed, torch.arange(6.0).view(3, 2))


def test_reader_rejects_unknown_prefix(tmp_path):
    _write_checkpoint(tmp_path, "transformer.")
    with pytest.raises(KeyError, match="embed_tokens.weight"):
        _HFReader(str(tmp_path))
