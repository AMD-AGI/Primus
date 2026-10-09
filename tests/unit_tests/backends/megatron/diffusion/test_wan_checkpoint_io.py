# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
A directory of torch checkpoint shards loads as one state dict.

The converter runs with ``strict=False`` by default, so a shard that is
silently skipped shows up only as missing weights in the trained model.
"""

import torch

from primus.backends.megatron.core.models.diffusion.wan.checkpoint_converter import (
    _load_state_dict,
)


def test_every_torch_shard_in_a_directory_is_loaded(tmp_path):
    torch.save({"a.weight": torch.ones(2)}, tmp_path / "pytorch_model-00001-of-00002.bin")
    torch.save({"b.weight": torch.zeros(3)}, tmp_path / "pytorch_model-00002-of-00002.bin")

    sd = _load_state_dict(tmp_path)

    assert set(sd) == {"a.weight", "b.weight"}
    assert torch.equal(sd["b.weight"], torch.zeros(3))


def test_a_shard_wrapped_in_state_dict_is_unwrapped(tmp_path):
    torch.save({"state_dict": {"a.weight": torch.ones(2)}}, tmp_path / "shard0.pt")
    torch.save({"b.weight": torch.ones(2)}, tmp_path / "shard1.pt")

    assert set(_load_state_dict(tmp_path)) == {"a.weight", "b.weight"}
