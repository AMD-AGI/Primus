###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Legacy SFT checkpoints reach the model through per-tensor copies, not the mmap storage."""

import torch

from primus.backends.megatron.megatron_sft_trainer import _stage_state_dict_to_device


def test_tensors_are_copied_and_other_entries_kept(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"w": torch.arange(6.0), "b": torch.ones(2), "_extra_state": None, "e": torch.empty(0)}, path)
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)

    staged = _stage_state_dict_to_device(sd, "cpu")

    assert torch.equal(staged["w"], torch.arange(6.0))
    assert staged["w"].data_ptr() != sd["w"].data_ptr()
    assert torch.equal(staged["b"], torch.ones(2))
    assert staged["_extra_state"] is None
    assert staged["e"] is sd["e"]
